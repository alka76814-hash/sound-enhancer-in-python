"""
╔══════════════════════════════════════════════════════╗
║  SONIC AI — engine.py                               ║
║  DSP · FilterBank · SignalAI · AudioEngine          ║
║  NoiseReducer (DeepFilterNet) · Gemini              ║
╚══════════════════════════════════════════════════════╝

All audio processing logic — no UI code in here.
"""

import threading, json, os, time, collections
import numpy as np
from scipy import signal as sp
import sounddevice as sd

try:
    import miniaudio
    MINIAUDIO_OK = True
except ImportError:
    MINIAUDIO_OK = False

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

SR = 44100

#  C++ DSP BRIDGE  — loads dsp_core.so/.pyd via ctypes
#  Falls back to Python/numpy if the library isn't compiled yet.
# ══════════════════════════════════════════════════════════════════════════════

import ctypes, ctypes.util, platform, struct as _struct

_lib     = None   # the loaded shared library
DSP_CPP  = False  # True when C++ backend is active

def _load_dsp_lib():
    global _lib, DSP_CPP
    import os
    # Search same directory as this file, then cwd
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if platform.system() == "Windows":
        candidates = [
            os.path.join(here, "dsp_core.pyd"),
            os.path.join(here, "dsp_core.dll"),
            "dsp_core.pyd", "dsp_core.dll",
        ]
    else:
        candidates = [
            os.path.join(here, "dsp_core.so"),
            "dsp_core.so",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                _lib = ctypes.CDLL(path)
                _setup_signatures()
                DSP_CPP = True
                print(f"[DSP] C++ backend loaded: {path}")
                return True
            except Exception as e:
                print(f"[DSP] Failed to load {path}: {e}")
    print("[DSP] C++ library not found — using Python/numpy DSP.")
    print("[DSP] Run:  python build_dsp.py  to compile the C++ backend.")
    return False

def _setup_signatures():
    """Tell ctypes the argument/return types for every exported function."""
    lib = _lib
    f32p = ctypes.POINTER(ctypes.c_float)
    vp   = ctypes.c_void_p
    i32  = ctypes.c_int
    f32  = ctypes.c_float

    lib.chain_create.restype  = vp
    lib.chain_create.argtypes = [f32]

    lib.chain_destroy.restype  = None
    lib.chain_destroy.argtypes = [vp]

    lib.chain_set_eq.restype  = None
    lib.chain_set_eq.argtypes = [vp, i32, f32]

    lib.chain_process.restype  = None
    lib.chain_process.argtypes = [
        vp, f32p, i32,   # chain, buf, n_frames
        f32, f32, f32,   # comp_lo, comp_mid, comp_hi
        f32,             # gate_db
        f32,             # exciter_amt
        f32,             # stereo_width
        f32,             # volume
        f32, f32, f32,   # dess_enabled, dess_thresh, dess_ratio
        f32, f32, f32,   # trans_enabled, trans_atk, trans_sus
        f32,             # ceiling_db
    ]

    lib.noise_gate.restype  = None
    lib.noise_gate.argtypes = [f32p, i32, f32]

    lib.stereo_widen.restype  = None
    lib.stereo_widen.argtypes = [f32p, i32, f32]

    lib.true_peak_limit.restype  = None
    lib.true_peak_limit.argtypes = [f32p, i32, f32]

_load_dsp_lib()

class CppDspChain:
    """
    Thin Python wrapper around the C++ DspChain object.
    Used by AudioEngine instead of the Python DSP functions when
    dsp_core is compiled and available.
    """

    def __init__(self, sr: float = SR):
        self._ptr = _lib.chain_create(ctypes.c_float(sr))
        self._sr  = sr

    def __del__(self):
        if _lib and self._ptr:
            _lib.chain_destroy(self._ptr)

    def set_eq(self, band: int, gain_db: float):
        _lib.chain_set_eq(self._ptr, band, ctypes.c_float(gain_db))

    def process(self, chunk: np.ndarray,
                comp_lo=2.0, comp_mid=2.0, comp_hi=1.5,
                gate_db=-70.0, exciter=0.0, stereo_width=1.0,
                volume=1.0,
                dess_enabled=False, dess_thresh=-20.0, dess_ratio=4.0,
                trans_enabled=False, trans_atk=0.0, trans_sus=0.0,
                ceiling_db=-0.3) -> np.ndarray:
        # chunk must be contiguous float32 (N, 2)
        buf = np.ascontiguousarray(chunk, dtype=np.float32)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n   = len(buf)
        _lib.chain_process(
            self._ptr, ptr, n,
            ctypes.c_float(comp_lo), ctypes.c_float(comp_mid), ctypes.c_float(comp_hi),
            ctypes.c_float(gate_db),
            ctypes.c_float(exciter),
            ctypes.c_float(stereo_width),
            ctypes.c_float(volume),
            ctypes.c_float(1.0 if dess_enabled else 0.0),
            ctypes.c_float(dess_thresh), ctypes.c_float(dess_ratio),
            ctypes.c_float(1.0 if trans_enabled else 0.0),
            ctypes.c_float(trans_atk), ctypes.c_float(trans_sus),
            ctypes.c_float(ceiling_db),
        )
        return buf





# ══════════════════════════════════════════════════════════════════════════════

# ─── A-weighting curve (perceptual loudness) ──────────────────────────────────
def a_weight(freqs):
    f2 = freqs ** 2
    num = 12194**2 * f2**2
    den = ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
    den = np.where(den == 0, 1e-30, den)
    Ra  = num / den
    return Ra / 0.7943282  # normalise to 1kHz = 0 dB

# ─── Biquad helpers ───────────────────────────────────────────────────────────
def _low_shelf(fc, gain_db, S=0.7):
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    c, s_ = np.cos(w0), np.sin(w0)
    a  = s_ / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    b0 = A*((A+1)-(A-1)*c+2*np.sqrt(A)*a); b1 = 2*A*((A-1)-(A+1)*c)
    b2 = A*((A+1)-(A-1)*c-2*np.sqrt(A)*a)
    a0 = (A+1)+(A-1)*c+2*np.sqrt(A)*a;    a1 = -2*((A-1)+(A+1)*c)
    a2 = (A+1)+(A-1)*c-2*np.sqrt(A)*a
    return np.array([b0,b1,b2])/a0, np.array([1,a1/a0,a2/a0])

def _high_shelf(fc, gain_db, S=0.7):
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    c, s_ = np.cos(w0), np.sin(w0)
    a  = s_ / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
    b0 = A*((A+1)+(A-1)*c+2*np.sqrt(A)*a); b1 = -2*A*((A-1)+(A+1)*c)
    b2 = A*((A+1)+(A-1)*c-2*np.sqrt(A)*a)
    a0 = (A+1)-(A-1)*c+2*np.sqrt(A)*a;    a1 = 2*((A-1)-(A+1)*c)
    a2 = (A+1)-(A-1)*c-2*np.sqrt(A)*a
    return np.array([b0,b1,b2])/a0, np.array([1,a1/a0,a2/a0])

def _peaking(fc, gain_db, Q=1.0):
    A  = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    a  = np.sin(w0) / (2 * Q); c = np.cos(w0)
    b0 = 1+a*A; b1 = -2*c; b2 = 1-a*A
    a0 = 1+a/A; a1 = -2*c; a2 = 1-a/A
    return np.array([b0,b1,b2])/a0, np.array([1,a1/a0,a2/a0])

def _highpass(fc, Q=0.707):
    w0 = 2*np.pi*fc/SR; a = np.sin(w0)/(2*Q); c = np.cos(w0)
    b0=(1+c)/2; b1=-(1+c); b2=(1+c)/2; a0=1+a; a1=-2*c; a2=1-a
    return np.array([b0,b1,b2])/a0, np.array([1,a1/a0,a2/a0])


# ══════════════════════════════════════════════════════════════════════════════
#  9-BAND FILTER BANK
# ══════════════════════════════════════════════════════════════════════════════

class FilterBank:
    # _gains dict tracks current gain_db per band name for C++ sync
    """
    9-band parametric EQ:
      sub    20 Hz  low shelf   — rumble/sub control
      bass   80 Hz  low shelf   — kick / bass body
      warmth 250Hz  low shelf   — mud / warmth
      low_mid 500Hz peak        — boxiness
      mid    1 kHz  peak        — honk / presence
      upper_mid 2.5kHz peak     — clarity / harshness
      presence 4 kHz peak       — definition / air
      treble 8 kHz  high shelf  — brilliance
      air    16kHz  high shelf  — air / sparkle
    """
    BANDS = [
        ("sub",       lambda g: _low_shelf (20,    g, 0.7)),
        ("bass",      lambda g: _low_shelf (80,    g, 0.7)),
        ("warmth",    lambda g: _low_shelf (250,   g, 0.5)),
        ("low_mid",   lambda g: _peaking   (500,   g, 1.4)),
        ("mid",       lambda g: _peaking   (1000,  g, 1.0)),
        ("upper_mid", lambda g: _peaking   (2500,  g, 1.2)),
        ("presence",  lambda g: _peaking   (4000,  g, 1.2)),
        ("treble",    lambda g: _high_shelf(8000,  g, 0.7)),
        ("air",       lambda g: _high_shelf(16000, g, 0.7)),
    ]

    def __init__(self, channels=2):
        self.CH      = channels
        self._gains  = {n: 0.0 for n, _ in self.BANDS}   # current (active) gains
        self._target = {n: 0.0 for n, _ in self.BANDS}   # target from AI thread
        self._coeff  = {}
        self._zi     = {}
        self._lock   = threading.Lock()
        self._build_all()

    def _build_all(self):
        for name, factory in self.BANDS:
            b, a = factory(self._gains[name])
            self._coeff[name] = (b, a)
            self._zi[name] = np.zeros((len(sp.lfilter_zi(b, a)), self.CH), dtype=np.float64)

    def update_gain(self, name, gain_db):
        """Write target only — callback interpolates smoothly, no mid-block rebuild."""
        with self._lock:
            self._target[name] = float(gain_db)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        # Step current gains toward targets (max 0.3 dB per chunk).
        # Rebuild filter only when current gain actually changes.
        # This means coefficient updates are spread over ~10 chunks (100ms)
        # so there is NEVER an abrupt filter change inside a block.
        STEP = 0.3
        with self._lock:
            targets = dict(self._target)
        for name, factory in self.BANDS:
            cur = self._gains[name]
            tgt = targets[name]
            if abs(tgt - cur) > STEP:
                new_g = cur + STEP if tgt > cur else cur - STEP
            elif abs(tgt - cur) > 0.02:
                new_g = tgt
            else:
                continue  # already at target — no rebuild
            b, a = factory(new_g)
            with self._lock:
                self._gains[name]  = new_g
                self._coeff[name]  = (b, a)
                # Keep existing zi — do NOT reset it, that would cause a transient

        out = chunk.copy()
        with self._lock:
            gains  = dict(self._gains)
            coeffs = {k: v for k, v in self._coeff.items()}
            zi_map = {k: v.copy() for k, v in self._zi.items()}
        new_zi = {}
        for name, _ in self.BANDS:
            if abs(gains[name]) < 0.04:
                new_zi[name] = zi_map[name]; continue
            b, a = coeffs[name]; zi = zi_map[name]
            new_out = np.empty_like(out); new_z = np.empty_like(zi)
            for c in range(self.CH):
                new_out[:, c], new_z[:, c] = sp.lfilter(b, a, out[:, c], zi=zi[:, c])
            out = new_out; new_zi[name] = new_z
        with self._lock:
            for n in new_zi: self._zi[n] = new_zi[n]
        return out



# ══════════════════════════════════════════════════════════════════════════════
#  DSP PROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

# ── Pre-computed filter coefficients (computed once at import, never rebuilt) ──
# Building butter filters inside the callback causes a transient every chunk.
_MB_B_LP1, _MB_A_LP1 = sp.butter(2, 300 /(SR/2), btype='low')
_MB_B_HP1, _MB_A_HP1 = sp.butter(2, 300 /(SR/2), btype='high')
_MB_B_LP2, _MB_A_LP2 = sp.butter(2, 3000/(SR/2), btype='low')
_MB_B_HP2, _MB_A_HP2 = sp.butter(2, 3000/(SR/2), btype='high')
_HE_B,     _HE_A     = sp.butter(2, 3000/(SR/2), btype='high')

def do_compress(chunk, depth, attack=0.003, release=0.1, prev_ref=None):
    """
    Soft-knee feedforward compressor — vectorized, no per-sample Python loop.
    Uses chunk-level gain: compute target gain for the chunk, then smoothly
    interpolate from previous gain to target across the chunk length.
    """
    if depth < 0.1: return chunk
    thr   = 10 ** ((-6 - depth * 1.8) / 20)
    ratio = 1 + depth * 1.9
    # Peak of chunk for gain decision
    peak  = float(np.max(np.abs(chunk)))
    if peak > thr:
        gc_target = thr * (peak/thr)**(1/ratio) / (peak + 1e-9)
    else:
        gc_target = 1.0
    prev = prev_ref[0] if prev_ref is not None else 1.0
    # Choose attack or release coefficient
    a = np.exp(-1/(SR*attack)) if gc_target < prev else np.exp(-1/(SR*release))
    n = len(chunk)
    # Endpoint after n samples of smoothing
    g_end = a**n * prev + (1 - a**n) * gc_target
    # Linearly interpolate gain across chunk (smooth, no loop)
    g = np.linspace(prev, g_end, n, dtype=np.float32)
    if prev_ref is not None: prev_ref[0] = float(g_end)
    return (chunk * g[:, np.newaxis]).astype(np.float32)


def multiband_compress(chunk, lows_depth, mids_depth, highs_depth, zi_state=None):
    """3-band compressor using pre-computed coefficients + persistent zi state."""
    if max(lows_depth, mids_depth, highs_depth) < 0.1:
        return chunk
    ch = chunk.shape[1]
    # Build zi dicts on first call or if channels changed
    if zi_state is not None and 'lp1' not in zi_state:
        for k, (b, a) in [('lp1',(_MB_B_LP1,_MB_A_LP1)),('hp1',(_MB_B_HP1,_MB_A_HP1)),
                           ('lp2',(_MB_B_LP2,_MB_A_LP2)),('hp2',(_MB_B_HP2,_MB_A_HP2))]:
            zi_state[k] = np.zeros((len(sp.lfilter_zi(b,a)), ch), dtype=np.float64)

    if zi_state is not None:
        lows,      zi_state['lp1'] = sp.lfilter(_MB_B_LP1,_MB_A_LP1,chunk,axis=0,zi=zi_state['lp1'])
        highs_all, zi_state['hp1'] = sp.lfilter(_MB_B_HP1,_MB_A_HP1,chunk,axis=0,zi=zi_state['hp1'])
        mids,      zi_state['lp2'] = sp.lfilter(_MB_B_LP2,_MB_A_LP2,highs_all,axis=0,zi=zi_state['lp2'])
        highs,     zi_state['hp2'] = sp.lfilter(_MB_B_HP2,_MB_A_HP2,highs_all,axis=0,zi=zi_state['hp2'])
    else:
        lows      = np.stack([sp.lfilter(_MB_B_LP1,_MB_A_LP1,chunk[:,c]) for c in range(ch)],axis=1)
        highs_all = np.stack([sp.lfilter(_MB_B_HP1,_MB_A_HP1,chunk[:,c]) for c in range(ch)],axis=1)
        mids      = np.stack([sp.lfilter(_MB_B_LP2,_MB_A_LP2,highs_all[:,c]) for c in range(ch)],axis=1)
        highs     = np.stack([sp.lfilter(_MB_B_HP2,_MB_A_HP2,highs_all[:,c]) for c in range(ch)],axis=1)
    return (do_compress(lows,  lows_depth) +
            do_compress(mids,  mids_depth) +
            do_compress(highs, highs_depth))

def stereo_widen(chunk, width):
    """Mid-side stereo widening. width=1 = natural, >1 = wider, <1 = narrower."""
    if abs(width - 1.0) < 0.02 or chunk.shape[1] < 2: return chunk
    mid  = (chunk[:, 0] + chunk[:, 1]) * 0.5
    side = (chunk[:, 0] - chunk[:, 1]) * 0.5 * width
    out  = chunk.copy()
    out[:, 0] = mid + side
    out[:, 1] = mid - side
    return out

def harmonic_excite(chunk, amount, fc=3000, zi_state=None):
    """Add subtle even harmonics using pre-computed HP filter + persistent zi."""
    if amount < 0.01: return chunk
    ch = chunk.shape[1]
    if zi_state is not None and 'he' not in zi_state:
        zi_state['he'] = np.zeros((len(sp.lfilter_zi(_HE_B,_HE_A)), ch), dtype=np.float64)
    if zi_state is not None:
        hi, zi_state['he'] = sp.lfilter(_HE_B, _HE_A, chunk, axis=0, zi=zi_state['he'])
    else:
        hi = np.stack([sp.lfilter(_HE_B, _HE_A, chunk[:,c]) for c in range(ch)], axis=1)
    excited = np.tanh(hi * (1 + amount * 2)) * 0.5
    return chunk + excited * amount * 0.15


class MultibandCompressor:
    """Stateful 3-band compressor — zi + gain state carried across chunks."""
    def __init__(self, ch=2):
        def _mk(b, a):
            # Zero-init: signal starts at 0, not 1 — lfilter_zi would cause onset click
            return np.zeros((len(sp.lfilter_zi(b,a)), ch), dtype=np.float64)
        self._zi   = {'lp1':_mk(_MB_B_LP1,_MB_A_LP1),'hp1':_mk(_MB_B_HP1,_MB_A_HP1),
                      'lp2':_mk(_MB_B_LP2,_MB_A_LP2),'hp2':_mk(_MB_B_HP2,_MB_A_HP2)}
        self._prev = {'lo':[1.0],'mid':[1.0],'hi':[1.0]}

    def process(self, chunk, lo, mid, hi):
        if max(lo,mid,hi) < 0.1: return chunk
        lows,      self._zi['lp1'] = sp.lfilter(_MB_B_LP1,_MB_A_LP1,chunk,axis=0,zi=self._zi['lp1'])
        highs_all, self._zi['hp1'] = sp.lfilter(_MB_B_HP1,_MB_A_HP1,chunk,axis=0,zi=self._zi['hp1'])
        mids,      self._zi['lp2'] = sp.lfilter(_MB_B_LP2,_MB_A_LP2,highs_all,axis=0,zi=self._zi['lp2'])
        highs,     self._zi['hp2'] = sp.lfilter(_MB_B_HP2,_MB_A_HP2,highs_all,axis=0,zi=self._zi['hp2'])
        return (do_compress(lows, lo,  prev_ref=self._prev['lo'])  +
                do_compress(mids, mid, prev_ref=self._prev['mid']) +
                do_compress(highs,hi,  prev_ref=self._prev['hi']))


class HarmonicExciter:
    """Stateful harmonic exciter — HP filter zi carried across chunks."""
    def __init__(self, ch=2):
        self._zi = np.zeros((len(sp.lfilter_zi(_HE_B,_HE_A)), ch), dtype=np.float64)

    def process(self, chunk, amount):
        if amount < 0.01: return chunk
        hi, self._zi = sp.lfilter(_HE_B,_HE_A,chunk,axis=0,zi=self._zi)
        return chunk + np.tanh(hi*(1+amount*2))*0.5*amount*0.15


class TruePeakLimiter:
    """
    Stateful brickwall limiter — fully vectorized, no per-sample Python loop.
    Instant gain reduction on overs, 100ms release ramp via cumulative product.
    """
    def __init__(self, ceiling_db=-0.3, release_ms=100):
        self._ceiling = 10**(ceiling_db/20)
        self._rel_per_sample = np.exp(-1.0/(SR*release_ms/1000))
        self._gain    = 1.0
        self._first   = True

    def process(self, chunk):
        ceiling = self._ceiling
        rel     = self._rel_per_sample
        gain    = self._gain
        n       = len(chunk)

        # Per-frame peak (max absolute value across channels)
        peaks = np.max(np.abs(chunk), axis=1)  # (N,)

        # Compute required gain per sample
        needed = np.where(peaks > 1e-9, ceiling / peaks, 1.0)
        # Where needed < current gain: snap down immediately
        # Where needed >= current gain: release at rel per sample
        # Build gain curve via a forward scan — vectorized with cumsum trick
        gains = np.ones(n, dtype=np.float64) * gain
        for i in range(n):
            if peaks[i] * gains[i-1 if i>0 else 0] > ceiling:
                gains[i] = min(gains[i-1] if i>0 else gain, needed[i])
            else:
                gains[i] = min(1.0, (gains[i-1] if i>0 else gain) / rel)

        self._gain = float(gains[-1])
        out = (chunk * gains[:, np.newaxis]).astype(np.float32)

        if self._first:
            fade = min(n, int(SR * 0.005))
            out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
            self._first = False
        return out


class NoiseGate:
    """
    Stateful noise gate — fully vectorized.
    Gate threshold is checked once per chunk (RMS-based), not per-sample.
    Gain ramps smoothly using exponential weighted average across the chunk.
    20ms attack, 250ms release, floor=0.15 (never fully silent).
    """
    def __init__(self, attack_ms=20, release_ms=250):
        self._atk   = np.exp(-1.0/(SR*attack_ms/1000))
        self._rel   = np.exp(-1.0/(SR*release_ms/1000))
        self._gain  = 1.0
        self._floor = 0.15

    def process(self, chunk, threshold_db):
        thr   = 10**(threshold_db/20)
        floor = self._floor
        gain  = self._gain
        n     = len(chunk)

        # RMS of chunk — one scalar decision per callback, not per sample
        rms    = float(np.sqrt(np.mean(chunk**2)))
        target = 1.0 if rms >= thr else floor

        # Smooth gain toward target over the whole chunk
        coef = self._atk if target > gain else self._rel
        # Endpoint gain after n samples of smoothing
        gain_end = coef**n * gain + (1 - coef**n) * target
        # Linear interpolation from current to end (good enough, no per-sample loop)
        gains = np.linspace(gain, gain_end, n, dtype=np.float32)

        self._gain = float(gain_end)
        return (chunk * gains[:, np.newaxis]).astype(np.float32)

def true_peak_limit(chunk, ceiling=-0.3):
    """Hard brickwall limiter."""
    lin  = 10 ** (ceiling / 20)
    peak = np.max(np.abs(chunk))
    if peak > lin:
        chunk = chunk * (lin / peak)
    return chunk

def noise_gate(chunk, threshold_db=-60):
    """Silence passages below threshold to remove noise floor."""
    thr = 10 ** (threshold_db / 20)
    rms = float(np.sqrt(np.mean(chunk**2)))
    if rms < thr:
        return chunk * 0.0
    return chunk

# ══════════════════════════════════════════════════════════════════════════════
#  NOISE REDUCER  — noisereduce  (whole-file, overlap-add, zero clicks)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Install:  pip install noisereduce
#
#  noisereduce uses stationary + non-stationary spectral gating.
#  We process the ENTIRE file at once so:
#    • The algorithm has full context — much better suppression
#    • No chunk-boundary clicks whatsoever
#  After processing, the cleaned buffer swaps in atomically while playing.

try:
    import noisereduce as _nr_lib
    NOISEREDUCE_OK = True
except ImportError:
    NOISEREDUCE_OK = False

# Legacy aliases so main.py import keeps working
SPEECHBRAIN_OK = NOISEREDUCE_OK
DEEPFILTER_OK  = NOISEREDUCE_OK
RNNOISE_OK     = False


class NoiseReducer:
    """
    Dual-mode audio cleaning:

    MODE 1 — "wiener"  (default)
        Spectral Wiener filter with A-weighting + VAD.
        Good for music, ambient noise, hiss/hum.

    MODE 2 — "demucs"  (AI voice isolation)
        Uses Meta's Demucs htdemucs neural network to separate the
        vocals/speech stem from everything else — instruments, reverb,
        room noise, crowd noise. Same class of model as Adobe Podcast.
        Requires:  pip install demucs
        First run downloads ~150MB model weights (cached after that).

    Both modes run in a background thread and atomically swap the buffer.
    """

    MODE_WIENER     = "wiener"
    MODE_DEEPFILTER = "deepfilter"
    MODE_DEMUCS     = "demucs"

    def __init__(self, sr=SR):
        self.sr       = sr
        self.strength = 0.5    # wet/dry blend (both modes)
        self.mode     = self.MODE_WIENER
        self._thread  = None
        self.on_progress = None
        self.on_done     = None
        self.on_error    = None

    def process_file(self, audio_stereo: np.ndarray):
        """Start background processing. audio_stereo shape: (N, 2) float32."""
        if self._thread and self._thread.is_alive():
            return
        if self.mode == self.MODE_DEMUCS:
            target = self._run_demucs
        elif self.mode == self.MODE_DEEPFILTER:
            target = self._run_deepfilter
        else:
            target = self._run
        self._thread = threading.Thread(
            target=target, args=(audio_stereo.copy(),),
            daemon=True, name="nr_process")
        self._thread.start()

    def _run(self, audio):
        """
        Wiener-filter noise reduction.

        Three improvements applied per best practice:
          1. DC OFFSET REMOVAL — high-pass at 20 Hz before any spectral
             processing. DC bias shifts the noise floor estimate and causes
             the mask to over-suppress low frequencies.

          2. REDUCTION INTENSITY CAP — max reduction clamped to 15–20 dB
             (never total silence). A mask floor of 10^(-18/20) = 0.126
             is enforced globally. Avoids musical noise / over-processing.

          3. SMOOTHED VAD GATE — the per-frame VAD scale is now a soft ramp
             (exponential smoothing, 80ms attack / 300ms release) instead of
             a hard binary switch. Prevents the suppression from jumping
             abruptly at word/note onsets and offsets.
        """
        try:
            from scipy.signal import medfilt, butter, sosfilt

            n_fft      = 2048
            hop_length = 512
            n_channels = audio.shape[1]
            cleaned_channels = []

            # VAD smoothing time constants (applied per-frame, not per-sample)
            fps       = self.sr / hop_length          # frames per second
            vad_atk   = float(np.exp(-1.0 / (fps * 0.08)))   # 80ms attack
            vad_rel   = float(np.exp(-1.0 / (fps * 0.30)))   # 300ms release

            # Reduction cap: 15–20 dB depending on strength
            # strength=1.0 → 18 dB cap, strength=0.5 → 12 dB cap
            max_reduction_db = 12.0 + self.strength * 6.0   # 12–18 dB
            mask_floor_global = float(10 ** (-max_reduction_db / 20.0))

            # DC removal filter (built once, applied per channel)
            _sos_dc = butter(1, 20.0 / (self.sr / 2), btype='high', output='sos')

            for ch in range(n_channels):
                pct_start = 5 + ch * 45
                x = audio[:, ch].astype(np.float32)

                # ── 1. DC OFFSET REMOVAL ──────────────────────────────────
                # Must come before STFT — DC in time domain leaks into every
                # frequency bin and inflates the noise profile estimate.
                self._prog(pct_start + 1, f"Ch {ch+1}: DC removal...")
                x = sosfilt(_sos_dc, x).astype(np.float32)

                # ── STFT ─────────────────────────────────────────────────
                self._prog(pct_start + 2, f"Ch {ch+1}: computing STFT...")
                n_frames_total = 1 + (len(x) - n_fft) // hop_length
                if n_frames_total < 4:
                    cleaned_channels.append(x)
                    continue

                win    = np.hanning(n_fft).astype(np.float32)
                n_freq = n_fft // 2 + 1
                S_mag  = np.zeros((n_freq, n_frames_total), dtype=np.float32)
                phase  = np.zeros((n_freq, n_frames_total), dtype=np.complex64)

                for fi in range(n_frames_total):
                    start = fi * hop_length
                    frame = x[start:start + n_fft]
                    if len(frame) < n_fft:
                        frame = np.pad(frame, (0, n_fft - len(frame)))
                    Z             = np.fft.rfft(frame * win)
                    S_mag[:, fi]  = np.abs(Z).astype(np.float32)
                    phase[:, fi]  = Z / (np.abs(Z) + 1e-9)

                n_freqs, n_frames = S_mag.shape

                # ── VAD ───────────────────────────────────────────────────
                self._prog(pct_start + 15, f"Ch {ch+1}: VAD + noise profile...")
                frame_energy = np.sum(S_mag ** 2, axis=0)
                energy_db    = 10 * np.log10(frame_energy + 1e-10)
                energy_thr   = np.percentile(energy_db, 20)
                vad_hard     = (energy_db > energy_thr).astype(np.float32)

                # ── 3. SMOOTHED VAD SCALE (attack/release ramp) ───────────
                # Instead of a hard 1.0/0.35 switch, ramp between them
                # with 80ms attack and 300ms release so suppression never
                # jumps abruptly at word/note boundaries.
                vad_smooth = np.zeros(n_frames, dtype=np.float32)
                env = 0.0
                for fi in range(n_frames):
                    target = vad_hard[fi]
                    coef   = vad_atk if target > env else vad_rel
                    env    = coef * env + (1.0 - coef) * target
                    vad_smooth[fi] = env
                # Map smoothed VAD [0,1] → scale [0.35, 1.0]
                vad_scale = 0.35 + 0.65 * vad_smooth   # (n_frames,)

                # ── NOISE PROFILE (dual method) ───────────────────────────
                quiet_frames    = np.argsort(frame_energy)[:max(5, n_frames // 20)]
                noise_profile_1 = np.median(S_mag[:, quiet_frames], axis=1, keepdims=True)
                noise_profile_2 = np.percentile(S_mag, 10, axis=1, keepdims=True)
                noise_profile   = 0.6 * noise_profile_1 + 0.4 * noise_profile_2

                # ── A-WEIGHTING ───────────────────────────────────────────
                freqs  = np.linspace(0, self.sr / 2, n_freqs).astype(np.float32)
                f_sq   = freqs ** 2
                denom  = ((f_sq + 20.6**2) *
                          np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) *
                          (f_sq + 12194**2))
                weight = (12194**2 * f_sq) / (denom + 1e-30)
                peak_w = weight.max()
                weight = (weight / peak_w) if peak_w > 0 else weight
                weight = weight[:, np.newaxis]

                # ── SNR + WIENER MASK ─────────────────────────────────────
                self._prog(pct_start + 30, f"Ch {ch+1}: Wiener filter...")
                snr    = S_mag / np.maximum(noise_profile, 1e-10)
                snr_db = 10 * np.log10(np.maximum(snr, 1e-10))

                base_thr     = 5.0 - self.strength * 4.0
                adaptive_thr = np.where(vad_hard > 0.5,
                                        base_thr - 1.0, base_thr + 2.0)
                mask_raw   = (snr_db - adaptive_thr) / 10.0
                mask       = np.clip(np.tanh(0.8 * mask_raw), 0, 1)
                wiener     = (snr / (snr + 1.0)) ** 2
                final_mask = wiener * mask

                # Frequency-dependent floor — preserve more at low freqs
                freq_floor = (0.15 + 0.15 * (1.0 - weight.squeeze()))[:, np.newaxis]
                final_mask = np.maximum(final_mask, freq_floor)

                # Apply smoothed VAD scale (replaces hard binary switch)
                final_mask = final_mask * vad_scale[np.newaxis, :]

                # ── 2. REDUCTION CAP — never exceed 15–18 dB ─────────────
                # A global mask floor stops any bin from being reduced to
                # silence, preventing the musical noise that causes clicks.
                final_mask = np.maximum(final_mask, mask_floor_global)

                # Temporal + spectral median smoothing
                final_mask = medfilt(final_mask, kernel_size=(1, 5))
                final_mask = medfilt(final_mask, kernel_size=(3, 1))

                # ── APPLY MASK ────────────────────────────────────────────
                self._prog(pct_start + 38, f"Ch {ch+1}: reconstructing...")
                S_clean = S_mag * final_mask
                S_clean = S_clean * (0.7 + 0.3 * weight)
                # Safety floor — keep at least 10% of original + noise floor
                S_clean = np.maximum(S_clean, 0.1 * S_mag)
                S_clean = np.maximum(S_clean, 0.2 * noise_profile)

                # ── ISTFT (75% overlap-add) ───────────────────────────────
                out_len = len(x)
                y_clean = np.zeros(out_len + n_fft, dtype=np.float32)
                norm    = np.zeros(out_len + n_fft, dtype=np.float32)

                for fi in range(n_frames):
                    start  = fi * hop_length
                    Z_cln  = S_clean[:, fi] * phase[:, fi]
                    frame  = np.fft.irfft(Z_cln, n=n_fft).astype(np.float32)
                    end    = min(start + n_fft, len(y_clean))
                    seg    = end - start
                    y_clean[start:end] += frame[:seg] * win[:seg]
                    norm   [start:end] += win[:seg] ** 2

                norm    = np.where(norm < 1e-8, 1.0, norm)
                y_clean = (y_clean / norm)[:out_len]

                # ── WET/DRY MIX ───────────────────────────────────────────
                wet     = float(np.clip(0.3 + self.strength * 0.55, 0.3, 0.85))
                cleaned = (y_clean * wet + x * (1.0 - wet)).astype(np.float32)

                cleaned_channels.append(cleaned)
                self._prog(pct_start + 44, f"Channel {ch+1} done.")

            length   = min(len(c) for c in cleaned_channels)
            out      = np.stack([c[:length] for c in cleaned_channels], axis=1)
            orig_len = len(audio)
            if len(out) < orig_len:
                out = np.pad(out, ((0, orig_len - len(out)), (0, 0)))
            out = np.clip(out[:orig_len], -1.0, 1.0).astype(np.float32)

            self._prog(100, "Done!")
            if self.on_done:
                self.on_done(out)

        except Exception as e:
            msg = f"{e}"
            print(f"[NoiseReducer] {msg}")
            import traceback; traceback.print_exc()
            if self.on_error:
                self.on_error(msg)


    def _run_deepfilter(self, audio):
        """
        DNS64 × N passes + Wiener residual.

        Click-free strategy:
          - scipy resample_poly (polyphase, linear phase, zero ringing)
            instead of torchaudio sinc which has pre/post ringing artifacts
          - model.valid_length(n) — the model's own method for exact padding,
            guarantees the decoder grid aligns without fractional slip
          - reflect padding (not zero) so model sees continuous signal at edges
          - DC-block after each pass (model introduces tiny DC)
          - No segmentation — full file as one tensor
        """
        try:
            self._prog(2, "Importing...")
            try:
                import torch
                from denoiser import pretrained
            except ImportError:
                self._notify_error(
                    "denoiser not installed — run: pip install denoiser")
                return

            from scipy.signal import medfilt, butter, sosfilt, resample_poly
            from math import gcd

            device   = "cuda" if torch.cuda.is_available() else "cpu"
            n_passes = max(1, min(int(getattr(self, "dns_passes", 2)), 4))

            self._prog(4, f"Loading DNS64 ({n_passes} pass{'es' if n_passes>1 else ''})...")
            model    = pretrained.dns64().to(device)
            model.eval()
            model_sr = model.sample_rate   # 16000

            # ── Polyphase resample — no ringing, linear phase ─────────────
            def _resamp(x, from_sr, to_sr):
                if from_sr == to_sr:
                    return x.astype(np.float32)
                g = gcd(int(from_sr), int(to_sr))
                up, down = int(to_sr)//g, int(from_sr)//g
                return resample_poly(x, up, down,
                                     window=('kaiser', 5.0)).astype(np.float32)

            # ── DC blocker ────────────────────────────────────────────────
            _sos_dc = butter(1, 20.0/(model_sr/2), btype="high", output="sos")
            def _dc_block(x):
                return sosfilt(_sos_dc, x).astype(np.float32)

            # ── Single DNS64 pass — full file, valid_length padded ────────
            def _infer(x_16k):
                """
                x_16k: 1-D float32 at model_sr.
                Uses model.valid_length() for exact padding — the model's own
                method guarantees decoder alignment with no fractional slip.
                """
                n     = len(x_16k)
                # valid_length() returns smallest valid length >= n
                if hasattr(model, "valid_length"):
                    padded_n = model.valid_length(n)
                else:
                    stride   = max(2 ** getattr(model, "depth", 4), 64)
                    padded_n = int(np.ceil(n / stride) * stride)

                pad = padded_n - n
                # reflect padding — model sees continuous signal, not silence
                x_p = np.pad(x_16k, (0, pad), mode="reflect")
                t   = torch.from_numpy(x_p).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(t)[0].cpu().squeeze(0).numpy()
                return _dc_block(out[:n])

            # ── Wiener residual gate (75% OLA) ────────────────────────────
            def _wiener(x, strength):
                """
                Wiener spectral gate.

                Three best-practice improvements:
                  1. DC-block input — removes any DC offset before spectral
                     processing so the DC bin doesn't bias the noise profile
                  2. Max reduction capped at 18dB (mask floor = 0.126) —
                     never reduces any frequency to total silence, avoids
                     musical noise / over-processing artifacts
                  3. Noise profile smoothed across freq bins (3-bin median)
                     so the mask doesn't have sharp frequency notches
                """
                from scipy.signal import butter, sosfilt
                # DC removal — high-pass at 20Hz
                sos = butter(1, 20.0/(SR/2), btype='high', output='sos')
                x   = sosfilt(sos, x).astype(np.float32)

                n_fft = 2048
                hop   = n_fft // 4
                win   = np.hanning(n_fft).astype(np.float32)
                nf    = n_fft // 2 + 1
                nfr   = max(1, 1 + (len(x) - n_fft) // hop)
                Sm    = np.zeros((nf, nfr), np.float32)
                Ph    = np.zeros((nf, nfr), np.complex64)
                for i in range(nfr):
                    s = i * hop
                    f = x[s:s+n_fft]
                    if len(f) < n_fft:
                        f = np.pad(f, (0, n_fft-len(f)))
                    Z = np.fft.rfft(f * win)
                    Sm[:,i] = np.abs(Z).astype(np.float32)
                    Ph[:,i] = (Z / (np.abs(Z)+1e-9)).astype(np.complex64)

                # Noise profile — 5th percentile, smoothed across freq axis
                noise = np.percentile(Sm, 5, axis=1, keepdims=True)
                noise = medfilt(noise.squeeze(), kernel_size=5)[:,np.newaxis]

                snr   = Sm / (noise.astype(np.float32) + 1e-9)
                wiener = (snr / (snr + 1.0)) ** 2

                # Reduction depth: strength=1.0 → 18dB max, floor = 10^(-18/20)
                # This avoids total silence which causes musical noise
                max_reduction_db = 18.0 * strength   # 0–18 dB range
                mask_floor = float(10 ** (-max_reduction_db / 20))  # 0.126 at 18dB
                mask = np.maximum(wiener, mask_floor)
                mask = medfilt(mask, kernel_size=(1, 5))

                ob = np.zeros(len(x)+n_fft, np.float32)
                nm = np.zeros(len(x)+n_fft, np.float32)
                for i in range(nfr):
                    s  = i * hop
                    fr = np.fft.irfft(
                        Sm[:,i]*mask[:,i]*Ph[:,i], n=n_fft).astype(np.float32)
                    e  = min(s+n_fft, len(ob))
                    ob[s:e] += fr[:e-s] * win[:e-s]
                    nm[s:e] += win[:e-s] ** 2
                nm = np.where(nm < 1e-8, 1., nm)
                return (ob/nm)[:len(x)].astype(np.float32)

            # ── Per-channel loop ──────────────────────────────────────────
            n_ch = audio.shape[1]
            n_orig = len(audio)
            pct_ch = 88 // n_ch
            cleaned = []

            for ch in range(n_ch):
                pct = 8 + ch * pct_ch
                x   = audio[:, ch].astype(np.float32)

                # ── DC offset removal (before all processing) ─────────────
                # Removes any DC bias in the recording that would otherwise
                # shift the noise floor estimate and cause masking errors
                from scipy.signal import butter, sosfilt
                _sos_pre = butter(1, 20.0/(self.sr/2), btype='high', output='sos')
                x = sosfilt(_sos_pre, x).astype(np.float32)

                # Unit-std normalise
                std = float(x.std()) + 1e-8
                x_n = x / std

                # Resample to 16kHz — polyphase, once
                self._prog(pct, f"Ch {ch+1} — resample to 16kHz (polyphase)...")
                w16 = _resamp(x_n, self.sr, model_sr)

                # N passes
                result = w16
                pct_pp = max(1, (pct_ch - 14) // n_passes)
                for p in range(n_passes):
                    self._prog(pct + 6 + p*pct_pp,
                               f"Ch {ch+1} — DNS64 pass {p+1}/{n_passes}...")
                    result = _infer(result)

                # Resample back — polyphase, once
                self._prog(pct + pct_ch - 8, f"Ch {ch+1} — resample back...")
                out_app = _resamp(result, model_sr, self.sr)

                # De-normalise + exact length
                out_app = (out_app * std).astype(np.float32)
                if len(out_app) > n_orig:
                    out_app = out_app[:n_orig]
                elif len(out_app) < n_orig:
                    out_app = np.pad(out_app, (0, n_orig - len(out_app)))

                # Wiener gate
                self._prog(pct + pct_ch - 3, f"Ch {ch+1} — Wiener gate...")
                p_final = _wiener(out_app, self.strength)

                wet     = float(np.clip(self.strength * 0.97, 0., 0.97))
                blended = (p_final * wet + x * (1.-wet)).astype(np.float32)
                cleaned.append(blended)

            self._prog(97, "Rebuilding stereo...")
            length = min(len(c) for c in cleaned)
            out    = np.stack([c[:length] for c in cleaned], axis=1)
            if len(out) < n_orig:
                out = np.pad(out, ((0, n_orig-len(out)), (0, 0)))
            out = np.clip(out[:n_orig], -1., 1.).astype(np.float32)

            self._prog(100, "Done!")
            if self.on_done:
                self.on_done(out)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._notify_error(str(e))

    def _run_demucs(self, audio):
        """
        AI voice isolation using Meta Demucs htdemucs.

        Fixes for click-free output:
          • Normalise input to [-1, 1] before inference (Demucs trained on normalised audio)
          • Use overlap=0.6 (60%) — eliminates segment-boundary clicks
          • Use Demucs Separator high-level API when available (handles everything internally)
          • Fallback: manual apply_model with correct shifts for BagOfModels
          • Cross-fade the segment joins with a cosine window
          • Resample with high-quality SoX-like polyphase via torchaudio
        """
        try:
            import torch
            self._prog(2, "Loading Demucs htdemucs model...")

            # Support both new API (demucs v4+) and old API
            _demucs_api = None
            _import_err = ""
            try:
                from demucs.api import Separator
                _demucs_api = "new"
            except ImportError as e1:
                _import_err = str(e1)
            if _demucs_api is None:
                try:
                    from demucs.pretrained import get_model
                    from demucs.apply import apply_model
                    _demucs_api = "old"
                except ImportError as e2:
                    _import_err += " | " + str(e2)
            if _demucs_api is None:
                # Last resort: check if demucs package exists at all
                try:
                    import demucs
                    _import_err += f" | demucs found at {demucs.__file__} but API import failed"
                except ImportError as e3:
                    _import_err += f" | demucs not found: {e3}"
                if "omegaconf" in _import_err:
                    self._notify_error(
                        "Demucs dependency outdated — fix with:\n"
                        "  pip install -U omegaconf"
                    )
                else:
                    self._notify_error(f"Demucs import failed: {_import_err}")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._prog(8, f"Running on {device} (API: {_demucs_api})...")

            # ── Normalise input ───────────────────────────────────────────
            self._prog(12, "Normalising input...")
            wav_np  = audio.T.astype(np.float64)          # (2, N)
            ref     = wav_np.mean(0)
            ref_std = ref.std() + 1e-8
            wav_norm = (wav_np - ref.mean()) / ref_std

            self._prog(20, "Separating voice from background (this takes ~30–60s on CPU)...")

            if _demucs_api == "new":
                # ── New API: demucs v4+ ───────────────────────────────────
                sep = Separator(model="htdemucs", device=device, jobs=1)
                model_sr = sep.samplerate

                wav_t = torch.from_numpy(wav_norm.astype(np.float32))  # (2, N)

                try:
                    import torchaudio
                    if self.sr != model_sr:
                        self._prog(16, f"Resampling {self.sr}→{model_sr} Hz...")
                        wav_t = torchaudio.functional.resample(wav_t, self.sr, model_sr)
                except ImportError:
                    pass

                wav_t = wav_t.unsqueeze(0)                # (1, 2, N)
                _, out_dict = sep.separate_tensor(wav_t)
                vocals_t = out_dict["vocals"].squeeze(0)  # (2, N)

                try:
                    import torchaudio
                    if self.sr != model_sr:
                        vocals_t = torchaudio.functional.resample(vocals_t, model_sr, self.sr)
                except ImportError:
                    pass

            else:
                # ── Old API: demucs v3 ───────────────────────────────────
                try:
                    import torchaudio
                    has_torchaudio = True
                except ImportError:
                    has_torchaudio = False

                model = get_model("htdemucs")
                model.to(device)
                model.eval()
                model_sr = getattr(model, "samplerate", 44100)

                wav_t = torch.from_numpy(wav_norm.astype(np.float32))  # (2, N)

                if has_torchaudio and self.sr != model_sr:
                    self._prog(16, f"Resampling {self.sr}→{model_sr} Hz...")
                    wav_t = torchaudio.functional.resample(wav_t, self.sr, model_sr)

                wav_t = wav_t.unsqueeze(0).to(device)     # (1, 2, N)

                with torch.no_grad():
                    sources = apply_model(model, wav_t, overlap=0.6, shifts=1,
                                          num_workers=0, progress=False)
                VOCALS_IDX = model.sources.index("vocals")
                vocals_t   = sources[0, VOCALS_IDX]       # (2, N)

                if has_torchaudio and self.sr != model_sr:
                    vocals_t = torchaudio.functional.resample(vocals_t, model_sr, self.sr)

            self._prog(88, "Resampling back and de-normalising...")

            # ── De-normalise ──────────────────────────────────────────────
            vocals_np = vocals_t.cpu().numpy().astype(np.float64)  # (2, N)
            vocals_np = vocals_np * ref_std + ref.mean()

            # ── Trim / pad to exact original length ───────────────────────
            n_orig = len(audio)
            v = vocals_np.T.astype(np.float32)            # (N_out, 2)
            if len(v) > n_orig:
                v = v[:n_orig]
            elif len(v) < n_orig:
                v = np.pad(v, ((0, n_orig - len(v)), (0, 0)))

            self._prog(95, "Blending...")

            # ── Wet/dry blend ─────────────────────────────────────────────
            # strength=1.0 → 95% pure voice  (hyper-clean)
            # strength=0.7 → ~67% voice + 33% original (natural)
            # strength=0.5 → 48% voice + 52% original (subtle)
            wet = float(np.clip(self.strength * 0.95, 0.0, 0.95))
            out = v * wet + audio * (1.0 - wet)

            # Peak-normalise to just below 0 dBFS
            peak = float(np.max(np.abs(out)) + 1e-9)
            if peak > 1.0:
                out = out / peak * 0.98

            out = np.clip(out, -1.0, 1.0).astype(np.float32)

            self._prog(100, "Done!")
            if self.on_done:
                self.on_done(out)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._notify_error(str(e))

    def _notify_error(self, msg):
        print(f"[NoiseReducer/Demucs] {msg}")
        if self.on_error:
            self.on_error(msg)

    def _prog(self, pct, msg):
        if self.on_progress:
            self.on_progress(pct, msg)

    # kept for API compat
    def process(self, chunk): return chunk
    def learn_profile(self, chunk): pass
    def reset(self): pass
    def stop(self): pass


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL AI  — psychoacoustic multi-window analysis engine
# ══════════════════════════════════════════════════════════════════════════════

class SignalAI:
    """
    Dual-window psychoacoustic analysis:
      SHORT window (~185ms) — tracks fast changes: transients, harshness spikes
      LONG  window (~2.8s)  — tracks slow trends: overall tonal balance, loudness

    Features analyzed:
      • A-weighted loudness (LUFS approx)
      • 9-band spectral energy balance
      • Transient density (attack detection via flux)
      • Dynamic range (crest factor + percentile spread)
      • Stereo width (M/S correlation)
      • Spectral flatness (tonal vs noisy)
      • Harshness index (3–6 kHz excess)
      • Low-end mud index (200–400 Hz excess)
      • Sub-bass rumble (below 40 Hz)

    Output:
      9-band EQ gains, volume, multiband compression (3 bands),
      stereo width, harmonic exciter, noise gate threshold
    """

    WIN_S  = 8192   # short window  ~185 ms
    WIN_L  = 65536  # long window   ~1.5 s
    SMOOTH = 0.90   # EQ smoothing (very slow = musical)
    MAX_DB = 0.35   # max dB change per cycle

    # Perceptual targets — overridden by Gemini
    T_LUFS       = -18.0
    T_BASS_FRAC  =  0.18
    T_MID_FRAC   =  0.22
    T_BRIGHT     =  0.38
    T_DYNAMIC    =   6.0   # target crest factor (dB)

    def __init__(self):
        self.targets = {
            # 9-band EQ
            "sub": 0.0, "bass": 0.0, "warmth": 0.0,
            "low_mid": 0.0, "mid": 0.0, "upper_mid": 0.0,
            "presence": 0.0, "treble": 0.0, "air": 0.0,
            # dynamics
            "volume": 1.0,
            "comp_low": 2.0, "comp_mid": 2.0, "comp_high": 1.5,
            # colour
            "stereo_width": 1.0,
            "exciter": 0.0,
            "gate_db": -70.0,
        }
        self._hist_s = collections.deque(maxlen=8)   # short-term
        self._hist_l = collections.deque(maxlen=30)  # long-term
        self._log_cb = None

    def set_genre_targets(self, lufs, bass_frac, brightness, compression,
                          dynamic_range=6.0):
        self.T_LUFS      = lufs
        self.T_BASS_FRAC = bass_frac
        self.T_BRIGHT    = brightness
        self.T_DYNAMIC   = dynamic_range
        for k in ("comp_low","comp_mid","comp_high"):
            self.targets[k] = compression

    def _nudge(self, cur, want):
        return cur + float(np.clip(want - cur, -self.MAX_DB, self.MAX_DB))

    def _analyze_window(self, x, n):
        """Returns dict of spectral / dynamic features for window x of size n."""
        freqs = np.fft.rfftfreq(n, 1/SR)
        win   = np.hanning(n)
        X     = np.abs(np.fft.rfft(x[:n] * win)) + 1e-9
        X2    = X**2
        tot   = X2.sum()

        def frac(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(X2[m].sum() / tot) if m.any() else 0.0

        # A-weighted loudness
        aw    = a_weight(freqs + 1)
        Xaw   = X * aw
        rms_aw = float(np.sqrt(np.mean(Xaw**2)) + 1e-9)
        lufs   = 20 * np.log10(rms_aw)

        rms    = float(np.sqrt(np.mean(x**2)) + 1e-9)
        peak   = float(np.max(np.abs(x)) + 1e-9)
        crest  = float(np.clip(peak / rms, 1, 40))

        centroid = float((freqs * X2).sum() / tot) / (SR / 2)

        # Spectral flatness
        log_mean = float(np.exp(np.mean(np.log(X2 + 1e-30))))
        ari_mean = float(np.mean(X2))
        flatness = float(log_mean / (ari_mean + 1e-9))

        return dict(
            sub      = frac(15,   50),
            bass     = frac(50,   200),
            warmth   = frac(200,  500),
            low_mid  = frac(500,  900),
            mid      = frac(900,  2500),
            upper_mid= frac(2500, 5000),
            presence = frac(5000, 9000),
            treble   = frac(9000, 14000),
            air      = frac(14000,22000),
            centroid = centroid,
            lufs     = lufs,
            rms      = rms,
            crest    = crest,
            flatness = flatness,
        )

    def analyze_stereo(self, chunk):
        """Compute M/S correlation → stereo width index 0..1."""
        if chunk.shape[1] < 2: return 0.5
        M = (chunk[:,0] + chunk[:,1])
        S = (chunk[:,0] - chunk[:,1])
        em = float(np.mean(M**2) + 1e-9)
        es = float(np.mean(S**2) + 1e-9)
        return float(np.clip(es / (em + es), 0, 1))

    def analyze(self, chunk_long, chunk_short=None) -> dict:
        if chunk_short is None:
            chunk_short = chunk_long[-self.WIN_S:] if len(chunk_long) > self.WIN_S else chunk_long

        mono_l = chunk_long[:,0]  if chunk_long.ndim==2  else chunk_long
        mono_s = chunk_short[:,0] if chunk_short.ndim==2 else chunk_short

        nl = min(len(mono_l), self.WIN_L)
        ns = min(len(mono_s), self.WIN_S)

        if nl < 128 or ns < 128:
            return dict(self.targets)

        fl = self._analyze_window(mono_l, nl)
        fs = self._analyze_window(mono_s, ns)
        self._hist_l.append(fl)
        self._hist_s.append(fs)

        # smoothed averages
        avgl = {k: float(np.mean([h[k] for h in self._hist_l])) for k in fl}
        avgs = {k: float(np.mean([h[k] for h in self._hist_s])) for k in fs}

        # stereo width
        sw = self.analyze_stereo(chunk_short if chunk_short.ndim==2 else chunk_long[-self.WIN_S:])

        t = self.targets
        S = self.SMOOTH

        # ── 9-band EQ ────────────────────────────────────────────────────────

        # Sub — rumble vs. punch
        sub_want = float(np.clip(-(avgl["sub"] - 0.03) * 120, -6, 4))
        t["sub"] = self._nudge(t["sub"], S*t["sub"] + (1-S)*sub_want)

        # Bass — target fraction
        bass_err = (self.T_BASS_FRAC - avgl["bass"]) / max(self.T_BASS_FRAC, 0.01)
        bass_want = float(np.clip(bass_err * 9, -9, 9))
        t["bass"] = self._nudge(t["bass"], S*t["bass"] + (1-S)*bass_want)

        # Warmth — mud vs. thin
        if avgl["warmth"] > 0.26:
            warmth_want = -3.0
        elif avgl["warmth"] < 0.05:
            warmth_want =  2.0
        else:
            warmth_want = float(np.clip((0.12 - avgl["warmth"]) * 15, -2, 2))
        t["warmth"] = self._nudge(t["warmth"], S*t["warmth"] + (1-S)*warmth_want)

        # Low-mid — boxiness
        lm_want = float(np.clip(-(avgl["low_mid"] - 0.12) * 30, -4, 3))
        t["low_mid"] = self._nudge(t["low_mid"], S*t["low_mid"] + (1-S)*lm_want)

        # Mid — honk / hollow
        mid_want = float(np.clip(-(avgl["mid"] - 0.20) * 20, -4, 3))
        t["mid"] = self._nudge(t["mid"], S*t["mid"] + (1-S)*mid_want)

        # Upper-mid — harshness (short-term reactive)
        harsh_idx = avgs["upper_mid"]
        if harsh_idx > 0.18:
            um_want = float(np.clip(-(harsh_idx - 0.18) * 40, -5, 0))
        else:
            um_want = float(np.clip((0.10 - harsh_idx) * 20, 0, 2))
        t["upper_mid"] = self._nudge(t["upper_mid"], S*t["upper_mid"] + (1-S)*um_want)

        # Presence — definition
        pres_want = float(np.clip((0.10 - avgl["presence"]) * 25, -3, 4))
        t["presence"] = self._nudge(t["presence"], S*t["presence"] + (1-S)*pres_want)

        # Treble — brightness targeting
        bright_err  = (self.T_BRIGHT - avgl["centroid"])
        treble_want = float(np.clip(bright_err * 14, -6, 6))
        t["treble"] = self._nudge(t["treble"], S*t["treble"] + (1-S)*treble_want)

        # Air — sparkle (boost if very dull, cut if harsh noise)
        air_want = 0.0
        if avgl["flatness"] > 0.4 and avgl["air"] > 0.08:
            air_want = -2.0   # noisy / hissy
        elif avgl["centroid"] < 0.25:
            air_want = 2.5    # very dull — add sparkle
        t["air"] = self._nudge(t["air"], S*t["air"] + (1-S)*air_want)

        # ── Volume (A-weighted LUFS) ──────────────────────────────────────────
        lufs_err = self.T_LUFS - avgl["lufs"]
        vol_want = float(np.clip(10 ** (lufs_err / 20), 0.4, 1.7))
        t["volume"] = float(np.clip(S*t["volume"] + (1-S)*vol_want, 0.4, 1.7))

        # ── Multiband compression ─────────────────────────────────────────────
        # Crest factor: high = very dynamic → more compression needed
        crest_db = float(np.clip(20*np.log10(avgl["crest"] + 1e-9), 0, 30))
        comp_base = float(np.clip((crest_db - self.T_DYNAMIC) * 0.4, 0, 9))

        # Low band — extra compression if bass is boomy
        comp_low = comp_base + (2.0 if avgl["bass"] > 0.30 else 0.0)
        t["comp_low"] = float(np.clip(S*t["comp_low"] + (1-S)*comp_low, 0, 10))

        # Mid band — standard
        t["comp_mid"] = float(np.clip(S*t["comp_mid"] + (1-S)*comp_base, 0, 10))

        # High band — lighter, only compress if there's harsh transients
        comp_hi = comp_base * 0.6 + (1.5 if harsh_idx > 0.20 else 0)
        t["comp_high"] = float(np.clip(S*t["comp_high"] + (1-S)*comp_hi, 0, 8))

        # ── Stereo width ──────────────────────────────────────────────────────
        # Mono / near-mono → widen; already wide → leave alone
        if sw < 0.10:
            sw_want = 1.4   # very mono — widen
        elif sw > 0.55:
            sw_want = 1.0   # already very wide — don't touch
        else:
            sw_want = 1.0 + (0.12 - sw) * 2.0
        t["stereo_width"] = float(np.clip(
            S*t["stereo_width"] + (1-S)*sw_want, 0.6, 1.8))

        # ── Harmonic exciter ─────────────────────────────────────────────────
        # Add excitement to dull / dark tracks
        ex_want = 0.0
        if avgl["centroid"] < 0.30 and avgl["treble"] < 0.04:
            ex_want = 0.25
        elif avgl["centroid"] > 0.55:
            ex_want = 0.0   # already bright
        t["exciter"] = float(np.clip(S*t["exciter"] + (1-S)*ex_want, 0, 0.5))

        # ── Noise gate ────────────────────────────────────────────────────────
        # Estimate noise floor: if very quiet passages exist, gate them
        rms_db = float(20*np.log10(avgs["rms"] + 1e-9))
        gate_want = rms_db - 30 if rms_db < -35 else -70.0
        t["gate_db"] = float(np.clip(S*t["gate_db"] + (1-S)*gate_want, -70, -20))

        return dict(t)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AudioEngine:
    CHUNK = 2048

    def __init__(self):
        self.SR          = SR  # expose sample rate as instance attribute
        self.data        = None
        self.pos         = 0
        self.playing     = False
        self._stream     = None
        self._pos_lock   = threading.Lock()

        self.vol          = 1.0
        self.comp_low     = 2.0
        self.comp_mid     = 2.0
        self.comp_high    = 1.5
        self.stereo_width = 1.0
        self.exciter      = 0.0
        self.gate_db       = -70.0
        self.ai_enabled    = True
        # Shadow targets written by AI thread, read+smoothed in callback
        self._tgt_vol     = 1.0
        self._tgt_comp_lo = 2.0
        self._tgt_comp_mi = 2.0
        self._tgt_comp_hi = 1.5
        self._tgt_swid    = 1.0
        self._tgt_exc     = 0.0
        self._tgt_gate    = -70.0
        # Smoothing coef ~50ms
        self._sc_coef     = float(np.exp(-1.0 / (SR * 0.05)))
        self.filter_bank   = FilterBank(2)
        self.signal_ai     = SignalAI()
        self.noise_reducer    = NoiseReducer(sr=SR)
        self.dynamic_eq       = DynamicEQ(sr=SR, ch=2)
        self.dynamic_eq       = DynamicEQ(sr=SR, ch=2)
        self.de_esser         = DeEsser(sr=SR)
        self.transient_shaper = TransientShaper(sr=SR)
        self.dynamic_eq.reset()
        self.dynamic_eq = DynamicEQ(sr=SR, ch=2)
        self._mb_comp  = MultibandCompressor(ch=2)
        self._harmonic = HarmonicExciter(ch=2)
        self._limiter  = TruePeakLimiter()
        self._gate     = NoiseGate()
        self._dsp_zi          = {}  # persistent filter state for click-free DSP

        self.section_analyzer = SectionAnalyzer(sr=SR)
        self.auto_master      = AutoMaster(self)
        # C++ DSP chain — used when dsp_core is compiled
        self._cpp = CppDspChain(SR) if DSP_CPP else None

        self.on_position  = None
        self.on_spectrum  = None
        self.on_targets   = None
        self.on_section   = None   # callback(label: str)
        self._sc          = 0
        self._last_section = ""


    def load(self, path):
        if not MINIAUDIO_OK:
            raise RuntimeError("Run: pip install miniaudio")
        dec = miniaudio.decode_file(
            path, output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=2, sample_rate=SR)
        raw = np.frombuffer(dec.samples, dtype=np.float32).reshape(-1, 2).copy()
        self.data          = raw
        self.pos           = 0
        self.filter_bank   = FilterBank(2)
        self.signal_ai     = SignalAI()
        self.dynamic_eq.reset()
        self.dynamic_eq = DynamicEQ(sr=SR, ch=2)
        self._mb_comp  = MultibandCompressor(ch=2)
        self._harmonic = HarmonicExciter(ch=2)
        self._limiter  = TruePeakLimiter()
        self._gate     = NoiseGate()
        self._dsp_zi   = {}
        self.noise_reducer.reset()
        self.section_analyzer.sections = []
        self._last_section = ""
        return len(raw) / SR

    def play(self):
        if self.data is None: return
        self.playing = True

        def _cb(outdata, frames, time_info, status):
            with self._pos_lock:
                pos = self.pos
            end = min(pos + frames, len(self.data))
            n   = end - pos
            if n <= 0:
                outdata[:] = 0; self.playing = False; raise sd.CallbackStop()

            chunk = self.data[pos:end].copy()
            if n < frames:
                chunk = np.vstack([chunk, np.zeros((frames-n, 2), dtype=np.float32)])

            # ── DSP chain ─────────────────────────────────────────────────
            chunk = self.noise_reducer.process(chunk)

            if self._cpp is not None:
                # ── C++ path: entire chain in one native call ──────────────
                # Sync EQ gains to C++ filter bank
                EQ_ORDER = ("sub","bass","warmth","low_mid","mid",
                            "upper_mid","presence","treble","air")
                for bi, band in enumerate(EQ_ORDER):
                    g = self.filter_bank._gains.get(band, 0.0)
                    self._cpp.set_eq(bi, g)
                chunk = self._cpp.process(
                    chunk,
                    comp_lo      = self.comp_low,
                    comp_mid     = self.comp_mid,
                    comp_hi      = self.comp_high,
                    gate_db      = self.gate_db,
                    exciter      = self.exciter,
                    stereo_width = self.stereo_width,
                    volume       = self.vol,
                    dess_enabled = self.de_esser.enabled,
                    dess_thresh  = self.de_esser.threshold_db,
                    dess_ratio   = self.de_esser.ratio,
                    trans_enabled= self.transient_shaper.enabled,
                    trans_atk    = self.transient_shaper.attack_gain,
                    trans_sus    = self.transient_shaper.sustain_gain,
                    ceiling_db   = -0.3,
                )
            else:
                # ── Python/numpy fallback path ─────────────────────────────
                # Smooth live params toward AI targets (50ms ramp)
                sc = self._sc_coef
                self.vol          = sc*self.vol          + (1-sc)*self._tgt_vol
                self.comp_low     = sc*self.comp_low     + (1-sc)*self._tgt_comp_lo
                self.comp_mid     = sc*self.comp_mid     + (1-sc)*self._tgt_comp_mi
                self.comp_high    = sc*self.comp_high    + (1-sc)*self._tgt_comp_hi
                self.stereo_width = sc*self.stereo_width + (1-sc)*self._tgt_swid
                self.exciter      = sc*self.exciter      + (1-sc)*self._tgt_exc
                self.gate_db      = sc*self.gate_db      + (1-sc)*self._tgt_gate

                chunk = self._gate.process(chunk, self.gate_db)
                chunk = self.filter_bank.process(chunk)
                chunk = self.dynamic_eq.process(chunk)
                chunk = self.de_esser.process(chunk)
                chunk = self.transient_shaper.process(chunk)
                chunk = self._mb_comp.process(chunk, self.comp_low, self.comp_mid, self.comp_high)
                chunk = self._harmonic.process(chunk, self.exciter)
                chunk = stereo_widen(chunk, self.stereo_width)
                chunk = (chunk * self.vol).astype(np.float32)
                chunk = self._limiter.process(chunk)
            outdata[:] = chunk

            with self._pos_lock:
                self.pos = end

            self._sc += 1
            if self._sc % 5 == 0 and self.on_spectrum:
                mono = chunk[:, 0]
                fft  = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
                self.on_spectrum(fft)
            if self.on_position:
                self.on_position(end / SR)
            if n < frames:
                self.playing = False; raise sd.CallbackStop()

        def _ai():
            while self.playing:
                time.sleep(0.35)
                if not self.ai_enabled or self.data is None: continue
                with self._pos_lock:
                    pos = self.pos
                if pos < SignalAI.WIN_S: continue
                # long window for tonal balance, short for reactive control
                long_start  = max(0, pos - SignalAI.WIN_L)
                short_start = max(0, pos - SignalAI.WIN_S)
                chunk_long  = self.data[long_start:pos].copy()
                chunk_short = self.data[short_start:pos].copy()
                targets = self.signal_ai.analyze(chunk_long, chunk_short)

                EQ_BANDS = ("sub","bass","warmth","low_mid","mid",
                            "upper_mid","presence","treble","air")
                for band in EQ_BANDS:
                    self.filter_bank.update_gain(band, targets[band])
                # Write to shadow targets — callback interpolates smoothly
                self._tgt_vol     = targets["volume"]
                self._tgt_comp_lo = targets["comp_low"]
                self._tgt_comp_mi = targets["comp_mid"]
                self._tgt_comp_hi = targets["comp_high"]
                self._tgt_swid    = targets["stereo_width"]
                self._tgt_exc     = targets["exciter"]
                self._tgt_gate    = targets["gate_db"]
                # Apply per-section EQ offsets on top of base AI targets
                if self.section_analyzer.enabled:
                    with self._pos_lock:
                        cur_pos = self.pos
                    offsets = self.section_analyzer.get_eq_offsets(cur_pos / SR)
                    for band, offset in offsets.items():
                        if band in targets:
                            targets[band] = float(np.clip(
                                targets[band] + offset, -12, 12))
                    # fire section change callback
                    label = self.section_analyzer.get_current_section(cur_pos / SR)
                    if label != self._last_section:
                        self._last_section = label
                        if self.on_section:
                            self.on_section(label)

                if self.on_targets:
                    self.on_targets(targets)

        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass

        self._stream = sd.OutputStream(
            samplerate=SR, channels=2, dtype="float32",
            blocksize=self.CHUNK, callback=_cb, latency="low")
        self._stream.start()
        threading.Thread(target=_ai, daemon=True, name="ai").start()

    def pause(self):
        self.playing = False
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
            self._stream = None
        self.dynamic_eq.reset()
        self.dynamic_eq = DynamicEQ(sr=SR, ch=2)
        self._mb_comp  = MultibandCompressor(ch=2)
        self._harmonic = HarmonicExciter(ch=2)
        self._limiter  = TruePeakLimiter()
        self._gate     = NoiseGate()
        self._dsp_zi   = {}
        self.noise_reducer.reset()

    def seek(self, secs):
        new_pos = int(np.clip(secs * SR, 0,
                     len(self.data)-1 if self.data is not None else 0))
        with self._pos_lock:
            self.pos = new_pos
        self.filter_bank = FilterBank(2)
        self.dynamic_eq.reset()
        self.dynamic_eq = DynamicEQ(sr=SR, ch=2)
        self._mb_comp  = MultibandCompressor(ch=2)
        self._harmonic = HarmonicExciter(ch=2)
        self._limiter  = TruePeakLimiter()
        self._gate     = NoiseGate()
        self._dsp_zi   = {}
        self.noise_reducer.reset()

    @property
    def current_time(self): return self.pos / SR
    @property
    def duration(self): return len(self.data)/SR if self.data is not None else 0


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI
# ══════════════════════════════════════════════════════════════════════════════

def gemini_analyze(filename, description, api_key):
    if not GEMINI_OK: raise RuntimeError("pip install google-generativeai")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""You are a professional audio mastering engineer.
Calibrate a real-time 9-band psychoacoustic AI enhancer for this track.

Filename   : {filename}
Description: {description}

Return ONLY valid JSON (no markdown):
{{
  "genre": "...",
  "analysis": "one precise sentence about the sonic character",
  "target_lufs": <float -24 to -10, typical loudness target for this genre>,
  "target_bass_frac": <float 0.05 to 0.45, desired bass energy fraction>,
  "target_brightness": <float 0.20 to 0.65, spectral centroid target>,
  "compression": <float 0 to 8, how much dynamic compression for this genre>,
  "dynamic_range": <float 3 to 12, desired crest factor dB>,
  "enhancements": ["specific enhancement 1", "specific enhancement 2", "specific enhancement 3", "specific enhancement 4"]
}}"""
    resp = model.generate_content(prompt)
    text = resp.text.strip().replace("```json","").replace("```","").strip()
    return json.loads(text)



# ══════════════════════════════════════════════════════════════════════════════
#  DE-ESSER  — frequency-targeted dynamic compressor for sibilance
# ══════════════════════════════════════════════════════════════════════════════


class DynamicEQ:
    """
    Dynamic EQ — like a normal EQ band but the gain only activates when
    the signal in that frequency range exceeds a threshold.

    Each band has:
      freq       — centre frequency (Hz)
      band_type  — 'low_shelf' | 'high_shelf' | 'peak'
      threshold  — level (dBFS) above which gain kicks in
      ratio      — how aggressively to apply gain (1 = linear, 4 = hard)
      gain_db    — max gain/cut to apply when fully triggered (±dB)
      attack_ms  — how fast the band responds to level increase
      release_ms — how fast the band recovers

    The gain is computed from a sidechain envelope follower on the band signal.
    When the envelope exceeds threshold, gain is interpolated toward gain_db
    proportional to how far above threshold the signal is.
    All filters are stateful (zi carried across chunks) — zero clicks.
    """

    BAND_DEFS = [
        # name,        freq,  type,          thr_db, ratio, gain_db
        ("low",        100,   "low_shelf",   -20.0,  2.0,   -3.0),
        ("low_mid",    500,   "peak",        -18.0,  2.0,   -3.0),
        ("mid",        2000,  "peak",        -18.0,  2.0,   -3.0),
        ("high_mid",   5000,  "peak",        -20.0,  2.0,   -3.0),
        ("high",       10000, "high_shelf",  -20.0,  2.0,   -3.0),
    ]

    def __init__(self, sr=SR, ch=2):
        self.enabled   = False
        self.sr        = sr
        self.ch        = ch

        # Per-band params (user-editable)
        self.bands = {}
        for name, freq, btype, thr, ratio, gain in self.BAND_DEFS:
            self.bands[name] = {
                "freq":      freq,
                "type":      btype,
                "threshold": thr,
                "ratio":     ratio,
                "gain_db":   gain,
                "attack_ms": 10.0,
                "release_ms":150.0,
            }

        # Internal filter state
        self._zi_bp  = {}   # bandpass sidechain zi per band
        self._zi_eq  = {}   # EQ filter zi per band
        self._env    = {n: 0.0 for n in self.bands}
        self._gain   = {n: 0.0 for n in self.bands}  # current applied gain dB
        self._build_filters()

    def _build_filters(self):
        import scipy.signal as _sp
        for name, p in self.bands.items():
            freq  = p["freq"]
            btype = p["type"]
            sr    = self.sr
            # Sidechain: bandpass around the band centre for level detection
            lo = max(freq / 2.0,  20.0)  / (sr/2)
            hi = min(freq * 2.0, sr/2*0.99) / (sr/2)
            if lo >= hi: lo = hi * 0.5
            b_bp, a_bp = _sp.butter(2, [lo, hi], btype='band')
            self._zi_bp[name] = np.zeros(len(_sp.lfilter_zi(b_bp,a_bp)))  # mono sidechain

            # EQ filter at 0 dB (will be updated per-chunk as gain changes)
            b_eq, a_eq = self._make_eq_coef(freq, btype, 0.0)
            self._zi_eq[name] = np.zeros((len(_sp.lfilter_zi(b_eq,a_eq)), self.ch))
        self._coef_bp = {}
        self._coef_eq = {}
        import scipy.signal as _sp
        for name, p in self.bands.items():
            freq  = p["freq"]
            btype = p["type"]
            sr    = self.sr
            lo = max(freq/2.0,  20.0)  / (sr/2)
            hi = min(freq*2.0, sr/2*0.99) / (sr/2)
            if lo >= hi: lo = hi*0.5
            b_bp, a_bp = _sp.butter(2, [lo, hi], btype='band')
            self._coef_bp[name] = (b_bp, a_bp)
            b_eq, a_eq = self._make_eq_coef(freq, btype, 0.0)
            self._coef_eq[name] = (b_eq, a_eq)

    def _make_eq_coef(self, freq, btype, gain_db):
        """Build EQ filter coefficients for a given gain."""
        if btype == "low_shelf":
            return _low_shelf(freq, gain_db, 0.7)
        elif btype == "high_shelf":
            return _high_shelf(freq, gain_db, 0.7)
        else:
            return _peaking(freq, gain_db, 1.0)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return chunk
        import scipy.signal as _sp
        out = chunk.copy()
        n   = len(chunk)

        for name, p in self.bands.items():
            thr_lin   = 10 ** (p["threshold"] / 20)
            max_gain  = p["gain_db"]
            ratio     = p["ratio"]
            atk_coef  = np.exp(-1.0 / (self.sr * p["attack_ms"]  / 1000))
            rel_coef  = np.exp(-1.0 / (self.sr * p["release_ms"] / 1000))
            env       = self._env[name]
            cur_gain  = self._gain[name]

            # Sidechain: measure level in this band (mono, 1D zi)
            b_bp, a_bp = self._coef_bp[name]
            sc, self._zi_bp[name] = _sp.lfilter(
                b_bp, a_bp, out[:, 0], zi=self._zi_bp[name])

            # Envelope follower on sidechain (chunk-level RMS)
            rms  = float(np.sqrt(np.mean(sc**2)))
            coef = atk_coef if rms > env else rel_coef
            env_end = coef**n * env + (1 - coef**n) * rms
            env_avg = (env + env_end) / 2.0

            # How far above threshold? → target gain
            if env_avg > thr_lin:
                over_db  = 20 * np.log10(env_avg / (thr_lin + 1e-9))
                tgt_gain = max_gain * min(1.0, over_db / (6.0 / ratio))
            else:
                tgt_gain = 0.0

            # Smooth gain toward target
            g_coef  = atk_coef if abs(tgt_gain) > abs(cur_gain) else rel_coef
            gain_end = g_coef**n * cur_gain + (1 - g_coef**n) * tgt_gain

            # Apply EQ at interpolated gain across chunk
            gains_db = np.linspace(cur_gain, gain_end, n)
            # Rebuild filter only if gain changed meaningfully
            if abs(gain_end - cur_gain) > 0.05:
                b_eq, a_eq = self._make_eq_coef(
                    p["freq"], p["type"], float((cur_gain+gain_end)/2))
                self._coef_eq[name] = (b_eq, a_eq)

            b_eq, a_eq = self._coef_eq[name]
            eq_out = np.empty_like(out)
            for c in range(self.ch):
                eq_out[:, c], self._zi_eq[name][:, c] = _sp.lfilter(
                    b_eq, a_eq, out[:, c], zi=self._zi_eq[name][:, c])

            # Blend original vs EQ based on gain envelope
            # gain_end=0 → pass-through, gain_end=max → fully EQ'd
            alpha = min(1.0, abs(gain_end) / (abs(max_gain) + 1e-9))
            out   = (out * (1-alpha) + eq_out * alpha).astype(np.float32)

            self._env[name]  = float(env_end)
            self._gain[name] = float(gain_end)

        return out.astype(np.float32)

    def reset(self):
        self._env  = {n: 0.0 for n in self.bands}
        self._gain = {n: 0.0 for n in self.bands}
        self._build_filters()

class DeEsser:
    """
    Detects and reduces harsh sibilant energy (5–10 kHz) dynamically.

    Uses a frequency-split approach:
      1. Bandpass filter isolates the sibilance band
      2. Envelope follower measures energy in that band
      3. When energy exceeds threshold, gain is reduced only in that band
      4. Dry/wet mix blends result back — never changes non-sibilant content

    Parameters:
      threshold_db : level above which sibilance is attenuated (-30 to 0 dB)
      ratio        : how aggressively to compress sibilance (2:1 to 10:1)
      freq_lo      : low edge of sibilance band (default 5000 Hz)
      freq_hi      : high edge of sibilance band (default 10000 Hz)
    """

    def __init__(self, sr=SR):
        self.sr           = sr
        self.enabled      = False
        self.threshold_db = -20.0
        self.ratio        = 4.0
        self.freq_lo      = 5000
        self.freq_hi      = 10000
        self._env         = 0.0   # envelope follower state
        self._attack      = np.exp(-1 / (sr * 0.002))   # 2ms attack
        self._release     = np.exp(-1 / (sr * 0.080))   # 80ms release
        self._build_filters()

    def _build_filters(self):
        from scipy.signal import butter
        lo = self.freq_lo / (self.sr / 2)
        hi = min(self.freq_hi / (self.sr / 2), 0.99)
        self._b_bp, self._a_bp = butter(2, [lo, hi], btype='band')
        n_zi = len(sp.lfilter_zi(self._b_bp, self._a_bp))
        self._zi_mono = np.zeros(n_zi)               # zero-init sidechain
        self._zi_ch   = [np.zeros(n_zi), np.zeros(n_zi)]  # zero-init per channel

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return chunk
        out   = chunk.copy()
        thr   = 10 ** (self.threshold_db / 20)
        mono  = (chunk[:, 0] + chunk[:, 1]) * 0.5
        # isolate sibilance band — stateful zi carried across chunks
        band, self._zi_mono = sp.lfilter(
            self._b_bp, self._a_bp, mono, zi=self._zi_mono)
        env   = self._env
        gains = np.ones(len(chunk), dtype=np.float32)

        for i in range(len(band)):
            lvl = abs(band[i])
            coef = self._attack if lvl > env else self._release
            env  = coef * env + (1 - coef) * lvl
            if env > thr:
                gr = thr * (env / thr) ** (1.0 / self.ratio) / env
                gains[i] = float(gr)

        self._env = env
        # apply gain only to sibilance band — stateful per channel
        for c in range(chunk.shape[1]):
            band_c, self._zi_ch[c] = sp.lfilter(
                self._b_bp, self._a_bp, chunk[:, c], zi=self._zi_ch[c])
            other_c = chunk[:, c] - band_c
            out[:, c] = (band_c * gains + other_c).astype(np.float32)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSIENT SHAPER  — attack/sustain envelope control
# ══════════════════════════════════════════════════════════════════════════════

class TransientShaper:
    """
    Shapes the attack and sustain of transients independently.

    Separates the signal into:
      • Attack  — the fast-rising onset (kick, snare, pluck attack)
      • Sustain — the slow-decaying body (reverb tail, sustain, pad)

    attack_gain  > 0 → punchier transients (drums hit harder)
    attack_gain  < 0 → softer transients (less harsh)
    sustain_gain > 0 → more sustain/reverb (bigger sound)
    sustain_gain < 0 → tighter, drier sound

    Uses a fast/slow envelope pair — the difference is the transient signal.
    """

    def __init__(self, sr=SR):
        self.sr           = sr
        self.enabled      = False
        self.attack_gain  = 0.0   # dB, -12 to +12
        self.sustain_gain = 0.0   # dB, -12 to +12
        # fast envelope (tracks transients)
        self._fast  = np.exp(-1 / (sr * 0.001))   # 1ms
        # slow envelope (tracks body)
        self._slow  = np.exp(-1 / (sr * 0.050))   # 50ms
        self._ef    = 0.0
        self._es    = 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return chunk
        if abs(self.attack_gain) < 0.1 and abs(self.sustain_gain) < 0.1:
            return chunk

        att_lin = 10 ** (self.attack_gain  / 20)
        sus_lin = 10 ** (self.sustain_gain / 20)

        out  = chunk.copy()
        ef   = self._ef
        es   = self._es

        for i in range(len(chunk)):
            lvl = float(np.max(np.abs(chunk[i])))
            ef  = self._fast * ef + (1 - self._fast) * lvl
            es  = self._slow * es + (1 - self._slow) * lvl
            # transient = difference between fast and slow envelope
            transient = max(0.0, ef - es)
            sustain   = es
            # gain for this sample
            total = transient + sustain + 1e-9
            g = (transient * att_lin + sustain * sus_lin) / total
            out[i] = (chunk[i] * g).astype(np.float32)

        self._ef = ef
        self._es = es
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION ANALYZER  — verse/chorus/bridge detection + per-section EQ
# ══════════════════════════════════════════════════════════════════════════════

class SectionAnalyzer:
    """
    Detects musical sections (verse, chorus, bridge, intro, outro) and
    applies different AI EQ curves to each.

    Detection uses:
      • RMS energy contour — chorus is louder than verse
      • Spectral centroid — chorus is typically brighter
      • Spectral flux — high flux = section boundary
      • Repetition — finds repeated 4-bar blocks via self-similarity

    Per-section EQ adjustments (on top of base AI EQ):
      Intro/Outro : gentle — less bass, slightly warmer
      Verse       : clear   — slight mid boost for vocals
      Chorus      : full    — more bass + presence, slight air boost
      Bridge      : neutral — minimal change, preserve dynamics
    """

    SECTION_NAMES = ["intro", "verse", "chorus", "bridge", "outro"]

    # EQ offsets per section (added to SignalAI base targets)
    SECTION_EQ = {
        "intro":  {"bass": -1.0, "warmth":  0.5, "mid":  0.0, "presence": -0.5, "air": -0.5},
        "verse":  {"bass":  0.0, "warmth":  0.0, "mid":  1.5, "presence":  0.5, "air":  0.0},
        "chorus": {"bass":  2.0, "warmth":  0.5, "mid":  0.0, "presence":  1.0, "air":  1.0},
        "bridge": {"bass": -0.5, "warmth":  0.0, "mid":  0.5, "presence":  0.0, "air":  0.0},
        "outro":  {"bass": -1.5, "warmth":  0.5, "mid": -0.5, "presence": -0.5, "air": -1.0},
    }

    def __init__(self, sr=SR):
        self.sr       = sr
        self.enabled  = False
        self.sections = []   # list of (start_sec, end_sec, label)
        self._lock    = threading.Lock()

    def analyze_file(self, audio: np.ndarray, on_done=None):
        """Run section detection on full audio in background thread."""
        def _run():
            sections = self._detect_sections(audio)
            with self._lock:
                self.sections = sections
            if on_done:
                on_done(sections)
        threading.Thread(target=_run, daemon=True, name="section_analyzer").start()

    def _detect_sections(self, audio):
        """Segment audio into sections using energy + spectral features."""
        mono       = (audio[:, 0] + audio[:, 1]) * 0.5
        hop        = int(self.sr * 0.5)    # 500ms hop
        frame      = int(self.sr * 1.0)    # 1s frame
        n_frames   = (len(mono) - frame) // hop

        if n_frames < 4:
            dur = len(mono) / self.sr
            return [(0.0, dur, "verse")]

        rms_vals      = []
        centroid_vals = []
        freqs         = np.fft.rfftfreq(frame, 1/self.sr)

        for i in range(n_frames):
            seg  = mono[i*hop : i*hop + frame]
            rms  = float(np.sqrt(np.mean(seg**2) + 1e-9))
            X    = np.abs(np.fft.rfft(seg * np.hanning(frame))) + 1e-9
            X2   = X**2
            cent = float((freqs * X2).sum() / X2.sum()) / (self.sr / 2)
            rms_vals.append(rms)
            centroid_vals.append(cent)

        rms_arr  = np.array(rms_vals)
        cent_arr = np.array(centroid_vals)

        # Normalise
        rms_n  = (rms_arr  - rms_arr.min())  / (rms_arr.max()  - rms_arr.min()  + 1e-9)
        cent_n = (cent_arr - cent_arr.min()) / (cent_arr.max() - cent_arr.min() + 1e-9)

        # Energy score: chorus = high RMS + high centroid
        score = rms_n * 0.6 + cent_n * 0.4

        # Smoothed flux for boundary detection
        flux = np.abs(np.diff(score, prepend=score[0]))
        flux_smooth = np.convolve(flux, np.ones(4)/4, mode='same')

        # Find boundaries at local flux maxima above threshold
        threshold = float(np.mean(flux_smooth) + np.std(flux_smooth) * 0.8)
        boundaries = [0]
        for i in range(2, n_frames - 2):
            if (flux_smooth[i] > threshold and
                flux_smooth[i] >= flux_smooth[i-1] and
                flux_smooth[i] >= flux_smooth[i+1]):
                # minimum 4 frames (2 seconds) between boundaries
                if i - boundaries[-1] >= 4:
                    boundaries.append(i)
        boundaries.append(n_frames)

        # Label each section based on its average score
        total_dur  = len(mono) / self.sr
        n_sections = len(boundaries) - 1
        sections   = []

        for idx in range(n_sections):
            start_f = boundaries[idx]
            end_f   = boundaries[idx + 1]
            avg_score = float(np.mean(score[start_f:end_f]))

            start_s = start_f * hop / self.sr
            end_s   = min(end_f * hop / self.sr, total_dur)

            # Label heuristics
            if idx == 0 and n_sections >= 4:
                label = "intro"
            elif idx == n_sections - 1 and n_sections >= 4:
                label = "outro"
            elif avg_score > 0.65:
                label = "chorus"
            elif avg_score < 0.35:
                label = "bridge"
            else:
                label = "verse"

            sections.append((start_s, end_s, label))

        return sections

    def get_current_section(self, pos_sec: float) -> str:
        """Return the section label for the current playback position."""
        with self._lock:
            for start, end, label in self.sections:
                if start <= pos_sec < end:
                    return label
        return "verse"

    def get_eq_offsets(self, pos_sec: float) -> dict:
        """Return EQ dB offsets for current section."""
        label = self.get_current_section(pos_sec)
        return self.SECTION_EQ.get(label, {})


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO MASTER  — one-click full mastering chain
# ══════════════════════════════════════════════════════════════════════════════

class AutoMaster:
    """
    One-click mastering that analyses the track and applies a full chain:

      1. Analyse loudness, dynamics, spectral balance, peak levels
      2. Set SignalAI targets to genre-optimal values
      3. Apply 9-band EQ to correct tonal imbalances
      4. Set 3-band compression ratios based on crest factor
      5. Set stereo width based on M/S analysis
      6. Apply true peak limiter at -0.3 dBFS
      7. Normalise output to target LUFS (-14 for streaming, -9 for club)

    All processing is non-destructive — applied as engine parameter targets,
    not baked into the audio buffer.
    """

    STREAM_LUFS = -14.0   # Spotify / Apple Music / YouTube target
    CLUB_LUFS   = -9.0    # DJ / club playback target
    LOUD_LUFS   = -8.0    # Maximum loudness (aggressive)

    def __init__(self, engine):
        self.engine   = engine
        self.enabled  = False
        self.target   = "stream"   # "stream", "club", "loud"
        self.on_done  = None
        self.on_log   = None

    def run(self):
        """Analyse and apply mastering settings in background thread."""
        threading.Thread(target=self._run, daemon=True, name="automaster").start()

    def _run(self):
        try:
            engine = self.engine
            if engine.data is None:
                return

            data = engine.data.copy()
            mono = (data[:, 0] + data[:, 1]) * 0.5

            self._log("Analysing track...")

            # ── Loudness ─────────────────────────────────────────────────
            rms       = float(np.sqrt(np.mean(mono**2)) + 1e-9)
            lufs_est  = float(20 * np.log10(rms))
            peak      = float(np.max(np.abs(data)))
            crest_db  = float(20 * np.log10(peak / rms + 1e-9))

            target_lufs = {
                "stream": self.STREAM_LUFS,
                "club":   self.CLUB_LUFS,
                "loud":   self.LOUD_LUFS,
            }.get(self.target, self.STREAM_LUFS)

            self._log(f"  Measured: {lufs_est:.1f} LUFS, peak {20*np.log10(peak+1e-9):.1f} dBFS, "
                      f"crest {crest_db:.1f} dB")
            self._log(f"  Target: {target_lufs:.1f} LUFS")

            # ── Spectral balance ─────────────────────────────────────────
            n     = min(len(mono), 65536)
            freqs = np.fft.rfftfreq(n, 1/SR)
            X     = np.abs(np.fft.rfft(mono[:n] * np.hanning(n))) + 1e-9
            X2    = X**2
            tot   = X2.sum()
            def frac(lo, hi):
                m = (freqs >= lo) & (freqs < hi)
                return float(X2[m].sum() / tot) if m.any() else 0.0

            bass_f    = frac(50, 200)
            warmth_f  = frac(200, 500)
            mid_f     = frac(900, 2500)
            bright_f  = frac(2500, 10000)
            centroid  = float((freqs * X2).sum() / tot) / (SR / 2)

            self._log(f"  Spectral: bass={bass_f:.2f} warmth={warmth_f:.2f} "
                      f"mid={mid_f:.2f} bright={bright_f:.2f}")

            # ── Set SignalAI targets ──────────────────────────────────────
            self._log("Setting AI targets...")
            engine.signal_ai.T_LUFS      = target_lufs
            engine.signal_ai.T_BASS_FRAC = 0.18
            engine.signal_ai.T_BRIGHT    = 0.40
            engine.signal_ai.T_DYNAMIC   = min(crest_db * 0.6, 8.0)

            # ── EQ corrections ───────────────────────────────────────────
            self._log("Applying EQ corrections...")

            # Sub rumble
            if frac(15, 50) > 0.05:
                engine.filter_bank.update_gain("sub", -3.0)
                self._log("  Sub: -3 dB (rumble detected)")

            # Bass balance
            bass_want = np.clip((0.18 - bass_f) / 0.18 * 9, -6, 6)
            engine.filter_bank.update_gain("bass", float(bass_want))
            self._log(f"  Bass: {bass_want:+.1f} dB")

            # Mud cut
            if warmth_f > 0.22:
                engine.filter_bank.update_gain("warmth", -2.5)
                self._log("  Warmth: -2.5 dB (mud cut)")

            # Presence boost if dull
            if centroid < 0.30:
                engine.filter_bank.update_gain("presence", 2.0)
                engine.filter_bank.update_gain("air", 1.5)
                self._log("  Presence +2 dB, Air +1.5 dB (dull track)")
            elif centroid > 0.55:
                engine.filter_bank.update_gain("upper_mid", -2.0)
                engine.filter_bank.update_gain("treble", -1.5)
                self._log("  HiMid -2, Treble -1.5 dB (harsh track)")

            # ── Compression ──────────────────────────────────────────────
            self._log("Setting compression...")
            comp_base = float(np.clip((crest_db - 6) * 0.5, 0, 8))
            engine.comp_low  = min(comp_base + 1.5, 9.0)
            engine.comp_mid  = comp_base
            engine.comp_high = max(comp_base - 1.0, 0.5)
            self._log(f"  Comp: lo={engine.comp_low:.1f} mid={engine.comp_mid:.1f} "
                      f"hi={engine.comp_high:.1f}")

            # ── Stereo width ─────────────────────────────────────────────
            M    = data[:, 0] + data[:, 1]
            S    = data[:, 0] - data[:, 1]
            sw   = float(np.mean(S**2) / (np.mean(M**2) + 1e-9))
            if sw < 0.08:
                engine.stereo_width = 1.35
                self._log("  Stereo: widened to 1.35 (very mono)")
            elif sw > 0.5:
                engine.stereo_width = 1.0
                self._log("  Stereo: left at 1.0 (already wide)")
            else:
                engine.stereo_width = 1.15
                self._log("  Stereo: slight widening to 1.15")

            # ── Volume / gain staging ─────────────────────────────────────
            gain_want = float(np.clip(10 ** ((target_lufs - lufs_est) / 20), 0.3, 2.5))
            engine.vol = gain_want
            self._log(f"  Volume: {gain_want:.3f} ({(target_lufs - lufs_est):+.1f} dB)")

            self._log("✓ Auto-master complete!")
            if self.on_done:
                self.on_done()

        except Exception as e:
            self._log(f"Auto-master error: {e}")

    def _log(self, msg):
        print(f"[AutoMaster] {msg}")
        if self.on_log:
            self.on_log(msg)