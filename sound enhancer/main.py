"""
╔══════════════════════════════════════════════════════╗
║  SONIC AI — main.py                                 ║
║  GUI — run this file to start the app               ║
╚══════════════════════════════════════════════════════╝

Run:
    python main.py

Install:
    pip install numpy scipy sounddevice miniaudio deepfilternet google-generativeai
"""

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading, os, time
import numpy as np

from engine import (
    SR, DEEPFILTER_OK, NOISEREDUCE_OK, SPEECHBRAIN_OK,
    FilterBank, SignalAI, AudioEngine, NoiseReducer,
    DeEsser, TransientShaper, SectionAnalyzer, AutoMaster,
    gemini_analyze,
)

def _check_deepfilternet():
    import importlib.util
    return importlib.util.find_spec("df") is not None
DEEPFILTER_NET_OK = _check_deepfilternet()

# ══════════════════════════════════════════════════════════════════════════════
#  UI CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BG     = "#0a0a14"
PANEL  = "#1a1a2e"
BORDER = "#333333"
ACCENT = "#00ff99"
DIM    = "#666666"
MONO   = ("Courier New", 9)

class App(tk.Tk):

    SLIDERS = [
        # 9-band EQ
        ("sub",        "SUB",        -8,   8,   0.0, "#ff2200"),
        ("bass",       "BASS",      -12,  12,   0.0, "#ff5533"),
        ("warmth",     "WARMTH",     -8,   8,   0.0, "#ff8800"),
        ("low_mid",    "LO-MID",     -8,   8,   0.0, "#ffaa00"),
        ("mid",        "MID",        -8,   8,   0.0, "#ffdd00"),
        ("upper_mid",  "HI-MID",     -8,   8,   0.0, "#aaff00"),
        ("presence",   "PRESENCE",   -8,   8,   0.0, "#00ff88"),
        ("treble",     "TREBLE",    -10,  10,   0.0, "#00ccff"),
        ("air",        "AIR",        -8,   8,   0.0, "#0088ff"),
    ]
    PARAM_SLIDERS = [
        # dynamics / colour
        ("volume",        "VOLUME",    0.3, 1.7,  1.0, "#cc88ff"),
        ("comp_low",      "CMP-LO",    0.0,10.0,  2.0, "#ff88aa"),
        ("comp_mid",      "CMP-MD",    0.0,10.0,  2.0, "#ff88cc"),
        ("comp_high",     "CMP-HI",    0.0, 8.0,  1.5, "#ff88ee"),
        ("stereo_width",  "WIDTH",     0.5, 2.0,  1.0, "#88ffcc"),
        ("exciter",       "EXCITE",    0.0, 0.5,  0.0, "#ffcc00"),
    ]

    def __init__(self):
        super().__init__()
        self.title("CLARIFE")
        self.geometry("880x700")
        self.minsize(700, 500)
        self.resizable(True, True)
        self.configure(bg=BG)
        self.engine         = AudioEngine()
        self.is_playing     = False
        self.repeat_enabled = False
        self.after_id       = None
        self.s_var          = {}
        self.ai_on          = tk.BooleanVar(value=True)
        self._spec_data     = np.zeros(64)
        self._log_visible   = True    # log visible by default
        self._spectrogram   = []      # rolling rows for waterfall
        self._SGRAM_ROWS    = 60      # number of rows in waterfall
        self._seeking       = False    # guard: suppress seek during prog.set()
        self._build()
        self.engine.on_position = lambda t: None
        self.engine.on_spectrum = self._cb_spec
        self.engine.on_targets  = self._cb_targets
        self.engine.on_section  = self._cb_section

    def _build(self):
        # ── Scrollable container ──────────────────────────────────────────
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self._vbar   = tk.Scrollbar(self, orient="vertical",
                                     command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vbar.set)
        self._vbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Inner frame — all widgets go here, not on self
        self._inner = tk.Frame(self._canvas, bg=BG)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw")

        # Resize inner frame width when window resizes
        def _on_canvas_resize(e):
            self._canvas.itemconfig(self._win_id, width=e.width)
        self._canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region when inner frame changes size
        def _on_frame_resize(e):
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._inner.bind("<Configure>", _on_frame_resize)

        # Mouse wheel scrolling — Windows, Mac, Linux
        def _on_mousewheel(e):
            self._canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        def _on_mousewheel_linux(e):
            self._canvas.yview_scroll(-1 if e.num==4 else 1, "units")
        self._canvas.bind_all("<MouseWheel>",   _on_mousewheel)
        self._canvas.bind_all("<Button-4>",     _on_mousewheel_linux)
        self._canvas.bind_all("<Button-5>",     _on_mousewheel_linux)

        # Shortcut: use _inner as parent for all widgets
        R = self._inner   # alias for inner scrollable frame

        # ── top bar with title + hamburger ────────────────────────────────
        topbar = tk.Frame(R, bg=BG)
        topbar.pack(fill="x", padx=14, pady=(10,0))
        tk.Label(topbar, text="CLARIFE", font=("Courier New",22,"bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        # Hamburger menu button (3 lines = log panel toggle)
        self.ham_btn = tk.Button(
            topbar, text="☰  LOG", command=self._toggle_log,
            font=("Courier New", 9, "bold"),
            bg="#111", fg="#555", activebackground="#1a2a1a",
            activeforeground=ACCENT, relief="flat", bd=0,
            padx=10, pady=6, cursor="hand2")
        self.ham_btn.pack(side="right", padx=(0,4))
        tk.Label(R, text="REAL-TIME SIGNAL INTELLIGENCE",
                 font=("Courier New",8), fg=DIM, bg=BG).pack()

        # row1: file + gemini
        r1 = tk.Frame(R, bg=BG)
        r1.pack(fill="x", padx=14, pady=(10,0))

        fp = self._panel(r1, "▸ TRACK")
        fp.pack(side="left", fill="both", expand=True, padx=(0,7))
        self.file_lbl = tk.Label(fp, text="No track loaded", font=MONO,
                                 fg="#444", bg=PANEL, wraplength=220)
        self.file_lbl.pack(pady=(0,6))
        self._btn(fp, "📂  OPEN FILE", self._open).pack(fill="x")
        self.save_btn = self._btn(fp, "💾  SAVE ENHANCED", self._save, w=20)
        self.save_btn.pack(fill="x", pady=(4,0))
        self.save_status = tk.Label(fp, text="", font=("Courier New",7),
                                    fg="#888", bg=PANEL, wraplength=220)
        self.save_status.pack(anchor="w")
        tk.Label(fp, text="Description (optional):",
                 font=("Courier New",8), fg="#444", bg=PANEL).pack(anchor="w", pady=(8,2))
        self.desc_var = tk.StringVar()
        tk.Entry(fp, textvariable=self.desc_var, font=MONO,
                 bg="#111", fg="#ccc", insertbackground=ACCENT,
                 relief="flat", bd=4).pack(fill="x")

        gp = self._panel(r1, "▸ GEMINI (optional)")
        gp.pack(side="left", fill="both", expand=True)
        tk.Label(gp, text="API Key:", font=("Courier New",8), fg="#444", bg=PANEL).pack(anchor="w")
        self.api_var = tk.StringVar(value=os.environ.get("GEMINI_API_KEY",""))
        tk.Entry(gp, textvariable=self.api_var, font=MONO, show="•",
                 bg="#111", fg="#ccc", insertbackground=ACCENT,
                 relief="flat", bd=4).pack(fill="x")
        tk.Label(gp, text="aistudio.google.com — free key",
                 font=("Courier New",7), fg="#333", bg=PANEL).pack(pady=(2,6))
        self.gem_btn = self._btn(gp, "✦  GEMINI GENRE SCAN", self._gemini)
        self.gem_btn.pack(fill="x")
        self.gem_lbl = tk.Label(gp, text="", font=("Courier New",8),
                                fg=ACCENT, bg=PANEL, wraplength=200, justify="left")
        self.gem_lbl.pack(pady=(6,0))

        # AI panel
        ap = self._panel(R, "▸ SIGNAL AI")
        ap.pack(fill="x", padx=14, pady=(8,0))
        arow = tk.Frame(ap, bg=PANEL)
        arow.pack(fill="x")
        tk.Checkbutton(arow, text="  AUTO-ENHANCE ENABLED",
                       variable=self.ai_on, command=self._toggle_ai,
                       font=("Courier New",9,"bold"), fg=ACCENT, bg=PANEL,
                       selectcolor="#111", activebackground=PANEL,
                       activeforeground=ACCENT).pack(side="left")
        self.ai_badge = tk.Label(arow, text="● IDLE",
                                  font=("Courier New",9,"bold"), fg="#333", bg=PANEL)
        self.ai_badge.pack(side="right", padx=8)
        self.meters = {}
        mrow = tk.Frame(ap, bg=PANEL)
        mrow.pack(fill="x", pady=(4,2))
        all_meter_bands = self.SLIDERS + self.PARAM_SLIDERS
        for key, label, lo, hi, _, color in all_meter_bands:
            col = tk.Frame(mrow, bg=PANEL)
            col.pack(side="left", expand=True, fill="x")
            tk.Label(col, text=label[:4], font=("Courier New",6),
                     fg=color, bg=PANEL).pack()
            bar = tk.Canvas(col, height=4, bg="#111", highlightthickness=0)
            bar.pack(fill="x", padx=1)
            self.meters[key] = (bar, lo, hi, color)

        # spectrum
        spp = self._panel(R, "▸ SPECTROGRAM  —  frequency over time")
        spp.pack(fill="x", padx=14, pady=(8,0))
        # Top row: live spectrum bar
        self.spec_cv = tk.Canvas(spp, height=55, bg="#060610", highlightthickness=0)
        self.spec_cv.pack(fill="x")
        # Bottom: rolling waterfall spectrogram
        self.sgram_cv = tk.Canvas(spp, height=80, bg="#000008", highlightthickness=0)
        self.sgram_cv.pack(fill="x")

        # playback
        pb = self._panel(R, "▸ PLAYBACK")
        pb.pack(fill="x", padx=14, pady=(8,0))
        pbr = tk.Frame(pb, bg=PANEL)
        pbr.pack(fill="x")
        self.play_btn = self._btn(pbr, "▶  PLAY", self._play_pause, w=12)
        self.play_btn.pack(side="left", padx=(0,8))
        self._btn(pbr, "↺  RESET AI", self._reset_ai, w=10).pack(side="left", padx=(0,8))
        self.am_btn = self._btn(pbr, "⚡  AUTO-MASTER", self._run_automaster, w=15)
        self.am_btn.pack(side="left", padx=(0,8))
        self.section_lbl = tk.Label(pbr, text="─", font=("Courier New",9,"bold"),
                                     fg="#555", bg=PANEL)
        self.section_lbl.pack(side="left", padx=(4,0))
        self.repeat_btn = tk.Button(
            pbr, text="🔁  REPEAT OFF", command=self._toggle_repeat,
            font=("Courier New", 9, "bold"),
            bg="#111", fg="#555", activebackground="#1a1a1a",
            activeforeground=ACCENT, relief="flat", bd=0,
            padx=8, pady=6, cursor="hand2", width=14)
        self.repeat_btn.pack(side="left")
        self.time_lbl = tk.Label(pbr, text="0:00 / 0:00", font=MONO,
                                  fg="#555", bg=PANEL)
        self.time_lbl.pack(side="right")
        self.prog = ttk.Scale(pb, from_=0, to=100, orient="horizontal",
                               command=self._seek)
        ttk.Style().configure("TScale", troughcolor=BORDER, background=ACCENT)
        self.prog.pack(fill="x", pady=(5,0))

        # ── Noise Reduction panel ─────────────────────────────────────────
        np_panel = self._panel(R, "▸ VOICE ISOLATION  +  NOISE REMOVAL")
        np_panel.pack(fill="x", padx=14, pady=(8,0))

        # ── Mode selector row ─────────────────────────────────────────────
        mode_row = tk.Frame(np_panel, bg=PANEL)
        mode_row.pack(fill="x", padx=6, pady=(6,2))

        tk.Label(mode_row, text="MODE:", font=("Courier New",8,"bold"),
                 fg="#aaaaaa", bg=PANEL).pack(side="left", padx=(0,8))

        self.nr_mode = tk.StringVar(value="wiener")

        # Wiener button
        self.btn_wiener = tk.Button(
            mode_row, text="🔊  SPECTRAL",
            command=lambda: self._set_nr_mode("wiener"),
            font=("Courier New",8,"bold"),
            bg="#001a2a", fg="#44aaff",
            activebackground="#002a44", activeforeground="#66ccff",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2")
        self.btn_wiener.pack(side="left", padx=(0,4))

        # DeepFilterNet button
        self.btn_deepfilter = tk.Button(
            mode_row, text="🧠  DEEP LEARNING",
            command=lambda: self._set_nr_mode("deepfilter"),
            font=("Courier New",8,"bold"),
            bg="#111", fg="#555",
            activebackground="#0a1a0a", activeforeground="#44ff88",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2")
        self.btn_deepfilter.pack(side="left", padx=(0,4))

        # Demucs button
        self.btn_demucs = tk.Button(
            mode_row, text="🎙  VOICE ISOLATION",
            command=lambda: self._set_nr_mode("demucs"),
            font=("Courier New",8,"bold"),
            bg="#111", fg="#555",
            activebackground="#1a0a2a", activeforeground="#cc88ff",
            relief="flat", bd=0, padx=8, pady=4, cursor="hand2")
        self.btn_demucs.pack(side="left", padx=(0,4))

        # Status badge
        self.nr_mode_badge = tk.Label(
            mode_row,
            text="● Wiener spectral filter active",
            font=("Courier New",8), fg="#44aaff", bg=PANEL)
        self.nr_mode_badge.pack(side="left", padx=(10,0))

        # ── Controls row ──────────────────────────────────────────────────
        nr_row = tk.Frame(np_panel, bg=PANEL)
        nr_row.pack(fill="x", padx=6, pady=(4,4))

        tk.Label(nr_row, text="STRENGTH:", font=("Courier New",8),
                 fg="#ff6666", bg=PANEL).pack(side="left", padx=(0,6))
        self.nr_strength = tk.DoubleVar(value=0.7)
        tk.Scale(nr_row, from_=0.0, to=1.0, orient="horizontal",
                 variable=self.nr_strength, resolution=0.05,
                 length=160, showvalue=False,
                 bg=PANEL, fg="#ff4444", troughcolor="#2a0000",
                 highlightthickness=0, bd=0, activebackground="#ff4444",
                 state="normal").pack(side="left")
        self.nr_val_lbl = tk.Label(nr_row, text="0.70", font=("Courier New",8),
                                   fg="#ff4444", bg=PANEL)
        self.nr_val_lbl.pack(side="left", padx=(4,0))
        self.nr_strength.trace_add("write",
            lambda *_: self.nr_val_lbl.config(
                text=f"{self.nr_strength.get():.2f}"))

        # DNS passes selector (only shown in DL mode)
        self.nr_passes_frame = tk.Frame(nr_row, bg=PANEL)
        self.nr_passes_frame.pack(side="left", padx=(14,0))
        tk.Label(self.nr_passes_frame, text="DNS PASSES:",
                 font=("Courier New",8), fg="#44ff88", bg=PANEL
                 ).pack(side="left", padx=(0,4))
        self.nr_passes = tk.IntVar(value=2)
        for n in (1, 2, 3, 4):
            tk.Radiobutton(
                self.nr_passes_frame, text=str(n),
                variable=self.nr_passes, value=n,
                font=("Courier New",8,"bold"),
                fg="#44ff88", bg=PANEL,
                selectcolor="#0a1a0a",
                activebackground=PANEL, activeforeground="#88ffaa",
                indicatoron=0,
                width=2, relief="flat", bd=1,
                cursor="hand2",
            ).pack(side="left", padx=1)
        self.nr_passes_frame.pack_forget()   # hidden until DL mode selected

        # Auto toggle
        self.nr_auto = tk.BooleanVar(value=True)
        tk.Checkbutton(nr_row, text="  AUTO on load",
                       variable=self.nr_auto,
                       font=("Courier New",8), fg="#ff8888", bg=PANEL,
                       selectcolor="#111", activebackground=PANEL,
                       activeforeground="#ffaaaa").pack(side="left", padx=(12,0))

        # Run button
        self.nr_btn = tk.Button(
            nr_row, text="🧹  PROCESS",
            command=self._run_nr,
            font=("Courier New",9,"bold"),
            bg="#1a0000", fg="#ff4444",
            activebackground="#2a0000", activeforeground="#ff6666",
            relief="flat", bd=0, padx=10, pady=5, cursor="hand2")
        self.nr_btn.pack(side="left", padx=(8,0))

        # Restore button
        self.nr_restore_btn = tk.Button(
            nr_row, text="↺  RESTORE",
            command=self._restore_original,
            font=("Courier New",9,"bold"),
            bg="#111", fg="#555",
            activebackground="#1a1a1a", activeforeground="#888",
            relief="flat", bd=0, padx=10, pady=5, cursor="hand2",
            state="disabled")
        self.nr_restore_btn.pack(side="left", padx=(4,0))

        # Progress bar + status
        self.nr_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            np_panel, variable=self.nr_progress_var,
            maximum=100, mode="determinate", length=400
        ).pack(fill="x", padx=8, pady=(2,2))
        self.nr_status = tk.Label(np_panel,
            text="Noise Filter mode — auto-processes on load",
            font=("Courier New",8), fg="#555", bg=PANEL)
        self.nr_status.pack(anchor="w", padx=8, pady=(0,5))


        # Store original audio for restore
        self._original_audio = None

        # ── De-esser panel ───────────────────────────────────────────────

        # ── Dynamic EQ panel ─────────────────────────────────────────────
        deqp = self._panel(R, "▸ DYNAMIC EQ  —  frequency-selective compression")
        deqp.pack(fill="x", padx=14, pady=(6,0))
        deq_top = tk.Frame(deqp, bg=PANEL)
        deq_top.pack(fill="x", padx=6, pady=(6,2))
        self.deq_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(deq_top, text="  ENABLE",
                       variable=self.deq_enabled, command=self._toggle_deq,
                       font=("Courier New",9,"bold"), fg="#ff8844", bg=PANEL,
                       selectcolor="#111", activebackground=PANEL).pack(side="left")
        tk.Label(deq_top,
                 text="  Each band only activates when that frequency range gets too loud",
                 font=("Courier New",7), fg="#555", bg=PANEL).pack(side="left", padx=(8,0))

        # One row per band
        BAND_COLORS = {
            "low":      "#4488ff",
            "low_mid":  "#44ffaa",
            "mid":      "#ffff44",
            "high_mid": "#ff8844",
            "high":     "#ff44aa",
        }
        BAND_LABELS = {
            "low":      "LOW  100Hz",
            "low_mid":  "LO-MID 500Hz",
            "mid":      "MID  2kHz",
            "high_mid": "HI-MID 5kHz",
            "high":     "HIGH 10kHz",
        }
        self._deq_vars = {}
        for band_name, color in BAND_COLORS.items():
            row = tk.Frame(deqp, bg=PANEL)
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=BAND_LABELS[band_name],
                     font=("Courier New",8,"bold"), fg=color, bg=PANEL,
                     width=14, anchor="w").pack(side="left")

            vars_ = {}
            for label, key, lo, hi, default, res in [
                ("THR dB",  "threshold", -40, 0,   -20.0, 1.0),
                ("RATIO",   "ratio",      1,  8,     2.0, 0.5),
                ("GAIN dB", "gain_db",  -12, 12,    -3.0, 0.5),
                ("ATK ms",  "attack_ms", 1,  100,   10.0, 1.0),
                ("REL ms",  "release_ms",10, 500,  150.0, 10.0),
            ]:
                tk.Label(row, text=f" {label}", font=("Courier New",7),
                         fg=color, bg=PANEL).pack(side="left")
                v = tk.DoubleVar(value=default)
                vars_[key] = v
                tk.Scale(row, from_=lo, to=hi, orient="horizontal",
                         variable=v, resolution=res, length=80, showvalue=True,
                         bg=PANEL, fg=color, troughcolor="#1a0a00",
                         highlightthickness=0, bd=0, activebackground=color,
                         font=("Courier New",6),
                         command=lambda val, bn=band_name, k=key: self._deq_set(bn, k, float(val))
                         ).pack(side="left")
            self._deq_vars[band_name] = vars_

        # ── Dynamic EQ ────────────────────────────────────────────────────
        dep = self._panel(R, "▸ DE-ESSER  —  sibilance control")
        dep.pack(fill="x", padx=14, pady=(8,0))
        de_row = tk.Frame(dep, bg=PANEL)
        de_row.pack(fill="x", padx=6, pady=6)
        self.de_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(de_row, text="  ENABLE",
                       variable=self.de_enabled, command=self._toggle_deesser,
                       font=("Courier New",9,"bold"), fg="#ffaa00", bg=PANEL,
                       selectcolor="#111", activebackground=PANEL).pack(side="left")
        for label, attr, lo, hi, default, color in [
            ("THRESHOLD dB", "threshold_db", -40, 0,  -20.0, "#ffaa00"),
            ("RATIO",        "ratio",         1,  10,   4.0, "#ffcc44"),
        ]:
            tk.Label(de_row, text=f"  {label}:", font=("Courier New",7),
                     fg=color, bg=PANEL).pack(side="left")
            v = tk.DoubleVar(value=default)
            tk.Scale(de_row, from_=lo, to=hi, orient="horizontal",
                     variable=v, resolution=0.5, length=100, showvalue=True,
                     bg=PANEL, fg=color, troughcolor="#1a1000",
                     highlightthickness=0, bd=0, activebackground=color,
                     font=("Courier New",7),
                     command=lambda val, a=attr: setattr(
                         self.engine.de_esser, a, float(val))
                     ).pack(side="left")

        # ── Transient shaper panel ────────────────────────────────────────
        trp = self._panel(R, "▸ TRANSIENT SHAPER  —  attack / sustain")
        trp.pack(fill="x", padx=14, pady=(6,0))
        tr_row = tk.Frame(trp, bg=PANEL)
        tr_row.pack(fill="x", padx=6, pady=6)
        self.tr_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(tr_row, text="  ENABLE",
                       variable=self.tr_enabled, command=self._toggle_transient,
                       font=("Courier New",9,"bold"), fg="#44ffcc", bg=PANEL,
                       selectcolor="#111", activebackground=PANEL).pack(side="left")
        for label, attr, lo, hi, default, color in [
            ("ATTACK dB",  "attack_gain",  -12, 12, 0.0, "#44ffcc"),
            ("SUSTAIN dB", "sustain_gain", -12, 12, 0.0, "#22ddaa"),
        ]:
            tk.Label(tr_row, text=f"  {label}:", font=("Courier New",7),
                     fg=color, bg=PANEL).pack(side="left")
            v = tk.DoubleVar(value=default)
            tk.Scale(tr_row, from_=lo, to=hi, orient="horizontal",
                     variable=v, resolution=0.5, length=110, showvalue=True,
                     bg=PANEL, fg=color, troughcolor="#001a14",
                     highlightthickness=0, bd=0, activebackground=color,
                     font=("Courier New",7),
                     command=lambda val, a=attr: setattr(
                         self.engine.transient_shaper, a, float(val))
                     ).pack(side="left")

        # ── Per-section AI panel ──────────────────────────────────────────
        sap = self._panel(R, "▸ PER-SECTION AI  —  verse / chorus / bridge EQ")
        sap.pack(fill="x", padx=14, pady=(6,0))
        sa_row = tk.Frame(sap, bg=PANEL)
        sa_row.pack(fill="x", padx=6, pady=4)
        self.sa_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(sa_row, text="  ENABLE SECTION DETECTION",
                       variable=self.sa_enabled, command=self._toggle_sections,
                       font=("Courier New",9,"bold"), fg="#aa88ff", bg=PANEL,
                       selectcolor="#111", activebackground=PANEL).pack(side="left")
        self.sa_status = tk.Label(sa_row, text="Load a track and enable to analyse sections",
                                   font=("Courier New",8), fg="#555", bg=PANEL)
        self.sa_status.pack(side="left", padx=(14,0))
        # section timeline bar (drawn after analysis)
        self.sa_canvas = tk.Canvas(sap, height=18, bg="#0a0a14", highlightthickness=0)
        self.sa_canvas.pack(fill="x", padx=6, pady=(0,4))

        # EQ sliders — 9 bands
        eq = self._panel(R, "▸ 9-BAND EQ  —  AI adjusts live")
        eq.pack(fill="x", padx=14, pady=(8,0))
        srow = tk.Frame(eq, bg=PANEL)
        srow.pack(fill="x")
        for key, label, lo, hi, default, color in self.SLIDERS:
            col = tk.Frame(srow, bg=PANEL)
            col.pack(side="left", expand=True, fill="both", padx=2)
            tk.Label(col, text=label, font=("Courier New",7),
                     fg=color, bg=PANEL).pack()
            dv = tk.DoubleVar(value=default)
            sv = tk.StringVar(value=f"{default:.1f}")
            sc = tk.Scale(col, from_=hi, to=lo, orient="vertical",
                          variable=dv, resolution=0.1,
                          length=90, showvalue=False,
                          bg=PANEL, fg=color, troughcolor="#1a1a2e",
                          highlightthickness=0, bd=0,
                          activebackground=color,
                          command=lambda v, k=key: self._manual(k, float(v)))
            sc.pack()
            tk.Label(col, textvariable=sv, font=("Courier New",7),
                     fg="#555", bg=PANEL).pack()
            self.s_var[key] = (dv, sv)

        # Dynamics / colour sliders
        dp = self._panel(R, "▸ DYNAMICS & COLOUR  —  AI adjusts live")
        dp.pack(fill="x", padx=14, pady=(6,0))
        drow = tk.Frame(dp, bg=PANEL)
        drow.pack(fill="x")
        for key, label, lo, hi, default, color in self.PARAM_SLIDERS:
            col = tk.Frame(drow, bg=PANEL)
            col.pack(side="left", expand=True, fill="both", padx=3)
            tk.Label(col, text=label, font=("Courier New",7),
                     fg=color, bg=PANEL).pack()
            dv = tk.DoubleVar(value=default)
            sv = tk.StringVar(value=f"{default:.2f}")
            sc = tk.Scale(col, from_=hi, to=lo, orient="vertical",
                          variable=dv, resolution=0.01,
                          length=70, showvalue=False,
                          bg=PANEL, fg=color, troughcolor="#1a1a2e",
                          highlightthickness=0, bd=0,
                          activebackground=color,
                          command=lambda v, k=key: self._manual(k, float(v)))
            sc.pack()
            tk.Label(col, textvariable=sv, font=("Courier New",7),
                     fg="#555", bg=PANEL).pack()
            self.s_var[key] = (dv, sv)

        # log — always visible by default, toggled via hamburger button
        self._log_frame = self._panel(R, "▸ AI LOG")
        self._log_frame.pack(fill="both", expand=True, padx=14, pady=(8,14))
        self._log_visible = True
        self.log_box = scrolledtext.ScrolledText(
            self._log_frame, height=6, font=("Courier New", 9),
            bg="#0a0a18", fg="#00dd77",
            insertbackground=ACCENT, relief="flat", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
        self._log("Ready. Load a track and press PLAY.")
        # Light up hamburger button since log starts visible
        self.ham_btn.config(fg=ACCENT)

    def _panel(self, parent, title):
        return tk.LabelFrame(parent, text=f"  {title}  ",
                             font=("Courier New",8), fg=DIM,
                             bg=PANEL, bd=1, relief="flat",
                             highlightbackground=BORDER, highlightthickness=1)

    def _btn(self, parent, text, cmd, w=None):
        kw = dict(text=text, command=cmd, font=("Courier New",9,"bold"),
                  bg="#111", fg=ACCENT, activebackground="#1a2a1a",
                  activeforeground=ACCENT, relief="flat", bd=0,
                  padx=8, pady=6, cursor="hand2")
        if w: kw["width"] = w
        return tk.Button(parent, **kw)

    def _log(self, msg):
        def _do():
            self.log_box.config(state="normal")
            self.log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(0, _do)

    def _fmt(self, s):
        s = max(0, int(s))
        return f"{s//60}:{s%60:02d}"

    def _set_slider(self, key, val):
        dv, sv = self.s_var[key]
        dv.set(round(val, 2))
        sv.set(f"{val:+.1f}" if key not in ("volume","compression") else f"{val:.2f}")

    def _manual(self, key, val):
        EQ_BANDS = ("sub","bass","warmth","low_mid","mid","upper_mid","presence","treble","air")
        if key in EQ_BANDS:
            self.engine.filter_bank.update_gain(key, val)
        elif key == "volume":       self.engine.vol          = val
        elif key == "comp_low":     self.engine.comp_low     = val
        elif key == "comp_mid":     self.engine.comp_mid     = val
        elif key == "comp_high":    self.engine.comp_high    = val
        elif key == "stereo_width": self.engine.stereo_width = val
        elif key == "exciter":      self.engine.exciter      = val

    def _run_nr(self):
        # Sync passes count to engine (DL mode only)
        if hasattr(self, 'nr_passes'):
            self.engine.noise_reducer.dns_passes = int(self.nr_passes.get())
        """Process the full loaded track with MetricGAN+ in background."""
        if self.engine.data is None:
            self._log("Load a track first."); return
        # save original so we can restore
        self._original_audio = self.engine.data.copy()

        self.nr_btn.config(state="disabled", text="⟳  PROCESSING...")
        self.nr_restore_btn.config(state="disabled")
        self.nr_progress_var.set(0)
        self.nr_status.config(text=f"Running {self.engine.noise_reducer.mode} — {'neural voice separation' if self.engine.noise_reducer.mode == 'demucs' else 'Wiener spectral filter'}...",
                              fg="#ff6666")
        self._log(f"Starting {self.engine.noise_reducer.mode.upper()} processing...")

        nr = self.engine.noise_reducer
        nr.strength = self.nr_strength.get()

        def _on_progress(pct, msg):
            self.after(0, lambda: self._nr_progress(pct, msg))

        def _on_done(cleaned):
            self.after(0, lambda: self._nr_done(cleaned))

        def _on_error(msg):
            self.after(0, lambda: self._nr_error(msg))

        nr.on_progress = _on_progress
        nr.on_done     = _on_done
        nr.on_error    = _on_error
        nr.process_file(self.engine.data)

    def _nr_progress(self, pct, msg):
        self.nr_progress_var.set(pct)
        self.nr_status.config(text=msg, fg="#ff8866")

    def _nr_done(self, cleaned):
        # Atomically swap buffer — playback keeps position, no glitch
        was_playing = self.engine.playing
        pos = self.engine.pos
        if was_playing:
            self.engine.pause()
        self.engine.data = cleaned
        self.engine.pos  = min(pos, len(cleaned) - 1)
        if was_playing:
            self.engine.play()
        self.nr_progress_var.set(100)
        self.nr_status.config(
            text="✓ CLEANED — noise suppressed. RE-CLEAN to reprocess or ↺ restore original.",
            fg="#00ff88")
        self.nr_btn.config(state="normal", text="🧹  CLEAN TRACK")
        self.nr_restore_btn.config(state="normal", fg="#ff6666",
                                   bg="#1a0000", activeforeground="#ff8888")
        self._log("✓ Noise removal complete — buffer swapped cleanly.")

    def _nr_error(self, msg):
        self.nr_status.config(text=f"✗ {msg}", fg="#ff4444")
        self.nr_btn.config(state="normal", text="🧹  CLEAN TRACK")
        self._log(f"Noise removal error: {msg}")

    def _restore_original(self):
        if self._original_audio is None:
            return
        was_playing = self.engine.playing
        pos = self.engine.pos
        if was_playing:
            self.engine.pause()
        self.engine.data = self._original_audio.copy()
        self.engine.pos  = min(pos, len(self.engine.data) - 1)
        if was_playing:
            self.engine.play()
        self.nr_progress_var.set(0)
        self.nr_status.config(
            text="Original audio restored.", fg="#888")
        self.nr_restore_btn.config(state="disabled", fg="#555", bg="#111")
        self._log("Original audio restored.")

    def _set_nr_mode(self, mode):
        self.engine.noise_reducer.mode = mode
        # Reset all buttons to inactive style
        self.btn_wiener.config(bg="#111", fg="#555")
        self.btn_deepfilter.config(bg="#111", fg="#555")
        self.btn_demucs.config(bg="#111", fg="#555")

        if mode == "deepfilter":
            self.btn_deepfilter.config(bg="#0a1a0a", fg="#44ff88")
            self.nr_mode_badge.config(
                text="● DNS64 neural suppression + Wiener gate",
                fg="#44ff88")
            self.nr_status.config(
                text="Deep Learning: DNS64 × N passes → Wiener gate  "
                     "(pip install denoiser)",
                fg="#44ff88")
            self.nr_strength.set(1.0)
            self.nr_passes_frame.pack(side="left", padx=(14,0))
        elif mode == "demucs":
            self.nr_passes_frame.pack_forget()
            self.btn_demucs.config(bg="#1a0a2a", fg="#cc88ff")
            self.nr_mode_badge.config(
                text="● Demucs AI voice isolation active",
                fg="#cc88ff")
            self.nr_status.config(
                text="Voice Isolation mode — separates voice from ALL background",
                fg="#cc88ff")
        else:
            self.nr_passes_frame.pack_forget()
            self.btn_wiener.config(bg="#001a2a", fg="#44aaff")
            self.nr_mode_badge.config(
                text="● Wiener spectral filter active",
                fg="#44aaff")
            self.nr_status.config(
                text="Spectral mode — Wiener filter + A-weighting + VAD",
                fg="#44aaff")

    def _set_nr_strength(self, val):
        pass   # handled by trace on nr_strength var

    def _learn_noise(self):
        pass

    def _toggle_ai(self):
        self.engine.ai_enabled = self.ai_on.get()
        self._log("Signal AI " + ("ENABLED" if self.ai_on.get() else "DISABLED"))

    def _save(self):
        if self.engine.data is None:
            self._log("Load a track first."); return

        current = self.file_lbl.cget("text")
        stem = os.path.splitext(current)[0] if current != "No track loaded" else "enhanced"
        default_name = stem + "_enhanced.wav"

        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[
                ("WAV",  "*.wav"),
                ("FLAC", "*.flac"),
                ("MP3",  "*.mp3"),
                ("All",  "*.*"),
            ],
            title="Save enhanced audio"
        )
        if not path: return

        self.save_btn.config(state="disabled", text="\u29d6  SAVING...")
        self.save_status.config(text="Exporting...", fg="#ffaa00")
        self._log("Saving to " + os.path.basename(path))

        def _do_save():
            try:
                import soundfile as sf
                import numpy as np
                ext = os.path.splitext(path)[1].lower()
                data = self.engine.data.copy()
                sr   = self.engine.SR

                if ext == ".mp3":
                    # Try lameenc first (pure Python, no ffmpeg needed)
                    # Install: pip install lameenc
                    try:
                        import lameenc
                        # Convert to 16-bit PCM integers
                        pcm = (data * 32767).clip(-32768, 32767).astype(np.int16)
                        enc = lameenc.Encoder()
                        enc.set_bit_rate(320)
                        enc.set_in_sample_rate(sr)
                        enc.set_channels(data.shape[1] if data.ndim > 1 else 1)
                        enc.set_quality(2)  # 2 = high, 7 = fastest
                        # lameenc expects interleaved stereo bytes (L R L R ...)
                        if data.ndim > 1:
                            interleaved = np.empty((len(pcm) * 2,), dtype=np.int16)
                            interleaved[0::2] = pcm[:, 0]
                            interleaved[1::2] = pcm[:, 1]
                            mp3_data = enc.encode(interleaved.tobytes())
                        else:
                            mp3_data = enc.encode(pcm.tobytes())
                        mp3_data += enc.flush()
                        with open(path, "wb") as fout:
                            fout.write(mp3_data)
                    except ImportError:
                        # Fallback: save as WAV
                        wav_path = path.replace(".mp3", ".wav")
                        sf.write(wav_path, data, sr, subtype="PCM_24")
                        msg = "lameenc not installed (pip install lameenc) — saved as WAV: " + os.path.basename(wav_path)
                        self.after(0, lambda m=msg: self.save_status.config(text=m, fg="#ffaa00"))
                        self.after(0, lambda: self.save_btn.config(state="normal", text="\U0001f4be  SAVE ENHANCED"))
                        self.after(0, lambda m=msg: self._log(m))
                        return
                elif ext == ".flac":
                    sf.write(path, data, sr, format="FLAC", subtype="PCM_24")
                else:
                    sf.write(path, data, sr, subtype="PCM_24")

                msg = "\u2713 Saved: " + os.path.basename(path)
                self.after(0, lambda m=msg: self.save_status.config(text=m, fg="#44ff88"))
                self.after(0, lambda: self.save_btn.config(state="normal", text="\U0001f4be  SAVE ENHANCED"))
                self.after(0, lambda m=msg: self._log(m))

            except ImportError:
                msg = "soundfile not installed - run: pip install soundfile"
                self.after(0, lambda m=msg: self.save_status.config(text=m, fg="#ff4444"))
                self.after(0, lambda: self.save_btn.config(state="normal", text="\U0001f4be  SAVE ENHANCED"))
                self.after(0, lambda m=msg: self._log(m))
            except Exception as e:
                msg = "Save error: " + str(e)
                self.after(0, lambda m=msg: self.save_status.config(text=m, fg="#ff4444"))
                self.after(0, lambda: self.save_btn.config(state="normal", text="\U0001f4be  SAVE ENHANCED"))
                self.after(0, lambda m=msg: self._log(m))

        threading.Thread(target=_do_save, daemon=True).start()

    def _open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio","*.mp3 *.wav *.flac *.ogg *.m4a *.aac"),
                       ("All","*.*")])
        if not path: return
        if self.is_playing: self._stop()
        self._log(f"Loading: {os.path.basename(path)}")
        try:
            dur = self.engine.load(path)
            self.file_lbl.config(text=os.path.basename(path), fg=ACCENT)
            self.prog.config(to=dur)
            self._log(f"Loaded — {self._fmt(dur)}")
            # Auto noise suppression if enabled
            if NOISEREDUCE_OK and self.nr_auto.get():
                self._log("Auto noise suppression starting...")
                self.after(300, self._run_nr)
        except Exception as e:
            self._log(f"ERROR: {e}")

    def _play_pause(self):
        if self.engine.data is None:
            self._log("Load a track first."); return
        if self.is_playing:
            self._stop()
        else:
            self.engine.play()
            self.is_playing = True
            self.play_btn.config(text="⏸  PAUSE")
            self.ai_badge.config(text="● LIVE", fg=ACCENT)
            self._log("▶ Playing — Signal AI active")
            self._ui_loop()

    def _stop(self):
        self.engine.pause()
        self.is_playing = False
        self.play_btn.config(text="▶  PLAY")
        self.ai_badge.config(text="● IDLE", fg="#333")
        if self.after_id:
            self.after_cancel(self.after_id)

    def _seek(self, val):
        if self._seeking:
            return   # called by prog.set() — ignore, not a user drag
        self.engine.seek(float(val))

    def _toggle_repeat(self):
        self.repeat_enabled = not self.repeat_enabled
        status = "ON" if self.repeat_enabled else "OFF"
        self.repeat_btn.config(text=f"🔁  REPEAT {status}",
                               fg=ACCENT if self.repeat_enabled else "#555")
        self._log(f"Repeat: {status}")

    def _reset_ai(self):
        self.engine.signal_ai = SignalAI()
        self._log("Signal AI reset.")

    def _toggle_log(self):
        """Show/hide the AI log dropdown panel."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_frame.pack(fill="both", expand=True, padx=14, pady=(6,10))
            self.ham_btn.config(fg=ACCENT)
        else:
            self._log_frame.pack_forget()
            self.ham_btn.config(fg="#555")

    def _ui_loop(self):
        if not self.is_playing: return
        t   = self.engine.current_time
        dur = self.engine.duration
        self._seeking = True
        self.prog.set(t)
        self._seeking = False
        self.time_lbl.config(text=f"{self._fmt(t)} / {self._fmt(dur)}")
        self._draw_spectrum()

        # Track finished
        if not self.engine.playing and t > 1:
            if self.repeat_enabled:
                # Full reset: position, filter state, AI state, spectrum, progress
                self.engine.pause()
                self.engine.filter_bank = FilterBank(2)
                self.engine.signal_ai   = SignalAI()
                self._spec_data         = np.zeros(64)
                self._seeking = True
                self.prog.set(0)
                self._seeking = False
                self.time_lbl.config(text=f"0:00 / {self._fmt(dur)}")
                with self.engine._pos_lock:
                    self.engine.pos = 0
                self.engine.playing = True
                self.engine.play()
                self._log("↻ Repeating track...")
                self.after_id = self.after(80, self._ui_loop)
            else:
                self._stop()
                self._spec_data = np.zeros(64)
                self._draw_spectrum()
                self._log("✓ Playback complete.")
            return

        self.after_id = self.after(80, self._ui_loop)

    def _cb_spec(self, fft_data):
        self.after(0, lambda d=fft_data: self._draw_spectrogram(d))
        n    = len(fft_data)
        bins = 64
        step = max(1, n // bins)
        spec = np.array([fft_data[i*step:(i+1)*step].mean() for i in range(bins)])
        mx   = spec.max()
        self._spec_data = spec / mx if mx > 0 else spec

    def _cb_targets(self, targets):
        def _apply():
            for key, val in targets.items():
                if key in self.s_var:
                    self._set_slider(key, val)
            self._update_meters(targets)
        self.after(0, _apply)


    # ── De-esser ──────────────────────────────────────────────────────────────

    def _toggle_deq(self):
        enabled = self.deq_enabled.get()
        self.engine.dynamic_eq.enabled = enabled
        self._log(f"Dynamic EQ {'ENABLED' if enabled else 'disabled'}")

    def _deq_set(self, band_name, key, value):
        self.engine.dynamic_eq.bands[band_name][key] = value


    def _toggle_deq(self):
        enabled = self.deq_enabled.get()
        self.engine.dynamic_eq.enabled = enabled
        self._log(f"Dynamic EQ {'ENABLED' if enabled else 'disabled'}")

    def _toggle_deesser(self):
        enabled = self.de_enabled.get()
        self.engine.de_esser.enabled = enabled
        self._log(f"De-esser {'ENABLED' if enabled else 'disabled'}")

    # ── Transient shaper ──────────────────────────────────────────────────────
    def _toggle_transient(self):
        enabled = self.tr_enabled.get()
        self.engine.transient_shaper.enabled = enabled
        self._log(f"Transient shaper {'ENABLED' if enabled else 'disabled'}")

    # ── Per-section AI ────────────────────────────────────────────────────────
    def _toggle_sections(self):
        enabled = self.sa_enabled.get()
        self.engine.section_analyzer.enabled = enabled
        if enabled and self.engine.data is not None:
            self.sa_status.config(text="Analysing sections...", fg="#aa88ff")
            self._log("Section analyzer running in background...")
            self.engine.section_analyzer.analyze_file(
                self.engine.data,
                on_done=lambda secs: self.after(0, lambda: self._sections_done(secs))
            )
        elif not enabled:
            self.sa_status.config(text="Section AI disabled", fg="#555")

    def _sections_done(self, sections):
        self.sa_status.config(
            text=f"✓ {len(sections)} sections detected: " +
                 "  ".join(f"{l.upper()}({e-s:.0f}s)" for s,e,l in sections),
            fg="#aa88ff")
        self._log(f"Sections: " + ", ".join(f"{l}@{s:.0f}s" for s,e,l in sections))
        self._draw_section_timeline(sections)

    def _draw_section_timeline(self, sections):
        c   = self.sa_canvas
        c.delete("all")
        dur = self.engine.duration
        if dur <= 0: return
        w   = c.winfo_width() or 820
        h   = 18
        COLORS = {"intro":"#446688","verse":"#4488aa","chorus":"#aa4488",
                  "bridge":"#44aa88","outro":"#886644"}
        for start, end, label in sections:
            x0 = int(start / dur * w)
            x1 = int(end   / dur * w)
            col = COLORS.get(label, "#555577")
            c.create_rectangle(x0, 1, x1-1, h-1, fill=col, outline="#222")
            if x1 - x0 > 30:
                c.create_text((x0+x1)//2, h//2, text=label.upper(),
                              font=("Courier New",7,"bold"), fill="white")

    def _cb_section(self, label):
        COLORS = {"intro":"#446688","verse":"#4488aa","chorus":"#ff44aa",
                  "bridge":"#44ffaa","outro":"#886644"}
        col = COLORS.get(label, "#aaaaaa")
        self.section_lbl.config(text=f"◉ {label.upper()}", fg=col)

    # ── Auto-master ───────────────────────────────────────────────────────────
    def _run_automaster(self):
        if self.engine.data is None:
            self._log("Load a track first."); return
        self.am_btn.config(state="disabled", text="⟳  MASTERING...")
        self._log("Auto-master started...")
        am = self.engine.auto_master
        am.on_log  = lambda msg: self.after(0, lambda m=msg: self._log(f"  {m}"))
        am.on_done = lambda: self.after(0, self._automaster_done)
        am.run()

    def _automaster_done(self):
        self.am_btn.config(state="normal", text="⚡  AUTO-MASTER")
        self._log("✓ Auto-master complete — settings applied live")

    # ── Spectrogram ───────────────────────────────────────────────────────────
    def _draw_spectrogram(self, fft_data):
        """Add a new row to the rolling waterfall spectrogram."""
        BINS = 128
        # Downsample FFT to BINS
        if len(fft_data) > BINS:
            step = len(fft_data) // BINS
            row  = fft_data[:step*BINS].reshape(BINS, step).max(axis=1)
        else:
            row = np.pad(fft_data, (0, BINS - len(fft_data)))

        # Normalise to 0-1
        peak = row.max()
        if peak > 0:
            row = row / peak

        self._spectrogram.append(row.copy())
        if len(self._spectrogram) > self._SGRAM_ROWS:
            self._spectrogram.pop(0)

        c = self.sgram_cv
        c.delete("all")
        cw = c.winfo_width() or 830
        ch = 80
        n_rows = len(self._spectrogram)
        if n_rows == 0: return
        row_h  = ch / self._SGRAM_ROWS
        bin_w  = cw / BINS

        for ri, row_data in enumerate(self._spectrogram):
            y0 = int(ri * row_h)
            y1 = max(y0+1, int((ri+1) * row_h))
            for bi, val in enumerate(row_data):
                if val < 0.01: continue
                x0 = int(bi * bin_w)
                x1 = max(x0+1, int((bi+1) * bin_w))
                # colour: black→blue→cyan→green→yellow→red (hot colormap)
                v  = float(val)
                if v < 0.25:
                    r,g,b = 0, 0, int(v*4*200)
                elif v < 0.5:
                    t = (v-0.25)*4
                    r,g,b = 0, int(t*220), 200
                elif v < 0.75:
                    t = (v-0.5)*4
                    r,g,b = int(t*255), 220, int((1-t)*200)
                else:
                    t = (v-0.75)*4
                    r,g,b = 255, int((1-t)*220), 0
                c.create_rectangle(x0, y0, x1, y1,
                                   fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

    def _draw_spectrum(self):
        c = self.spec_cv
        c.delete("all")
        w = c.winfo_width() or 830
        h = 70
        n = len(self._spec_data)
        bw = w / n
        for i, v in enumerate(self._spec_data):
            hue   = int(120 + i / n * 200)
            r,g,b = self._hsl(hue, 1.0, 0.5)
            col   = f"#{r:02x}{g:02x}{b:02x}"
            bh    = max(2, int(v * (h-4)))
            c.create_rectangle(i*bw, h-bh, i*bw+max(1,bw-1), h, fill=col, outline="")

    def _hsl(self, h, s, l):
        h /= 360
        def hue2rgb(p,q,t):
            t %= 1
            if t < 1/6: return p+(q-p)*6*t
            if t < 1/2: return q
            if t < 2/3: return p+(q-p)*(2/3-t)*6
            return p
        q = l*(1+s) if l < 0.5 else l+s-l*s
        p = 2*l-q
        return (int(hue2rgb(p,q,h+1/3)*255),
                int(hue2rgb(p,q,h    )*255),
                int(hue2rgb(p,q,h-1/3)*255))

    def _update_meters(self, targets):
        for key, (bar, lo, hi, color) in self.meters.items():
            val = targets.get(key, lo)
            pct = max(0, min(1, (val-lo)/(hi-lo)))
            w   = bar.winfo_width() or 80
            bar.delete("all")
            bar.create_rectangle(0,0,int(pct*w),5, fill=color, outline="")
            bar.create_rectangle(int(pct*w),0,w,5, fill="#111", outline="")

    def _gemini(self):
        key = self.api_var.get().strip()
        if not key: self._log("Enter Gemini API key."); return
        if self.engine.data is None: self._log("Load a track first."); return
        self.gem_btn.config(state="disabled", text="⟳  SCANNING...")
        fname = self.file_lbl.cget("text")
        desc  = self.desc_var.get().strip() or fname
        self._log("Sending to Gemini...")

        def _run():
            try:
                r = gemini_analyze(fname, desc, key)
                self.after(0, lambda: self._apply_gemini(r))
            except Exception as e:
                self.after(0, lambda: self._gem_err(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _apply_gemini(self, r):
        self.engine.signal_ai.set_genre_targets(
            lufs          = float(r.get("target_lufs",       -18)),
            bass_frac     = float(r.get("target_bass_frac",  0.18)),
            brightness    = float(r.get("target_brightness", 0.38)),
            compression   = float(r.get("compression",       2.0)),
            dynamic_range = float(r.get("dynamic_range",     6.0)),
        )
        genre = r.get("genre","?")
        self.gem_lbl.config(text=f"Genre: {genre}")
        self.gem_btn.config(state="normal", text="✦  GEMINI GENRE SCAN")
        self._log(f"Gemini → {genre}: {r.get('analysis','')}")
        for e in r.get("enhancements",[]): self._log(f"  ✓ {e}")

    def _gem_err(self, msg):
        self.gem_btn.config(state="normal", text="✦  GEMINI GENRE SCAN")
        self._log(f"Gemini error: {msg}")


if __name__ == "__main__":
    app = App()
    app.mainloop()