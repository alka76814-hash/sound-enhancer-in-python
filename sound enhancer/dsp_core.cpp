/*
 * dsp_core.cpp — SONIC AI C++ DSP Engine
 * ========================================
 * Full real-time audio processing chain implemented in C++.
 * Exposed as a plain C API so Python ctypes can load it directly
 * on Windows (.dll), macOS (.dylib), or Linux (.so).
 *
 * Compile (Windows MinGW):
 *   g++ -O3 -march=native -shared -fPIC -o dsp_core.pyd dsp_core.cpp
 *
 * Compile (Windows MSVC):
 *   cl /O2 /LD dsp_core.cpp /Fe:dsp_core.pyd
 *
 * Compile (macOS):
 *   g++ -O3 -march=native -shared -fPIC -o dsp_core.so dsp_core.cpp
 *
 * Compile (Linux):
 *   g++ -O3 -march=native -shared -fPIC -o dsp_core.so dsp_core.cpp
 *
 * All buffers are interleaved stereo float32: [L0,R0,L1,R1,...,Ln,Rn]
 * n_frames = number of sample pairs (buffer length / 2)
 */

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>

#ifdef _WIN32
  #define EXPORT extern "C" __declspec(dllexport)
#else
  #define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ═══════════════════════════════════════════════════════════════════
//  CONSTANTS
// ═══════════════════════════════════════════════════════════════════

static const float PI  = 3.14159265358979323846f;
static const float TAU = 6.28318530717958647692f;

// ═══════════════════════════════════════════════════════════════════
//  BIQUAD FILTER  — Direct Form II Transposed
// ═══════════════════════════════════════════════════════════════════
//
//  One biquad state holds two delay elements per channel.
//  Layout: [w1_L, w2_L, w1_R, w2_R]

struct BiquadState {
    float w[4] = {0,0,0,0};  // w1_L, w2_L, w1_R, w2_R
};

struct BiquadCoeffs {
    float b0, b1, b2, a1, a2;
};

// Compute biquad coefficients
static BiquadCoeffs make_peaking(float fc, float gain_db, float Q, float sr) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / sr;
    float cs = cosf(w0), sn = sinf(w0);
    float alpha = sn / (2.0f * Q);
    float b0 =  1.0f + alpha * A;
    float b1 = -2.0f * cs;
    float b2 =  1.0f - alpha * A;
    float a0 =  1.0f + alpha / A;
    float a1 = -2.0f * cs;
    float a2 =  1.0f - alpha / A;
    return { b0/a0, b1/a0, b2/a0, a1/a0, a2/a0 };
}

static BiquadCoeffs make_low_shelf(float fc, float gain_db, float S, float sr) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / sr;
    float cs = cosf(w0), sn = sinf(w0);
    float alpha = sn / 2.0f * sqrtf((A + 1.0f/A) * (1.0f/S - 1.0f) + 2.0f);
    float sqA2 = 2.0f * sqrtf(A) * alpha;
    float a0 =        (A+1) + (A-1)*cs + sqA2;
    float b0 =   A * ((A+1) - (A-1)*cs + sqA2);
    float b1 = 2*A * ((A-1) - (A+1)*cs);
    float b2 =   A * ((A+1) - (A-1)*cs - sqA2);
    float a1 =  -2 * ((A-1) + (A+1)*cs);
    float a2 =        (A+1) + (A-1)*cs - sqA2;
    return { b0/a0, b1/a0, b2/a0, a1/a0, a2/a0 };
}

static BiquadCoeffs make_high_shelf(float fc, float gain_db, float S, float sr) {
    float A  = powf(10.0f, gain_db / 40.0f);
    float w0 = TAU * fc / sr;
    float cs = cosf(w0), sn = sinf(w0);
    float alpha = sn / 2.0f * sqrtf((A + 1.0f/A) * (1.0f/S - 1.0f) + 2.0f);
    float sqA2 = 2.0f * sqrtf(A) * alpha;
    float a0 =        (A+1) - (A-1)*cs + sqA2;
    float b0 =   A * ((A+1) + (A-1)*cs + sqA2);
    float b1 =-2*A * ((A-1) + (A+1)*cs);
    float b2 =   A * ((A+1) + (A-1)*cs - sqA2);
    float a1 =   2 * ((A-1) - (A+1)*cs);
    float a2 =        (A+1) - (A-1)*cs - sqA2;
    return { b0/a0, b1/a0, b2/a0, a1/a0, a2/a0 };
}

static BiquadCoeffs make_highpass(float fc, float Q, float sr) {
    float w0 = TAU * fc / sr;
    float cs = cosf(w0), sn = sinf(w0);
    float alpha = sn / (2.0f * Q);
    float a0 =  1.0f + alpha;
    float b0 = (1.0f + cs) / 2.0f;
    float b1 = -(1.0f + cs);
    float b2 = (1.0f + cs) / 2.0f;
    float a1 = -2.0f * cs;
    float a2 =  1.0f - alpha;
    return { b0/a0, b1/a0, b2/a0, a1/a0, a2/a0 };
}

// Process one biquad in-place on interleaved stereo buffer
static void biquad_process(float* buf, int n_frames,
                            const BiquadCoeffs& c, BiquadState& s) {
    for (int i = 0; i < n_frames; ++i) {
        // Left channel
        float xL = buf[i*2];
        float yL = c.b0*xL + s.w[0];
        s.w[0]   = c.b1*xL - c.a1*yL + s.w[1];
        s.w[1]   = c.b2*xL - c.a2*yL;
        buf[i*2] = yL;
        // Right channel
        float xR = buf[i*2+1];
        float yR = c.b0*xR + s.w[2];
        s.w[2]   = c.b1*xR - c.a1*yR + s.w[3];
        s.w[3]   = c.b2*xR - c.a2*yR;
        buf[i*2+1] = yR;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  9-BAND EQ FILTER BANK
// ═══════════════════════════════════════════════════════════════════

// Band definitions: {fc_hz, type, Q/S}
//  type: 0=peaking, 1=low_shelf, 2=high_shelf, 3=highpass
struct BandDef { float fc; int type; float Q; };

static const BandDef BAND_DEFS[9] = {
    {  30.0f, 3, 0.707f },  // sub      — highpass (rumble cut)
    {  80.0f, 0, 1.0f   },  // bass
    { 200.0f, 1, 0.7f   },  // warmth   — low shelf
    { 500.0f, 0, 1.2f   },  // low_mid
    {1000.0f, 0, 1.0f   },  // mid
    {2500.0f, 0, 1.2f   },  // upper_mid
    {4500.0f, 0, 1.2f   },  // presence
    { 8000.f, 0, 1.0f   },  // treble
    {16000.f, 2, 0.7f   },  // air      — high shelf
};

struct FilterBankState {
    BiquadCoeffs coeffs[9];
    BiquadState  states[9];
    float        gains[9];   // current gain_db per band
    float        sr;
};

EXPORT FilterBankState* filterbank_create(float sr) {
    auto* fb = new FilterBankState();
    fb->sr = sr;
    for (int i = 0; i < 9; ++i) fb->gains[i] = 0.0f;
    // build identity filters (0 dB)
    for (int i = 0; i < 9; ++i) {
        const auto& d = BAND_DEFS[i];
        switch (d.type) {
            case 0: fb->coeffs[i] = make_peaking   (d.fc, 0.0f, d.Q, sr); break;
            case 1: fb->coeffs[i] = make_low_shelf (d.fc, 0.0f, d.Q, sr); break;
            case 2: fb->coeffs[i] = make_high_shelf(d.fc, 0.0f, d.Q, sr); break;
            case 3: fb->coeffs[i] = make_highpass  (d.fc, d.Q,       sr); break;
        }
    }
    return fb;
}

EXPORT void filterbank_destroy(FilterBankState* fb) { delete fb; }

EXPORT void filterbank_set_gain(FilterBankState* fb, int band, float gain_db) {
    if (band < 0 || band >= 9) return;
    if (fabsf(gain_db - fb->gains[band]) < 0.01f) return;
    fb->gains[band] = gain_db;
    const auto& d = BAND_DEFS[band];
    switch (d.type) {
        case 0: fb->coeffs[band] = make_peaking   (d.fc, gain_db, d.Q, fb->sr); break;
        case 1: fb->coeffs[band] = make_low_shelf (d.fc, gain_db, d.Q, fb->sr); break;
        case 2: fb->coeffs[band] = make_high_shelf(d.fc, gain_db, d.Q, fb->sr); break;
        case 3: fb->coeffs[band] = make_highpass  (d.fc, d.Q,          fb->sr); break;
    }
}

EXPORT void filterbank_process(FilterBankState* fb, float* buf, int n_frames) {
    for (int i = 0; i < 9; ++i) {
        biquad_process(buf, n_frames, fb->coeffs[i], fb->states[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  NOISE GATE
// ═══════════════════════════════════════════════════════════════════

EXPORT void noise_gate(float* buf, int n_frames, float threshold_db) {
    float thr = powf(10.0f, threshold_db / 20.0f);
    for (int i = 0; i < n_frames; ++i) {
        float L = buf[i*2], R = buf[i*2+1];
        float rms = sqrtf((L*L + R*R) * 0.5f);
        if (rms < thr) {
            buf[i*2]   = 0.0f;
            buf[i*2+1] = 0.0f;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  MULTIBAND COMPRESSOR  (3 bands: lo/mid/hi)
// ═══════════════════════════════════════════════════════════════════

struct CompressorState {
    // crossover filters (2nd order Linkwitz-Riley at 200Hz and 2kHz)
    BiquadCoeffs lo_lp[2], lo_hp[2];
    BiquadCoeffs hi_lp[2], hi_hp[2];
    BiquadState  lo_lp_s[2], lo_hp_s[2];
    BiquadState  hi_lp_s[2], hi_hp_s[2];
    // envelope followers per band
    float env_lo, env_mid, env_hi;
    float atk, rel;
};

static BiquadCoeffs make_butterworth_lp(float fc, float Q, float sr) {
    float w0 = TAU * fc / sr;
    float cs = cosf(w0), sn = sinf(w0);
    float alpha = sn / (2.0f * Q);
    float a0 =  1.0f + alpha;
    float b1 =  1.0f - cs;
    float b0 = b1 / 2.0f;
    float b2 = b0;
    float a1 = -2.0f * cs;
    float a2 =  1.0f - alpha;
    return { b0/a0, b1/a0, b2/a0, a1/a0, a2/a0 };
}

static BiquadCoeffs make_butterworth_hp(float fc, float Q, float sr) {
    float w0 = TAU * fc / sr;
    float cs = cosf(w0), sn = sinf(w0);
    float alpha = sn / (2.0f * Q);
    float a0 =  1.0f + alpha;
    float b0 = (1.0f + cs) / 2.0f;
    float b1 = -(1.0f + cs);
    float b2 = b0;
    float a1 = -2.0f * cs;
    float a2 =  1.0f - alpha;
    return { b0/a0, b1/a0, b2/a0, a1/a0, a2/a0 };
}

EXPORT CompressorState* compressor_create(float sr) {
    auto* c = new CompressorState();
    memset(c, 0, sizeof(*c));
    float Q_lr = 0.7071f;  // Butterworth Q for 2nd order LR
    for (int i = 0; i < 2; ++i) {
        c->lo_lp[i] = make_butterworth_lp(200.0f,  Q_lr, sr);
        c->lo_hp[i] = make_butterworth_hp(200.0f,  Q_lr, sr);
        c->hi_lp[i] = make_butterworth_lp(2000.0f, Q_lr, sr);
        c->hi_hp[i] = make_butterworth_hp(2000.0f, Q_lr, sr);
    }
    c->atk = expf(-1.0f / (sr * 0.003f));   // 3ms attack
    c->rel = expf(-1.0f / (sr * 0.100f));   // 100ms release
    return c;
}

EXPORT void compressor_destroy(CompressorState* c) { delete c; }

// Static helper: compress mono signal in-place using precomputed envelope
static void compress_band(float* lo, float* hi,
                          float depth, float& env,
                          float atk, float rel, int n_frames) {
    if (depth < 0.1f) return;
    const float threshold = 0.25f;
    const float ratio     = 1.0f + depth;
    for (int i = 0; i < n_frames; ++i) {
        float lvl = fabsf(lo[i] + hi[i]);
        float coef = (lvl > env) ? atk : rel;
        env = coef * env + (1.0f - coef) * lvl;
        if (env > threshold) {
            float gr = threshold * powf(env / threshold, 1.0f / ratio) / env;
            lo[i] *= gr;
            hi[i] *= gr;
        }
    }
}

EXPORT void compressor_process(CompressorState* c, float* buf, int n_frames,
                               float lo_depth, float mid_depth, float hi_depth) {
    // Split into separate L/R arrays for filter processing
    float* lo_L = new float[n_frames];
    float* lo_R = new float[n_frames];
    float* mid_L = new float[n_frames];
    float* mid_R = new float[n_frames];
    float* hi_L  = new float[n_frames];
    float* hi_R  = new float[n_frames];

    // Deinterleave
    for (int i = 0; i < n_frames; ++i) {
        lo_L[i] = mid_L[i] = hi_L[i] = buf[i*2];
        lo_R[i] = mid_R[i] = hi_R[i] = buf[i*2+1];
    }

    // We use a simplified approach: apply gain reduction per sample using
    // the full stereo sidechain to avoid image artifacts
    const float threshold = 0.25f;

    // Low band (0–200 Hz): apply LR lowpass twice
    // Create temp interleaved buffer for band
    float* tmp = new float[n_frames * 2];
    
    auto deinterleave_to = [&](float* dstL, float* dstR, float* src, int n) {
        for (int i = 0; i < n; ++i) { dstL[i] = src[i*2]; dstR[i] = src[i*2+1]; }
    };
    auto interleave_from = [&](float* dst, float* srcL, float* srcR, int n) {
        for (int i = 0; i < n; ++i) { dst[i*2] = srcL[i]; dst[i*2+1] = srcR[i]; }
    };

    // --- Low band ---
    memcpy(tmp, buf, n_frames * 2 * sizeof(float));
    for (int p = 0; p < 2; ++p) biquad_process(tmp, n_frames, c->lo_lp[p], c->lo_lp_s[p]);
    deinterleave_to(lo_L, lo_R, tmp, n_frames);

    // --- High band ---
    memcpy(tmp, buf, n_frames * 2 * sizeof(float));
    for (int p = 0; p < 2; ++p) biquad_process(tmp, n_frames, c->hi_hp[p], c->hi_hp_s[p]);
    deinterleave_to(hi_L, hi_R, tmp, n_frames);

    // --- Mid band (what's left) ---
    for (int i = 0; i < n_frames; ++i) {
        mid_L[i] = buf[i*2]   - lo_L[i] - hi_L[i];
        mid_R[i] = buf[i*2+1] - lo_R[i] - hi_R[i];
    }

    // Per-band gain reduction using stereo sidechain
    auto compress_stereo = [&](float* bL, float* bR, float depth, float& env) {
        if (depth < 0.1f) return;
        float ratio = 1.0f + depth;
        for (int i = 0; i < n_frames; ++i) {
            float lvl = sqrtf((bL[i]*bL[i] + bR[i]*bR[i]) * 0.5f);
            float coef = (lvl > env) ? c->atk : c->rel;
            env = coef * env + (1.0f - coef) * lvl;
            if (env > threshold) {
                float gr = threshold * powf(env / threshold, 1.0f / ratio) / env;
                bL[i] *= gr;
                bR[i] *= gr;
            }
        }
    };

    compress_stereo(lo_L,  lo_R,  lo_depth,  c->env_lo);
    compress_stereo(mid_L, mid_R, mid_depth, c->env_mid);
    compress_stereo(hi_L,  hi_R,  hi_depth,  c->env_hi);

    // Recombine
    for (int i = 0; i < n_frames; ++i) {
        buf[i*2]   = lo_L[i] + mid_L[i] + hi_L[i];
        buf[i*2+1] = lo_R[i] + mid_R[i] + hi_R[i];
    }

    delete[] lo_L; delete[] lo_R;
    delete[] mid_L; delete[] mid_R;
    delete[] hi_L;  delete[] hi_R;
    delete[] tmp;
}

// ═══════════════════════════════════════════════════════════════════
//  HARMONIC EXCITER  — soft-clip saturation on high shelf
// ═══════════════════════════════════════════════════════════════════

struct ExciterState {
    BiquadCoeffs hp;
    BiquadState  hp_s;
};

EXPORT ExciterState* exciter_create(float fc, float sr) {
    auto* e = new ExciterState();
    e->hp   = make_highpass(fc, 0.707f, sr);
    return e;
}

EXPORT void exciter_destroy(ExciterState* e) { delete e; }

EXPORT void exciter_process(ExciterState* e, float* buf, int n_frames, float amount) {
    if (amount < 0.001f) return;
    float tmp[4096*2];
    int chunk = std::min(n_frames, 4096);
    // Work in chunks to avoid large stack allocation
    for (int off = 0; off < n_frames; off += chunk) {
        int n = std::min(chunk, n_frames - off);
        memcpy(tmp, buf + off*2, n * 2 * sizeof(float));
        // high-pass to isolate HF content
        biquad_process(tmp, n, e->hp, e->hp_s);
        // soft-clip saturation: tanh approximation
        for (int i = 0; i < n*2; ++i) {
            float x = tmp[i] * (1.0f + amount * 2.0f);
            // fast tanh: x/(1+|x|)
            tmp[i] = x / (1.0f + fabsf(x));
        }
        // add harmonics back at reduced level
        for (int i = 0; i < n*2; ++i)
            buf[(off + i/2)*2 + (i%2)] += tmp[i] * amount * 0.3f;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  STEREO WIDENER  — M/S matrix
// ═══════════════════════════════════════════════════════════════════

EXPORT void stereo_widen(float* buf, int n_frames, float width) {
    float mid_gain  = 1.0f;
    float side_gain = width;
    for (int i = 0; i < n_frames; ++i) {
        float L = buf[i*2], R = buf[i*2+1];
        float M = (L + R) * 0.5f * mid_gain;
        float S = (L - R) * 0.5f * side_gain;
        buf[i*2]   = M + S;
        buf[i*2+1] = M - S;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  TRUE PEAK LIMITER  — lookahead brickwall
// ═══════════════════════════════════════════════════════════════════

EXPORT void true_peak_limit(float* buf, int n_frames, float ceiling_db) {
    float ceiling = powf(10.0f, ceiling_db / 20.0f);
    // Simple peak limiter with instantaneous release and 1-sample lookahead
    float gain = 1.0f;
    for (int i = 0; i < n_frames; ++i) {
        float peak = std::max(fabsf(buf[i*2]), fabsf(buf[i*2+1]));
        if (peak * gain > ceiling)
            gain = ceiling / (peak + 1e-9f);
        else
            gain = std::min(1.0f, gain * 1.0005f);  // slow recovery
        buf[i*2]   *= gain;
        buf[i*2+1] *= gain;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  DE-ESSER  — frequency-targeted dynamic gain reduction
// ═══════════════════════════════════════════════════════════════════

struct DeEsserState {
    BiquadCoeffs  bp[2];    // bandpass (two cascaded biquads for steeper slope)
    BiquadState   bp_s[2];
    float         env;
    float         atk, rel;
};

EXPORT DeEsserState* deesser_create(float freq_lo, float freq_hi, float sr) {
    auto* d = new DeEsserState();
    memset(d, 0, sizeof(*d));
    float fc_mid = sqrtf(freq_lo * freq_hi);
    float Q = fc_mid / (freq_hi - freq_lo);
    // Two cascaded peaking filters centred on the sibilance band
    d->bp[0] = make_peaking(fc_mid * 0.85f, 12.0f, Q * 0.8f, sr);
    d->bp[1] = make_peaking(fc_mid * 1.15f, 12.0f, Q * 0.8f, sr);
    d->atk = expf(-1.0f / (sr * 0.002f));   // 2ms
    d->rel = expf(-1.0f / (sr * 0.080f));   // 80ms
    return d;
}

EXPORT void deesser_destroy(DeEsserState* d) { delete d; }

EXPORT void deesser_process(DeEsserState* d, float* buf, int n_frames,
                            float threshold_db, float ratio) {
    float thr = powf(10.0f, threshold_db / 20.0f);
    float* sib = new float[n_frames * 2];
    memcpy(sib, buf, n_frames * 2 * sizeof(float));

    // Isolate sibilance band in sidechain
    for (int p = 0; p < 2; ++p)
        biquad_process(sib, n_frames, d->bp[p], d->bp_s[p]);

    for (int i = 0; i < n_frames; ++i) {
        float lvl = sqrtf((sib[i*2]*sib[i*2] + sib[i*2+1]*sib[i*2+1]) * 0.5f);
        float coef = (lvl > d->env) ? d->atk : d->rel;
        d->env = coef * d->env + (1.0f - coef) * lvl;

        if (d->env > thr) {
            // Gain reduction only on the sibilant band
            float gr = thr * powf(d->env / thr, 1.0f / ratio) / (d->env + 1e-9f);
            // Apply as blend: full signal * gr, non-sib signal * (1-gr) compensated
            float scale = gr + (1.0f - gr) * 0.5f;  // partial attenuation
            buf[i*2]   *= scale;
            buf[i*2+1] *= scale;
        }
    }
    delete[] sib;
}

// ═══════════════════════════════════════════════════════════════════
//  TRANSIENT SHAPER  — fast/slow envelope attack/sustain control
// ═══════════════════════════════════════════════════════════════════

struct TransientState {
    float ef, es;  // fast envelope, slow envelope
    float fast_coef, slow_coef;
};

EXPORT TransientState* transient_create(float sr) {
    auto* t = new TransientState();
    t->ef = t->es = 0.0f;
    t->fast_coef = expf(-1.0f / (sr * 0.001f));  // 1ms
    t->slow_coef = expf(-1.0f / (sr * 0.050f));  // 50ms
    return t;
}

EXPORT void transient_destroy(TransientState* t) { delete t; }

EXPORT void transient_process(TransientState* t, float* buf, int n_frames,
                              float attack_db, float sustain_db) {
    if (fabsf(attack_db) < 0.1f && fabsf(sustain_db) < 0.1f) return;
    float att_lin = powf(10.0f, attack_db  / 20.0f);
    float sus_lin = powf(10.0f, sustain_db / 20.0f);

    for (int i = 0; i < n_frames; ++i) {
        float L = buf[i*2], R = buf[i*2+1];
        float lvl = sqrtf((L*L + R*R) * 0.5f) + 1e-9f;

        t->ef = t->fast_coef * t->ef + (1.0f - t->fast_coef) * lvl;
        t->es = t->slow_coef * t->es + (1.0f - t->slow_coef) * lvl;

        float transient = std::max(0.0f, t->ef - t->es);
        float sustain   = t->es;
        float total     = transient + sustain + 1e-9f;

        float g = (transient * att_lin + sustain * sus_lin) / total;
        buf[i*2]   = L * g;
        buf[i*2+1] = R * g;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  FULL DSP CHAIN  — process everything in one call
//  (avoids ctypes overhead of many small calls per callback)
// ═══════════════════════════════════════════════════════════════════

struct DspChain {
    FilterBankState* fb;
    CompressorState* comp;
    ExciterState*    exc;
    TransientState*  trans;
    DeEsserState*    dess;
    float sr;
};

EXPORT DspChain* chain_create(float sr) {
    auto* c    = new DspChain();
    c->fb      = filterbank_create(sr);
    c->comp    = compressor_create(sr);
    c->exc     = exciter_create(3000.0f, sr);
    c->trans   = transient_create(sr);
    c->dess    = deesser_create(5000.0f, 10000.0f, sr);
    c->sr      = sr;
    return c;
}

EXPORT void chain_destroy(DspChain* c) {
    filterbank_destroy(c->fb);
    compressor_destroy(c->comp);
    exciter_destroy(c->exc);
    transient_destroy(c->trans);
    deesser_destroy(c->dess);
    delete c;
}

EXPORT void chain_set_eq(DspChain* c, int band, float gain_db) {
    filterbank_set_gain(c->fb, band, gain_db);
}

EXPORT void chain_process(DspChain* c, float* buf, int n_frames,
                          // compressor
                          float comp_lo, float comp_mid, float comp_hi,
                          // gate
                          float gate_db,
                          // exciter
                          float exciter_amt,
                          // stereo
                          float stereo_width,
                          // volume
                          float volume,
                          // de-esser
                          float dess_enabled, float dess_thresh, float dess_ratio,
                          // transient
                          float trans_enabled, float trans_atk, float trans_sus,
                          // ceiling
                          float ceiling_db) {
    noise_gate(buf, n_frames, gate_db);
    filterbank_process(c->fb, buf, n_frames);
    if (dess_enabled > 0.5f)
        deesser_process(c->dess, buf, n_frames, dess_thresh, dess_ratio);
    if (trans_enabled > 0.5f)
        transient_process(c->trans, buf, n_frames, trans_atk, trans_sus);
    compressor_process(c->comp, buf, n_frames, comp_lo, comp_mid, comp_hi);
    exciter_process(c->exc, buf, n_frames, exciter_amt);
    stereo_widen(buf, n_frames, stereo_width);
    // volume
    for (int i = 0; i < n_frames * 2; ++i)
        buf[i] = std::min(1.2f, std::max(-1.2f, buf[i] * volume));
    true_peak_limit(buf, n_frames, ceiling_db);
}