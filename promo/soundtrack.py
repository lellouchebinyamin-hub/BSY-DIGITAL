"""Original score + sound design for the BSY Digital promo (30 s, 120 BPM, A minor).

Everything is synthesized here, so the track is free of third-party licences.
Cue times mirror the GSAP timeline in ad.html.

    python3 soundtrack.py out.wav
"""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 30.0
N = int(SR * DUR)
BPM = 120
BEAT = 60 / BPM
BAR = BEAT * 4
rs = np.random.default_rng(2026)


def buf():
    return np.zeros((2, N))


DRUMS, BASS, PAD, ARP, SFX, HIT = (buf() for _ in range(6))
SEND = buf()   # reverb send


def tt(n):
    return np.arange(n) / SR


def pan2(x, pan):
    a = (pan + 1) * np.pi / 4
    return np.vstack([x * np.cos(a), x * np.sin(a)])


def add(bus, x, t0, gain=1.0, pan=0.0, send=0.0):
    if x.ndim == 1:
        x = pan2(x, pan)
    i0 = int(round(t0 * SR))
    if i0 < 0:
        x = x[:, -i0:]
        i0 = 0
    n = min(x.shape[1], N - i0)
    if n <= 0:
        return
    bus[:, i0:i0 + n] += x[:, :n] * gain
    if send:
        SEND[:, i0:i0 + n] += x[:, :n] * gain * send


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def sos_lp(fc, order=2):
    return signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')


def sos_hp(fc, order=2):
    return signal.butter(order, fc, 'high', fs=SR, output='sos')


def sos_bp(lo, hi, order=2):
    return signal.butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos')


def tv_filter(x, design, block=256):
    """Time-varying filter: design(t_seconds) -> sos, applied block by block with carried state."""
    y = np.zeros_like(x)
    zi = None
    for i in range(0, len(x), block):
        sos = design(i / SR)
        if zi is None or zi.shape[0] != sos.shape[0]:
            zi = np.zeros((sos.shape[0], 2))
        y[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
    return y


# ---------------------------------------------------------------- instruments
_tables = {}


def saw_table(k):
    if k not in _tables:
        ph = np.arange(4096) / 4096
        tab = np.zeros(4096)
        for h in range(1, k + 1):
            tab += np.sin(2 * np.pi * h * ph) / h
        _tables[k] = tab / np.max(np.abs(tab))
    return _tables[k]


def saw(freq, n, phase0=0.0):
    k = int(max(1, min(160, 16000 / freq)))
    tab = saw_table(k)
    ph = (phase0 + freq * np.arange(n) / SR) % 1.0
    idx = ph * 4096
    i0 = idx.astype(int)
    fr = idx - i0
    return tab[i0] * (1 - fr) + tab[(i0 + 1) % 4096] * fr


def env_adsr(n, a, d, s, r, hold):
    t = tt(n)
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = np.clip((t - hold) / max(r, 1e-4), 0, None)
    return e * np.exp(-rel * 5) * np.where(t > hold + r * 2, 0, 1)


def kick(amp=1.0):
    n = int(0.55 * SR)
    t = tt(n)
    f = 44 + 110 * np.exp(-t / 0.032)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.3)
    click = signal.sosfilt(sos_hp(2500), rs.standard_normal(n)) * np.exp(-t / 0.0025) * 0.5
    return np.tanh((body + click) * 1.6) * amp


def clap(amp=1.0):
    n = int(0.45 * SR)
    t = tt(n)
    nz = signal.sosfilt(sos_bp(900, 3200), rs.standard_normal(n))
    e = np.zeros(n)
    for k, d in enumerate([0, 0.009, 0.019]):
        e += (t >= d) * np.exp(-np.clip(t - d, 0, None) / 0.004) * (0.8 + 0.1 * k)
    e += (t >= 0.026) * np.exp(-np.clip(t - 0.026, 0, None) / 0.11) * 0.55
    return nz * e * amp * 1.8


def snare(amp=1.0):
    n = int(0.3 * SR)
    t = tt(n)
    nz = signal.sosfilt(sos_bp(1500, 7000), rs.standard_normal(n)) * np.exp(-t / 0.055)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04) * 0.6
    return (nz + tone) * amp


def hat(open_=False, amp=1.0):
    n = int((0.45 if open_ else 0.09) * SR)
    t = tt(n)
    nz = signal.sosfilt(sos_hp(7500, 4), rs.standard_normal(n))
    return nz * np.exp(-t / (0.16 if open_ else 0.022)) * amp


def crash(amp=1.0, dur=3.0):
    n = int(dur * SR)
    t = tt(n)
    nz = signal.sosfilt(sos_hp(3800, 2), rs.standard_normal(n))
    ring = sum(np.sin(2 * np.pi * f * t + rs.random() * 6) for f in (3127, 4211, 5323, 6871, 8219)) * 0.08
    e = np.exp(-t / (dur / 3.2)) * np.minimum(1, t / 0.002)
    return (nz * 0.7 + ring) * e * amp


def pluck(freq, amp=1.0, dur=0.7):
    n = int(dur * SR)
    t = tt(n)
    y = np.zeros(n)
    for k in range(1, 11):
        if k * freq > 16000:
            break
        y += np.sin(2 * np.pi * k * freq * t + k * 0.3) / k ** 1.25 * np.exp(-t * (2.4 + 2.2 * k))
    return y * np.minimum(1, t / 0.002) * amp


def bell(freq, amp=1.0, dur=2.5, ratio=3.5, index=2.4):
    n = int(dur * SR)
    t = tt(n)
    I = index * np.exp(-t / 0.35)
    y = np.sin(2 * np.pi * freq * t + I * np.sin(2 * np.pi * freq * ratio * t))
    return y * np.exp(-t / (dur / 3)) * np.minimum(1, t / 0.003) * amp


def blip(freq, amp=1.0, dec=0.03):
    n = int(dec * 8 * SR)
    t = tt(n)
    return np.sin(2 * np.pi * freq * t) * np.exp(-t / dec) * np.minimum(1, t / 0.0015) * amp


def noise_sweep(dur, f0, f1, shape_peak=0.65, q=1.2, amp=1.0, curve=1.0):
    n = int(dur * SR)
    t = tt(n)
    x = rs.standard_normal(n)

    def design(ts):
        k = min(1.0, ts / dur) ** curve
        fc = f0 * (f1 / f0) ** k
        bw = fc / q
        return sos_bp(max(30, fc - bw / 2), fc + bw / 2, 2)
    y = tv_filter(x, design)
    u = t / dur
    e = np.where(u < shape_peak, (u / shape_peak) ** 2, ((1 - u) / (1 - shape_peak)) ** 1.6)
    return y * e * amp


def whoosh(t0, dur=0.6, up=True, amp=0.5, pan_from=-0.5, pan_to=0.5, peak=0.6):
    f0, f1 = (350, 4200) if up else (4200, 300)
    y = noise_sweep(dur, f0, f1, shape_peak=peak, q=1.6, amp=amp)
    n = len(y)
    p = np.linspace(pan_from, pan_to, n)
    a = (p + 1) * np.pi / 4
    st = np.vstack([y * np.cos(a), y * np.sin(a)])
    add(SFX, st, t0, send=0.25)


def riser(t0, t1, amp=0.5, f0=250, f1=7000):
    dur = t1 - t0
    y = noise_sweep(dur, f0, f1, shape_peak=0.995, q=2.5, amp=amp, curve=1.6)
    n = len(y)
    t = tt(n)
    f = 180 * (4.5 ** ((t / dur) ** 1.4))
    tone = (np.sin(2 * np.pi * np.cumsum(f) / SR) + np.sin(2 * np.pi * np.cumsum(f * 1.008) / SR)) * 0.18 * (t / dur) ** 2
    add(SFX, y + tone, t0, send=0.35)


def impact(t0, amp=1.0):
    n = int(2.6 * SR)
    t = tt(n)
    f = 34 + 52 * np.exp(-t / 0.07)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.85)
    thud = signal.sosfilt(sos_lp(1800), rs.standard_normal(n)) * np.exp(-t / 0.09) * 0.8
    ramp = np.minimum(1, t / 0.004)   # soft 4 ms attack keeps AAC from overshooting on the hit
    add(HIT, np.tanh((boom * 1.2 + thud) * 1.4) * ramp * amp, t0, send=0.35)
    add(HIT, crash(0.55 * amp, 3.4), t0, pan=0.0, send=0.4)


# ---------------------------------------------------------------- harmony
CH = {
    'Am9': [57, 60, 64, 67, 71], 'Fmaj9': [53, 57, 60, 64, 67],
    'Cmaj9': [55, 60, 64, 67, 71], 'G69': [55, 59, 62, 64, 69],
}
ROOT = {'Am9': 33, 'Fmaj9': 29, 'Cmaj9': 36, 'G69': 31}
PROG = ['Fmaj9', 'G69', 'Am9', 'Fmaj9', 'Cmaj9', 'G69', 'Am9', 'Fmaj9', 'Cmaj9', 'G69', 'Am9', 'Fmaj9', 'G69', 'Am9', 'Am9']
ARPN = {
    'Am9': [69, 72, 76, 79, 83, 79, 76, 72], 'Fmaj9': [65, 69, 72, 76, 79, 76, 72, 69],
    'Cmaj9': [67, 72, 76, 79, 83, 79, 76, 72], 'G69': [67, 71, 74, 76, 81, 76, 74, 71],
}

# pad, one chord per bar, filter opens through the intro
for b, ch in enumerate(PROG):
    t0 = b * BAR
    n = int((BAR + 1.6) * SR)
    t = tt(n)
    tone = np.zeros((2, n))
    for m in CH[ch]:
        for v, det in enumerate((-0.11, -0.05, 0.0, 0.05, 0.11)):
            f = mtof(m) * 2 ** (det / 12)
            pan = (v - 2) * 0.35
            tone += pan2(saw(f, n, rs.random()), pan) * 0.06
    hold = BAR - 0.02 if b < len(PROG) - 1 else 2.0
    env = env_adsr(n, 0.25 if b > 1 else 0.9, 0.8, 0.85, 0.35, hold)
    if b < 2:
        fc0, fc1 = (380, 900) if b == 0 else (900, 2600)
        design = lambda ts, a=fc0, c=fc1: sos_lp(a * (c / a) ** min(1, ts / BAR))
        y = np.vstack([tv_filter(tone[0], design), tv_filter(tone[1], design)])
    else:
        fc = 3400 if b not in (12,) else 2400
        if b >= 13:
            fc = 4200
        y = signal.sosfilt(sos_lp(fc), tone, axis=1)
    lvl = 0.55 if b < 2 else 0.42
    add(PAD, y * env * lvl, t0, send=0.5)

# sub bass on the off-beats (deep-house pocket), 4 s to 23.5 s + final note
GROOVE = [(4.0, 19.0), (21.0, 23.5)]
for b, ch in enumerate(PROG):
    for s in range(16):
        t0 = b * BAR + s * BEAT / 4
        if not any(a <= t0 < c for a, c in GROOVE):
            continue
        if s in (2, 6, 10, 14) or (s == 7 and b % 2):
            m = ROOT[ch] + (12 if s == 7 else 0)
            n = int(0.22 * SR)
            t = tt(n)
            f = mtof(m)
            y = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)) * np.minimum(1, t / 0.004) * np.exp(-t / 0.16)
            add(BASS, np.tanh(y * 2.0) * 0.6, t0)
# long sub notes for the stabs, the intro drop and the outro
for t0, m, d in [(4.0, 33, 1.0), (19.0, 31, 0.45), (19.5, 31, 0.45), (20.0, 33, 0.45), (20.5, 33, 0.45), (26.0, 33, 2.4)]:
    n = int(d * SR)
    t = tt(n)
    f = mtof(m)
    y = np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.005) * np.exp(-t / (d * 0.55))
    add(BASS, y * 0.8, t0)

# drums
for b in range(len(PROG)):
    for s in range(16):
        t0 = b * BAR + s * BEAT / 4
        on = any(a <= t0 < c for a, c in GROOVE)
        if on and s % 4 == 0:
            add(DRUMS, kick(0.95), t0)
        if on and s in (4, 12):
            add(DRUMS, clap(0.42), t0, send=0.3)
        if on and s in (2, 6, 10, 14):
            add(DRUMS, hat(True, 0.13), t0, pan=0.25, send=0.1)
        if on and s % 2 == 1:
            add(DRUMS, hat(False, 0.09 + 0.03 * (s % 4 == 3)), t0, pan=-0.3)
        # breakdown: light hats only
        if 23.5 <= t0 < 26.0 and s % 2 == 0:
            add(DRUMS, hat(False, 0.06), t0, pan=0.3)
# intro ticks during the zoom-out (data feel)
for k in range(12):
    add(DRUMS, hat(False, 0.05 + 0.03 * k / 12), 0.6 + k * BEAT / 4, pan=-0.4 + 0.07 * k)
# snare roll into the drop
roll = [3.0 + i * 0.25 for i in range(2)] + [3.5 + i * 0.125 for i in range(2)] + [3.75 + i * 0.0625 for i in range(4)]
for i, t0 in enumerate(roll):
    add(DRUMS, snare(0.12 + 0.3 * i / len(roll)), t0, send=0.3)
# quick-fire stabs (19 - 20.5): kick + clap + chord stab
for i, t0 in enumerate((19.0, 19.5, 20.0, 20.5)):
    add(DRUMS, kick(1.0), t0)
    add(DRUMS, clap(0.5), t0, send=0.35)
    ch = 'G69' if t0 < 20 else 'Am9'
    n = int(0.5 * SR)
    t = tt(n)
    st = np.zeros((2, n))
    for m in CH[ch]:
        for v, det in enumerate((-0.08, 0.0, 0.08)):
            st += pan2(saw(mtof(m + 12) * 2 ** (det / 12), n, rs.random()), (v - 1) * 0.5) * 0.05
    st = signal.sosfilt(sos_lp(5200), st, axis=1) * np.exp(-t / 0.11)
    add(HIT, st * 0.9, t0, send=0.45)

# arpeggio (pluck) with ping-pong delay, 6 s to 29 s
ARP_ON = [(6.0, 19.0), (21.0, 29.5)]
for b, ch in enumerate(PROG):
    pat = ARPN[ch]
    for s in range(16):
        t0 = b * BAR + s * BEAT / 4
        if not any(a <= t0 < c for a, c in ARP_ON):
            continue
        if t0 >= 26.0 and s % 2:      # outro thins out
            continue
        vel = (0.95 if s % 4 == 0 else 0.6) * (0.75 if 23.5 <= t0 < 26 else 1.0)
        if t0 >= 26.0:
            vel *= max(0.15, 1 - (t0 - 26) / 3.6)
        m = pat[s % len(pat)]
        add(ARP, pluck(mtof(m), 0.16 * vel), t0, pan=0.35 if s % 2 else -0.35, send=0.3)

# ---------------------------------------------------------------- sound design cues
# S1 — tile builds itself
for k in range(12):
    add(SFX, blip(1400 + 90 * k, 0.07, 0.012), 0.05 + 0.028 * k, pan=(k % 3 - 1) * 0.3)
whoosh(0.5, 1.6, up=False, amp=0.35, pan_from=0.0, pan_to=0.0, peak=0.55)    # zoom out
whoosh(0.85, 0.5, up=True, amp=0.16, peak=0.4)                                 # question in
# templates fall
n = int(1.0 * SR)
t = tt(n)
f = 160 * np.exp(-t / 0.35) + 40
add(SFX, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.4) * 0.5, 2.0, send=0.2)
whoosh(1.95, 0.9, up=False, amp=0.35, pan_from=0.3, pan_to=-0.3, peak=0.3)
# the one tile turns gold
for k, m in enumerate((81, 85, 88, 93)):
    add(SFX, bell(mtof(m), 0.07, 2.4, ratio=2.0, index=1.2), 2.12 + k * 0.06, pan=-0.4 + 0.27 * k, send=0.6)
whoosh(2.42, 0.45, up=True, amp=0.14, peak=0.35)
riser(2.3, 4.0, amp=0.42)
whoosh(3.3, 0.7, up=True, amp=0.5, peak=0.92)                                 # dive into the tile
# S2 — drop
impact(4.0, 1.0)
add(SFX, noise_sweep(0.9, 3000, 13000, 0.5, q=2, amp=0.16), 4.45, send=0.4)   # logo shine
add(SFX, bell(mtof(88), 0.05, 2.0, 3.0, 1.4), 4.6, pan=0.3, send=0.6)
whoosh(5.4, 0.7, up=True, amp=0.28, pan_from=-0.4, pan_to=0.6)               # logo to nav
whoosh(7.55, 0.5, up=False, amp=0.22)
# S3 — services
for t0 in (7.95, 9.75, 11.75):
    whoosh(t0, 0.55, up=True, amp=0.3, pan_from=-0.6, pan_to=0.3, peak=0.45)
add(SFX, blip(1760, 0.08, 0.025), 8.4, pan=0.4)                                # badge pop
for k, t0 in enumerate((8.5, 8.88, 9.26)):                                     # new leads
    add(SFX, blip(mtof(83 + k * 2), 0.12, 0.06), t0, pan=-0.4, send=0.3)
    add(SFX, blip(mtof(88 + k * 2), 0.1, 0.09), t0 + 0.07, pan=-0.4, send=0.3)
add(SFX, signal.sosfilt(sos_hp(1500), rs.standard_normal(int(0.012 * SR))) * np.exp(-tt(int(0.012 * SR)) / 0.002) * 0.4, 10.8, pan=-0.2)  # click
add(SFX, blip(620, 0.12, 0.03), 10.82, pan=-0.2)
add(SFX, bell(mtof(88), 0.08, 1.2, 2.0, 1.0), 11.28, pan=0.4, send=0.4)        # into the cart
add(SFX, bell(mtof(95), 0.07, 1.2, 2.0, 1.0), 11.36, pan=0.45, send=0.4)
add(SFX, blip(mtof(86), 0.07, 0.07), 11.35, pan=0.3, send=0.3)                 # 02:14 order
for k in range(14):                                                            # bars grow
    add(SFX, blip(700 * 2 ** (k / 12 * 1.4), 0.035, 0.015), 12.2 + k * 0.035, pan=0.5 - k * 0.07)
add(SFX, blip(mtof(84), 0.1, 0.06), 12.95, pan=-0.5, send=0.3)                 # booked
add(SFX, blip(mtof(91), 0.09, 0.1), 13.02, pan=-0.5, send=0.3)
whoosh(13.68, 0.5, up=False, amp=0.25)
# S4 — work
whoosh(13.85, 1.1, up=True, amp=0.35, pan_from=0.5, pan_to=-0.5, peak=0.25)
add(HIT, kick(0.5), 14.0)
whoosh(15.5, 0.6, up=True, amp=0.32, peak=0.85)
whoosh(15.95, 0.5, up=False, amp=0.22)
for k in range(1, 6):
    t0 = 16 + k * 0.5 - 0.14
    whoosh(t0, 0.32, up=True, amp=0.16, pan_from=0.6 if k % 2 else -0.6, pan_to=-0.2 if k % 2 else 0.2, peak=0.5)
whoosh(18.68, 0.34, up=False, amp=0.2)
# S5 — why
riser(18.2, 19.0, amp=0.16, f0=1200, f1=9000)
whoosh(20.95, 0.5, up=True, amp=0.2)
for k, m in enumerate((81, 84, 88, 93)):                                       # process nodes
    add(SFX, pluck(mtof(m), 0.2, 0.6), 21.6 + k * 0.43, pan=0.5 - k * 0.33, send=0.4)
add(SFX, bell(mtof(93), 0.06, 1.6, 2.0, 1.0), 22.95, pan=-0.5, send=0.5)
whoosh(23.18, 0.45, up=False, amp=0.22)
# S6 — one message
whoosh(23.45, 0.6, up=True, amp=0.26, pan_from=-0.5, pan_to=0.0)
for k in range(13):
    tk = 23.86 + k * 0.055 + rs.random() * 0.012
    n = int(0.02 * SR)
    clk = signal.sosfilt(sos_bp(1800, 6000), rs.standard_normal(n)) * np.exp(-tt(n) / 0.0025) * 0.16
    add(SFX, clk + blip(260 + rs.random() * 60, 0.05, 0.006)[:n], tk, pan=-0.35)
add(SFX, noise_sweep(0.32, 600, 5000, 0.7, q=1.4, amp=0.22), 24.6, pan=-0.3)  # send
add(SFX, blip(mtof(93), 0.05, 0.02), 24.95, pan=-0.3)
add(SFX, blip(mtof(86), 0.12, 0.08), 25.38, pan=0.3, send=0.3)                 # reply
add(SFX, blip(mtof(93), 0.1, 0.12), 25.46, pan=0.3, send=0.3)
riser(24.6, 26.05, amp=0.3, f0=400, f1=9000)
whoosh(25.6, 0.5, up=True, amp=0.35, peak=0.85)
# S7 — end card
impact(26.05, 0.95)
add(SFX, noise_sweep(1.1, 3000, 14000, 0.5, q=2, amp=0.15), 26.35, send=0.45)
for k, m in enumerate((76, 81, 84, 88, 93)):
    add(SFX, bell(mtof(m), 0.05, 3.0, 3.0, 1.2), 26.4 + k * 0.07, pan=-0.5 + 0.25 * k, send=0.6)
add(SFX, blip(mtof(81), 0.1, 0.05), 26.8, send=0.3)                            # CTA pop
add(SFX, blip(mtof(88), 0.07, 0.08), 26.86, send=0.3)

# ---------------------------------------------------------------- mix
# side-chain duck from the kicks
duck = np.ones(N)
t = tt(N)
for b in range(len(PROG)):
    for s in range(0, 16, 4):
        t0 = b * BAR + s * BEAT / 4
        if any(a <= t0 < c for a, c in GROOVE):
            i0 = int(t0 * SR)
            n = int(0.3 * SR)
            k = np.arange(min(n, N - i0)) / SR
            duck[i0:i0 + len(k)] = np.minimum(duck[i0:i0 + len(k)], 1 - 0.55 * np.exp(-k / 0.09))
PAD *= duck
ARP *= 0.4 + 0.6 * duck

# reverb: decorrelated decaying noise, damped
irn = int(2.4 * SR)
ti = tt(irn)
ir = np.vstack([signal.sosfilt(sos_lp(6500), rs.standard_normal(irn)) for _ in range(2)]) * np.exp(-ti / 0.55)
ir[:, :int(0.018 * SR)] = 0
ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
wet = np.vstack([signal.fftconvolve(SEND[c], ir[c])[:N] for c in range(2)]) * 0.55

mix = DRUMS * 0.55 + BASS * 0.6 + PAD * 1.7 + ARP * 1.35 + SFX * 0.95 + HIT * 0.62 + wet * 0.9
mix = signal.sosfilt(sos_hp(28), mix, axis=1)
# gentle glue + soft clip
mix = np.tanh(mix * 1.15) / 1.15
fade = np.ones(N)
fi = int(29.2 * SR)
fade[fi:] = np.cos(np.linspace(0, np.pi / 2, N - fi)) ** 2
fade[:int(0.01 * SR)] = np.linspace(0, 1, int(0.01 * SR))
mix *= fade
mix /= np.max(np.abs(mix)) / 0.89
out = sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav'
wavfile.write(out, SR, (mix.T * 32767).astype(np.int16))
print('wrote', out, mix.shape, 'peak', np.max(np.abs(mix)))
