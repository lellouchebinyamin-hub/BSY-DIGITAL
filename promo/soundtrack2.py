"""Original score + sound design for "Same Customer" (ad2.html): 45 s, 96 BPM, D major.

Felt piano motif, glassy FM plucks, warm pad, half-time groove (soft kick, snap, shaker, sub).
Muffled during the problem scene, a real tape-style rewind into the turn, a gentle bloom on the logo,
and the hook motif returns on the end card. Everything is synthesized here.

    python3 soundtrack2.py out/soundtrack2.wav
"""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DUR = 45.0
N = int(SR * DUR)
BEAT = 60 / 96
BAR = BEAT * 4
S16 = BEAT / 4
rs = np.random.default_rng(96)


def buf():
    return np.zeros((2, N))


PIANO, PAD, ARP, DRUMS, BASS, SFX, HIT = (buf() for _ in range(7))
SEND = buf()


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
        x, i0 = x[:, -i0:], 0
    n = min(x.shape[1], N - i0)
    if n <= 0:
        return
    bus[:, i0:i0 + n] += x[:, :n] * gain
    if send:
        SEND[:, i0:i0 + n] += x[:, :n] * gain * send


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(fc, o=2):
    return signal.butter(o, min(fc, SR * .45), 'low', fs=SR, output='sos')


def hp(fc, o=2):
    return signal.butter(o, fc, 'high', fs=SR, output='sos')


def bp(lo, hi, o=2):
    return signal.butter(o, [lo, min(hi, SR * .45)], 'band', fs=SR, output='sos')


def tv(x, design, block=256):
    y = np.zeros_like(x)
    zi = None
    for i in range(0, x.shape[-1], block):
        sos = design(i / SR)
        if zi is None:
            zi = np.zeros((x.shape[0], sos.shape[0], 2)) if x.ndim == 2 else np.zeros((sos.shape[0], 2))
        if x.ndim == 2:
            for c in range(x.shape[0]):
                y[c, i:i + block], zi[c] = signal.sosfilt(sos, x[c, i:i + block], zi=zi[c])
        else:
            y[i:i + block], zi = signal.sosfilt(sos, x[i:i + block], zi=zi)
    return y


# ---------------------------------------------------------------- instruments
def piano(f, vel=0.6, dur=3.2):
    """Felt piano: stretched partials, brightness from velocity, soft hammer."""
    n = int(dur * SR)
    t = tt(n)
    y = np.zeros(n)
    B = 0.00035
    for k in range(1, 14):
        fk = k * f * np.sqrt(1 + B * k * k)
        if fk > 15000:
            break
        a = (1 / k ** 1.7) * (vel ** (0.4 * (k - 1)))
        y += a * np.sin(2 * np.pi * fk * t + 0.4 * k) * np.exp(-t * (0.55 + 0.42 * k))
    ham = signal.sosfilt(lp(2200), rs.standard_normal(n)) * np.exp(-t / 0.006) * 0.06
    env = np.minimum(1, t / 0.004)
    return (y + ham) * env * vel


def glass(f, vel=0.5, dur=1.4):
    n = int(dur * SR)
    t = tt(n)
    I = 2.1 * np.exp(-t / 0.07)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * 2 * f * t))
    y += 0.3 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.12)
    return y * np.exp(-t / 0.42) * np.minimum(1, t / 0.002) * vel


_tab = {}


def saw(f, n, ph0=0.0):
    k = int(max(1, min(120, 15000 / f)))
    if k not in _tab:
        p = np.arange(4096) / 4096
        tab = sum(np.sin(2 * np.pi * h * p) / h for h in range(1, k + 1))
        _tab[k] = tab / np.max(np.abs(tab))
    tab = _tab[k]
    ph = (ph0 + f * np.arange(n) / SR) % 1.0
    idx = ph * 4096
    i0 = idx.astype(int)
    fr = idx - i0
    return tab[i0] * (1 - fr) + tab[(i0 + 1) % 4096] * fr


def bell(f, vel=0.5, dur=3.0, ratio=3.0, index=1.2):
    n = int(dur * SR)
    t = tt(n)
    I = index * np.exp(-t / 0.4)
    return np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t)) * np.exp(-t / (dur / 3)) * np.minimum(1, t / 0.003) * vel


def blip(f, vel=0.5, dec=0.03):
    n = int(dec * 8 * SR)
    t = tt(n)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / dec) * np.minimum(1, t / 0.0015) * vel


def tick(vel=0.3, f=4200):
    n = int(0.03 * SR)
    t = tt(n)
    return (signal.sosfilt(bp(f * .6, f * 1.6), rs.standard_normal(n)) * np.exp(-t / 0.004) + np.sin(2 * np.pi * f * t) * np.exp(-t / 0.006) * .3) * vel


def kick(vel=0.8):
    n = int(0.5 * SR)
    t = tt(n)
    f = 50 + 70 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    click = signal.sosfilt(bp(1500, 5000), rs.standard_normal(n)) * np.exp(-t / 0.002) * 0.15
    return np.tanh((body + click) * 1.3) * np.minimum(1, t / 0.001) * vel


def snap(vel=0.5):
    n = int(0.35 * SR)
    t = tt(n)
    nz = signal.sosfilt(bp(1400, 5200), rs.standard_normal(n))
    e = np.exp(-t / 0.012) + 0.35 * np.exp(-np.clip(t - .008, 0, None) / 0.05) * (t > .008)
    return nz * e * vel * np.minimum(1, t / 0.0008)


def shaker(vel=0.12):
    n = int(0.09 * SR)
    t = tt(n)
    nz = signal.sosfilt(hp(6000, 4), rs.standard_normal(n))
    return nz * np.sin(np.pi * np.clip(t / 0.05, 0, 1)) ** 2 * np.exp(-t / 0.03) * vel


def noise_sweep(dur, f0, f1, peak=0.6, q=1.5, amp=1.0, curve=1.0):
    n = int(dur * SR)
    t = tt(n)
    x = rs.standard_normal(n)

    def design(ts):
        k = min(1.0, ts / dur) ** curve
        fc = f0 * (f1 / f0) ** k
        bw = fc / q
        return bp(max(30, fc - bw / 2), fc + bw / 2)
    y = tv(x, design)
    u = t / dur
    e = np.where(u < peak, (u / peak) ** 2, ((1 - u) / (1 - peak)) ** 1.5)
    return y * e * amp


def whoosh(t0, dur=0.7, up=True, amp=0.25, pfrom=-0.4, pto=0.4, peak=0.55, lo=300, hi=3600):
    y = noise_sweep(dur, lo if up else hi, hi if up else lo, peak, q=1.4, amp=amp)
    p = np.linspace(pfrom, pto, len(y))
    a = (p + 1) * np.pi / 4
    add(SFX, np.vstack([y * np.cos(a), y * np.sin(a)]), t0, send=0.3)


def chime(t0, notes, vel=0.12, gap=0.07, pan=0.0, send=0.45):
    for k, m in enumerate(notes):
        add(SFX, bell(mtof(m), vel, 1.8, 2.0, 0.8), t0 + k * gap, pan=pan, send=send)


# ---------------------------------------------------------------- harmony
CH = {
    'Dmaj9': [50, 57, 61, 64, 66], 'Gmaj9': [55, 59, 62, 66, 69], 'Bm9': [59, 62, 66, 69, 73],
    'Gm9': [55, 58, 62, 65, 69], 'Em9': [52, 55, 59, 62, 66], 'F#m7': [54, 57, 61, 64, 69],
    'A6sus': [57, 62, 64, 66, 71],
}
ROOT = {'Dmaj9': 38, 'Gmaj9': 31, 'Bm9': 35, 'Gm9': 31, 'Em9': 28, 'F#m7': 30, 'A6sus': 33}
PROG = ['Dmaj9', 'Gmaj9', 'Bm9', 'Gm9', 'Em9', 'Dmaj9', 'Bm9', 'Gmaj9', 'Dmaj9', 'F#m7', 'Gmaj9', 'A6sus',
        'Bm9', 'Gmaj9', 'Gmaj9', 'A6sus', 'Dmaj9', 'Dmaj9']
ARPN = {
    'Dmaj9': [74, 78, 81, 85, 88, 85, 81, 78], 'Gmaj9': [74, 78, 81, 83, 86, 83, 81, 78],
    'Bm9': [73, 74, 78, 81, 85, 81, 78, 74], 'F#m7': [73, 76, 78, 81, 85, 81, 78, 76],
    'A6sus': [74, 76, 78, 81, 83, 81, 78, 76], 'Em9': [71, 74, 78, 79, 83, 79, 78, 74], 'Gm9': [70, 74, 77, 81, 82, 81, 77, 74],
}

# pad
PAD_LVL = {0: .38, 1: .42, 2: .40, 3: .40, 4: .45, 5: .55, 16: .55, 17: .5}
for b, ch in enumerate(PROG):
    t0 = b * BAR
    n = int((BAR + 2.2) * SR)
    t = tt(n)
    st = np.zeros((2, n))
    for m in CH[ch]:
        for v, det in enumerate((-0.07, 0.0, 0.07)):
            st += pan2(saw(mtof(m) * 2 ** (det / 12), n, rs.random()), (v - 1) * 0.55) * 0.05
        st += pan2(np.sin(2 * np.pi * mtof(m) * t), 0) * 0.03
    att = 0.9 if b in (0, 4, 5, 16) else 0.35
    hold = BAR if b < 17 else 1.6
    env = np.minimum(1, t / att) * np.where(t < hold, 1, np.exp(-(t - hold) / 0.7))
    st = signal.sosfilt(lp(2400 if b >= 5 else 1600), st, axis=1)
    add(PAD, st * env * PAD_LVL.get(b, .36), t0, send=0.55)

# felt piano: the hook motif (bars 0-1), sparse chords in the middle, motif returns on the end card
MOTIF = {0: [74, 69, 78, 69, 76, 69, 73, 69], 1: [74, 71, 78, 71, 81, 71, 78, 74]}
for b in (0, 1):
    for k, m in enumerate(MOTIF[b]):
        add(PIANO, piano(mtof(m), 0.5 if k % 2 == 0 else 0.34), b * BAR + k * BEAT / 2, pan=0.15 if k % 2 else -0.1, send=0.5)
for b in (16, 17):
    mot = MOTIF[0] if b == 16 else [74, 69, 78, 69, 76, 73, 74, 74]
    for k, m in enumerate(mot):
        if b == 17 and k >= 6:
            continue
        add(PIANO, piano(mtof(m), (0.5 if k % 2 == 0 else 0.34) * (1 - 0.08 * (b - 16) * k), 3.6), b * BAR + k * BEAT / 2, pan=0.15 if k % 2 else -0.1, send=0.55)
add(PIANO, piano(mtof(62), 0.45, 4.0), 17 * BAR + 3 * BEAT, send=0.6)      # final D
add(PIANO, piano(mtof(50), 0.35, 4.0), 17 * BAR + 3 * BEAT, send=0.6)
for b in (2, 3):                                                            # problem: slow, unsure
    for k, m in enumerate(CH[PROG[b]][1:4]):
        add(PIANO, piano(mtof(m), 0.32, 3.0), b * BAR + k * BEAT * 1.5, pan=-0.2 + 0.2 * k, send=0.5)
add(PIANO, piano(mtof(66), 0.42, 4.5), 4 * BAR + 0.2, send=0.6)             # the turn: one held note
add(PIANO, piano(mtof(71), 0.3, 4.0), 4 * BAR + 1.45, send=0.6)
for b in range(6, 16):                                                      # chord pulses on beat 1
    for m in CH[PROG[b]][1:4]:
        add(PIANO, piano(mtof(m + 12), 0.36, 2.4), b * BAR, pan=0.0, send=0.45)

# glass arp from the craft scene on: eighths, with a dotted-eighth ping-pong delay
for b in range(6, 16):
    pat = ARPN[PROG[b]]
    for k in range(8):
        t0 = b * BAR + k * BEAT / 2
        vel = (0.16 if k % 2 == 0 else 0.11) * (1.15 if b >= 14 else 1.0)
        g = glass(mtof(pat[k]), vel)
        add(ARP, g, t0, pan=-0.25 if k % 2 else 0.25, send=0.35)
        for r, (dt, gain, pan) in enumerate(((0.469, .38, .7), (0.938, .18, -.7))):
            add(ARP, g * gain, t0 + dt, pan=pan, send=0.25)

# groove: half-time, soft
def on(t0):
    return 15.0 <= t0 < 40.0
for b in range(len(PROG)):
    for s in range(16):
        t0 = b * BAR + s * S16
        if not on(t0):
            continue
        intro = t0 < 17.5
        if s in (0, 6) or (s == 10 and b % 2 == 1):
            add(DRUMS, kick(0.75 if s == 0 else 0.55), t0)
        if s == 8 and not intro:
            add(DRUMS, snap(0.42), t0, send=0.35)
        if s % 2 == 1 or not intro:
            add(DRUMS, shaker(0.07 + 0.04 * (s % 4 == 2)), t0 + (0.012 if s % 2 else 0), pan=0.3 if s % 2 else -0.2)
        if s == 14 and b % 2 == 0 and not intro:
            add(DRUMS, tick(0.12, 2600), t0, pan=-0.3, send=0.2)
        if b >= 14 and s in (2, 6, 10, 14):                                 # lift into the result
            add(DRUMS, signal.sosfilt(hp(7000, 4), rs.standard_normal(int(.25 * SR))) * np.exp(-tt(int(.25 * SR)) / .09) * .06, t0, pan=.35)
        if s in (0, 6) or (s == 11 and b % 2 == 0):
            m = ROOT[PROG[b]] + (12 if s == 11 else 0)
            n = int((0.55 if s == 0 else 0.4) * SR)
            t = tt(n)
            f = mtof(m + 12) if m < 30 else mtof(m)
            y = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)) * np.minimum(1, t / 0.006) * np.exp(-t / 0.3)
            add(BASS, np.tanh(y * 1.6) * 0.5, t0)
# problem pulse (heartbeat)
for k in range(8):
    t0 = 5.0 + k * BEAT
    add(DRUMS, kick(0.28 if k % 2 == 0 else 0.18) * 0.8, t0)

# ---------------------------------------------------------------- sound design
whoosh(0.0, 1.4, up=True, amp=0.10, pfrom=0, pto=0, peak=0.75, lo=800, hi=7000)
for k in range(14):                                                         # typing on the phone
    add(SFX, tick(0.10 + rs.random() * .04, 3200 + rs.random() * 900), 0.62 + k * 0.092 + rs.random() * .01, pan=-0.25)
for k in range(3):
    add(SFX, blip(mtof(86 + k * 2), 0.05, 0.02), 2.0 + k * .15, pan=-0.3)
add(SFX, tick(0.3, 5200), 3.75, pan=-0.25)                                  # tap
add(SFX, blip(mtof(81), 0.08, 0.04), 3.76, pan=-0.25, send=.3)
whoosh(3.85, 0.8, up=True, amp=0.14, pfrom=-.3, pto=.0, peak=0.7, lo=500, hi=6000)
whoosh(4.2, 1.3, up=False, amp=0.10, pfrom=-.3, pto=.1, peak=0.45, lo=200, hi=1800)
for k in range(14):                                                         # spinner
    add(SFX, tick(0.035, 2400), 5.0 + k * 0.066)
for k, t0 in enumerate((5.9, 6.1, 6.3, 6.55, 6.8, 7.1)):                    # clunky, dated pops
    n = int(.12 * SR)
    t = tt(n)
    th = np.sin(2 * np.pi * (180 + 37 * k) * t) * np.exp(-t / .03) * .22 + signal.sosfilt(bp(300, 1200), rs.standard_normal(n)) * np.exp(-t / .01) * .06
    add(SFX, th, t0, pan=(k % 3 - 1) * .3)
for k, t0 in enumerate((6.25, 6.875, 7.5)):                                 # the three remarks
    add(SFX, bell(mtof(59 + [0, 1, 0][k]), 0.12, 1.4, 1.0, 0.6), t0, pan=[-.5, .5, -.5][k], send=.35)
whoosh(8.66, 0.45, up=True, amp=0.16, pfrom=-.5, pto=.6, peak=0.45)       # swipe back
add(SFX, tick(0.22, 5000), 9.1, pan=-.2)
# the turn: one held shimmer on the gold line
add(SFX, bell(mtof(90), 0.05, 4.5, 2.0, 0.4), 9.95, send=.7)
# the logo blooms
add(SFX, noise_sweep(0.6, 400, 6000, peak=0.97, q=2, amp=0.10, curve=1.4), 11.9, send=.4)
n = int(2.4 * SR)
t = tt(n)
thump = np.sin(2 * np.pi * np.cumsum(40 + 30 * np.exp(-t / .06)) / SR) * np.exp(-t / .7) * np.minimum(1, t / .004)
add(HIT, thump * .55, 12.5, send=.25)
chime(12.5, [74, 78, 81, 86], vel=0.07, gap=0.06, send=.7)
add(SFX, noise_sweep(1.0, 3500, 14000, 0.5, q=2.4, amp=0.06), 12.9, send=.5)  # shine
whoosh(14.3, 0.9, up=True, amp=0.12, pfrom=-.2, pto=.6, peak=0.6)
# the build
for k in range(12):
    add(SFX, blip(1800 + 120 * k, 0.025, .012), 15.2 + k * .045, pan=-0.6 + 0.1 * k)
for t0 in (15.625, 16.25, 16.875):
    n = int(.08 * SR)
    t = tt(n)
    add(SFX, (np.sin(2 * np.pi * 620 * t) * np.exp(-t / .018) * .14 + signal.sosfilt(bp(1500, 5000), rs.standard_normal(n)) * np.exp(-t / .004) * .07), t0, pan=-.3)
whoosh(17.15, 1.0, up=True, amp=0.12, pfrom=.4, pto=-.4, peak=0.5)
for k, m in enumerate((86, 88, 90, 93)):
    add(SFX, glass(mtof(m), 0.10, 1.2), 17.5 + k * .625, pan=[.3, -.4, -.5, .4][k], send=.4)
whoosh(18.7, 0.7, up=True, amp=0.08, pfrom=-.6, pto=-.2, peak=0.5)
# the four solutions
for t0 in (19.45, 23.3, 27.05, 30.8, 34.3):
    whoosh(t0, 0.95, up=t0 < 34, amp=0.14, pfrom=.0, pto=.0, peak=0.5, lo=250, hi=2800)
for k in range(7):
    add(SFX, tick(0.06, 2800 + rs.random() * 500), 24.9 + k * .055, pan=.2)
for k in range(9):
    add(SFX, tick(0.06, 2800 + rs.random() * 500), 25.3 + k * .03, pan=.2)
add(SFX, tick(0.18, 3600), 25.6, pan=.2)
chime(25.68, [81, 88], vel=0.08, gap=.08, pan=.2)
chime(26.25, [86, 93], vel=0.10, gap=.09, pan=-.4)
add(SFX, tick(0.18, 3600), 28.75, pan=.2)
add(SFX, blip(mtof(76), 0.10, .05), 28.8, pan=.2)
add(SFX, tick(0.15, 3600), 29.36, pan=.2)
chime(29.4, [81, 85, 88], vel=0.07, gap=.06, pan=.1)
chime(30.0, [88, 93], vel=0.10, gap=.09, pan=-.4)
for k in range(16):
    add(SFX, blip(900 * 2 ** (k / 12 * 1.2), 0.018, .012), 31.35 + k * .04, pan=-.5 + k * .06)
add(SFX, glass(mtof(86), 0.12, 1.0), 32.5, pan=-.2, send=.4)
chime(33.75, [86, 93], vel=0.10, gap=.09, pan=-.4)
# same customer
whoosh(34.95, 0.8, up=False, amp=0.10, pfrom=-.3, pto=-.3, peak=0.4)
add(SFX, tick(0.28, 5200), 35.6, pan=-.25)
whoosh(35.85, 0.6, up=True, amp=0.10, pfrom=-.3, pto=-.3, peak=0.45, lo=600, hi=5000)
add(SFX, tick(0.28, 5200), 37.5, pan=-.25)
whoosh(37.55, 0.5, up=True, amp=0.09, pfrom=-.3, pto=-.3, peak=0.4, lo=300, hi=2500)
chime(38.1, [81, 85, 88, 93], vel=0.10, gap=.07, pan=-.2, send=.55)
add(SFX, noise_sweep(1.0, 600, 9000, peak=0.98, q=2, amp=0.12, curve=1.3), 39.0, send=.5)   # bloom
# end card
add(HIT, thump * .45, 40.0, send=.3)
chime(40.0, [74, 78, 81, 86, 90], vel=0.06, gap=.05, send=.75)
add(SFX, noise_sweep(1.2, 3500, 14000, 0.5, q=2.4, amp=0.05), 43.1, send=.5)

# ---------------------------------------------------------------- tape rewind (9.4 s)
MUS = PIANO + PAD + ARP
i0, i1 = int(8.35 * SR), int(9.40 * SR)
seg = MUS[:, i0:i1][:, ::-1]
L = int(0.58 * SR)
src = np.linspace(0, seg.shape[1] - 1, L) ** 1.0
u = np.linspace(0, 1, L)
pos = (u ** 0.7) * (seg.shape[1] - 1)                          # accelerating rewind
rew = np.vstack([np.interp(pos, np.arange(seg.shape[1]), seg[c]) for c in range(2)])
rew *= np.sin(np.pi * np.clip(u / 0.2, 0, 1) / 2) * (1 - u) ** 0.6
r0 = int(9.42 * SR)
for B in (PIANO, PAD, ARP):                                    # the music stops while it rewinds
    fade = np.ones(N)
    fade[r0 - int(.04 * SR):r0] = np.linspace(1, 0, int(.04 * SR))
    fade[r0:int(10.0 * SR)] = 0
    # pad of the turn comes back gently
    B *= fade
add(SFX, rew * 1.1, 9.42, send=.2)
add(SFX, noise_sweep(0.58, 6000, 300, peak=0.2, q=1.2, amp=0.08), 9.42)

# ---------------------------------------------------------------- mix
# muffle the music through the problem, open it again for the turn
def cutoff(ts):
    if ts < 4.4:
        return 12000
    if ts < 5.0:
        return 12000 * (900 / 12000) ** ((ts - 4.4) / .6)
    if ts < 10.0:
        return 900
    if ts < 12.5:
        return 900 * (14000 / 900) ** ((ts - 10.0) / 2.5)
    return 14000
PAD = signal.sosfilt(hp(190), PAD, axis=1)
MUSIC = tv(PIANO * 0.5 + PAD * 0.68 + ARP * 1.35, lambda ts: lp(cutoff(ts)), block=512)
duck = np.ones(N)
for b in range(len(PROG)):
    for s in (0, 6):
        t0 = b * BAR + s * S16
        if on(t0):
            i = int(t0 * SR)
            k = np.arange(min(int(.25 * SR), N - i)) / SR
            duck[i:i + len(k)] = np.minimum(duck[i:i + len(k)], 1 - 0.3 * np.exp(-k / .1))
MUSIC *= duck

irn = int(3.0 * SR)
ti = tt(irn)
ir = np.vstack([signal.sosfilt(lp(7000), rs.standard_normal(irn)) for _ in range(2)]) * np.exp(-ti / 0.75)
ir[:, :int(.02 * SR)] = 0
ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
send = tv(SEND, lambda ts: lp(cutoff(ts) * 1.5), block=512)
wet = np.vstack([signal.fftconvolve(send[c], ir[c])[:N] for c in range(2)]) * 0.5

mix = MUSIC * 1.0 + DRUMS * 0.55 + BASS * 0.55 + SFX * 0.95 + HIT * 0.7 + wet * 0.9
mix = signal.sosfilt(hp(30), mix, axis=1)
mix = mix + 0.45 * signal.sosfilt(hp(6000), mix, axis=1)          # air shelf
mix = mix - 0.18 * signal.sosfilt(bp(280, 650), mix, axis=1)      # clear the low mids
peak = np.max(np.abs(mix))
mix = np.tanh(mix / peak * 1.25) / np.tanh(1.25)
fade = np.ones(N)
fi = int(43.6 * SR)
fade[fi:] = np.cos(np.linspace(0, np.pi / 2, N - fi)) ** 1.5
fade[:int(.008 * SR)] = np.linspace(0, 1, int(.008 * SR))
mix *= fade * 0.9
out = sys.argv[1] if len(sys.argv) > 1 else 'soundtrack2.wav'
wavfile.write(out, SR, (mix.T * 32767).astype(np.int16))
print('wrote', out, 'pre-peak', round(float(peak), 3))
