"""Original score + sound design for the launch film (ad3.html): 30 s, 120 BPM, E major.

Reads the same beat map as the animation (cues3.json), so every hit lands on the frame it belongs to:
kick/snare on the key words, hats on the drawing lines and particles, bass swells under the camera's
depth moves, big accents on the logo (10.0 s) and the end card (26.0 s).

    python3 soundtrack3.py out/soundtrack3.wav
"""
import json
import os
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
CUE = json.load(open(os.path.join(HERE, 'cues3.json')))
SR = 48000
DUR = float(CUE['dur'])
N = int(SR * DUR)
BEAT = 60 / CUE['bpm']
BAR = BEAT * 4
S16 = BEAT / 4
rs = np.random.default_rng(3030)


def buf():
    return np.zeros((2, N))


DRUMS, BASS, PAD, ARP, KEYS, SFX, HIT = (buf() for _ in range(7))
SEND = buf()


def tt(n):
    return np.arange(n) / SR


def pan2(x, p):
    a = (p + 1) * np.pi / 4
    return np.vstack([x * np.cos(a), x * np.sin(a)])


def add(bus, x, t0, g=1.0, p=0.0, send=0.0):
    if x.ndim == 1:
        x = pan2(x, p)
    i0 = int(round(t0 * SR))
    if i0 < 0:
        x, i0 = x[:, -i0:], 0
    n = min(x.shape[1], N - i0)
    if n <= 0:
        return
    bus[:, i0:i0 + n] += x[:, :n] * g
    if send:
        SEND[:, i0:i0 + n] += x[:, :n] * g * send


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


# ---------------------------------------------------------------- instruments
def kick(v=1.0):
    n = int(.45 * SR)
    t = tt(n)
    f = 48 + 120 * np.exp(-t / .026)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .22)
    click = signal.sosfilt(bp(2000, 7000), rs.standard_normal(n)) * np.exp(-t / .0018) * .3
    return np.tanh((body + click) * 1.6) * np.minimum(1, t / .0008) * v


def clap(v=1.0):
    n = int(.4 * SR)
    t = tt(n)
    nz = signal.sosfilt(bp(1000, 4200), rs.standard_normal(n))
    e = sum((t >= d) * np.exp(-np.clip(t - d, 0, None) / .0045) for d in (0, .008, .017))
    e = e + (t >= .022) * np.exp(-np.clip(t - .022, 0, None) / .09) * .5
    return nz * e * v * np.minimum(1, t / .0006)


def snare(v=1.0):
    n = int(.3 * SR)
    t = tt(n)
    nz = signal.sosfilt(bp(1800, 8000), rs.standard_normal(n)) * np.exp(-t / .07)
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t / .05) * .5
    return (nz + tone) * v * np.minimum(1, t / .0006)


def hat(v=.1, open_=False):
    n = int((.35 if open_ else .07) * SR)
    t = tt(n)
    return signal.sosfilt(hp(8000, 4), rs.standard_normal(n)) * np.exp(-t / (.12 if open_ else .018)) * v


def crash(v=.5, dur=2.6):
    n = int(dur * SR)
    t = tt(n)
    nz = signal.sosfilt(hp(4500), rs.standard_normal(n))
    ring = sum(np.sin(2 * np.pi * f * t + rs.random() * 6) for f in (3301, 4423, 5711, 7013)) * .06
    return (nz * .6 + ring) * np.exp(-t / (dur / 3.2)) * np.minimum(1, t / .002) * v


def pluck(f, v=.2, dur=.6):
    n = int(dur * SR)
    t = tt(n)
    I = 1.8 * np.exp(-t / .06)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * 2 * f * t)) + .25 * np.sin(2 * np.pi * 4.01 * f * t) * np.exp(-t / .05)
    return y * np.exp(-t / .2) * np.minimum(1, t / .0015) * v


def keys(f, v=.4, dur=2.4):
    n = int(dur * SR)
    t = tt(n)
    y = np.zeros(n)
    for k in range(1, 10):
        fk = k * f * np.sqrt(1 + .0003 * k * k)
        if fk > 14000:
            break
        y += np.sin(2 * np.pi * fk * t + .3 * k) / k ** 1.6 * np.exp(-t * (.7 + .5 * k))
    return y * np.minimum(1, t / .004) * v


def bell(f, v=.3, dur=2.5, ratio=3.0, index=1.2):
    n = int(dur * SR)
    t = tt(n)
    I = index * np.exp(-t / .35)
    return np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t)) * np.exp(-t / (dur / 3)) * np.minimum(1, t / .003) * v


def blip(f, v=.3, dec=.03):
    n = int(dec * 8 * SR)
    t = tt(n)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / dec) * np.minimum(1, t / .0012) * v


def tick(v=.2, f=4500):
    n = int(.03 * SR)
    t = tt(n)
    return (signal.sosfilt(bp(f * .6, f * 1.6), rs.standard_normal(n)) * np.exp(-t / .003) + np.sin(2 * np.pi * f * t) * np.exp(-t / .005) * .3) * v


def sweep_tone(f0, f1, dur, v=.1, curve=1.0):
    n = int(dur * SR)
    t = tt(n)
    u = t / dur
    f = f0 * (f1 / f0) ** (u ** curve)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u) ** 1.2 * v


def noise_sweep(dur, f0, f1, peak=.6, q=1.5, amp=1.0, curve=1.0):
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


def whoosh(t0, dur=.6, up=True, amp=.25, pf=-.4, pt=.4, peak=.6, lo=300, hi=4200):
    y = noise_sweep(dur, lo if up else hi, hi if up else lo, peak, 1.4, amp)
    p = np.linspace(pf, pt, len(y))
    a = (p + 1) * np.pi / 4
    add(SFX, np.vstack([y * np.cos(a), y * np.sin(a)]), t0, send=.3)


def impact(t0, v=1.0):
    n = int(2.2 * SR)
    t = tt(n)
    f = 36 + 50 * np.exp(-t / .06)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .6) * np.minimum(1, t / .004)
    thud = signal.sosfilt(lp(1600), rs.standard_normal(n)) * np.exp(-t / .07) * .6 * np.minimum(1, t / .004)
    add(HIT, np.tanh((boom + thud) * 1.3) * v, t0, send=.3)
    add(HIT, crash(.42 * v, 2.6), t0, send=.4)


def chime(t0, notes, v=.1, gap=.06, p=0.0, send=.5):
    for k, m in enumerate(notes):
        add(SFX, bell(mtof(m), v, 1.8, 2.0, .8), t0 + k * gap, p=p, send=send)


# ---------------------------------------------------------------- harmony (E major)
CH = {'Emaj9': [52, 56, 59, 63, 66], 'Amaj9': [57, 61, 64, 68, 71], 'C#m9': [49, 56, 59, 63, 64],
      'B6sus': [59, 64, 66, 68, 71], 'Asus2': [57, 59, 64, 69, 71]}
ROOT = {'Emaj9': 28, 'Amaj9': 33, 'C#m9': 37, 'B6sus': 35, 'Asus2': 33}
PROG = ['Emaj9', 'Amaj9', 'C#m9', 'Asus2', 'Amaj9', 'Emaj9', 'C#m9', 'Amaj9', 'B6sus', 'Emaj9', 'C#m9', 'Amaj9', 'B6sus', 'Emaj9', 'Emaj9']
ARPN = {'Emaj9': [76, 80, 83, 87, 88, 87, 83, 80], 'Amaj9': [76, 81, 85, 88, 92, 88, 85, 81],
        'C#m9': [73, 76, 80, 83, 87, 83, 80, 76], 'B6sus': [76, 78, 80, 83, 88, 83, 80, 78], 'Asus2': [71, 76, 81, 83, 88, 83, 81, 76]}
H, P, B, E = CUE['hook'], CUE['problem'], CUE['brand'], CUE['end']
PJ, VA = CUE['projects'], CUE['value']

# pad
for b, ch in enumerate(PROG):
    t0 = b * BAR
    n = int((BAR + 2.0) * SR)
    t = tt(n)
    st = np.zeros((2, n))
    for m in CH[ch]:
        for v, det in enumerate((-.08, 0., .08)):
            st += pan2(saw(mtof(m) * 2 ** (det / 12), n, rs.random()), (v - 1) * .6) * .045
    st = signal.sosfilt(lp(1500 if b < 2 else (3200 if b not in (11,) else 2000)), st, axis=1)
    st = signal.sosfilt(hp(160), st, axis=1)
    env = np.minimum(1, t / (.6 if b in (0, 5, 13) else .25)) * np.where(t < BAR, 1, np.exp(-(t - BAR) / .6))
    lvl = {0: .5, 1: .55, 11: .65, 13: .6, 14: .55}.get(b, .42)
    add(PAD, st * env * lvl, t0, send=.45)

# hook: a pluck on each word, rising
for i, (t0, m) in enumerate(zip(H['words'], (76, 80, 83))):
    add(KEYS, keys(mtof(m), .5, 2.0), t0, p=.2 - .2 * i, send=.5)
    add(KEYS, keys(mtof(m - 12), .3, 2.0), t0, send=.4)
# ticking 16ths through the hook and the anatomy (drawing lines)
for k in range(int(8.0 / S16)):
    t0 = .25 + k * S16
    if t0 > 7.9:
        break
    add(DRUMS, hat(.035 + .03 * (k % 4 == 2) + (.02 if t0 > 4 else 0)), t0, p=.25 if k % 2 else -.25)

# groove plan
def groove(t0):
    if 4.0 <= t0 < 8.0:
        return 'half'
    if 10.0 <= t0 < 22.0 or 26.0 <= t0 < 28.0:
        return 'full'
    return None
for b in range(len(PROG)):
    for s in range(16):
        t0 = b * BAR + s * S16
        g = groove(t0)
        if g == 'half':
            if s in (0, 6):
                add(DRUMS, kick(.8), t0)
            if s == 8:
                add(DRUMS, snare(.32), t0, send=.3)
            if s in (0, 6):
                n = int(.45 * SR); tb = tt(n); f = mtof(ROOT[PROG[b]] + 12)
                add(BASS, np.tanh((np.sin(2 * np.pi * f * tb) + .3 * np.sin(4 * np.pi * f * tb)) * np.exp(-tb / .3) * 1.5) * .55, t0)
        elif g == 'full':
            if s % 4 == 0:
                add(DRUMS, kick(.85 if t0 < 26 else .6), t0)
            if s in (4, 12):
                add(DRUMS, clap(.4), t0, send=.3)
            add(DRUMS, hat(.05 + .04 * (s % 4 == 2)), t0, p=.3 if s % 2 else -.2)
            if s in (2, 6, 10, 14):
                add(DRUMS, hat(.07, True), t0, p=.35)
            if s % 2 == 1 or s in (2, 10):                       # off-beat pulsing bass
                n = int(.18 * SR); tb = tt(n); f = mtof(ROOT[PROG[b]] + 12)
                add(BASS, np.tanh((np.sin(2 * np.pi * f * tb) + .35 * np.sin(4 * np.pi * f * tb)) * np.minimum(1, tb / .004) * np.exp(-tb / .12) * 1.8) * .45, t0)
# arp from the logo onwards (16ths), thinner in the breakdown
for b in range(4, 15):
    pat = ARPN[PROG[b]]
    for s in range(16):
        t0 = b * BAR + s * S16
        if t0 < 8.4 or t0 >= 29.0:
            continue
        v = (.16 if s % 4 == 0 else .1)
        if 22.0 <= t0 < 26.0:
            v *= .55
        if t0 >= 28.0:
            v *= max(.1, 1 - (t0 - 28) / 1.0)
        add(ARP, pluck(mtof(pat[s % 8]), v), t0, p=.35 if s % 2 else -.35, send=.3)
# keys on the downbeats of the main section
for b in range(5, 14):
    for m in CH[PROG[b]][1:4]:
        add(KEYS, keys(mtof(m + 12), .2, 1.8), b * BAR, send=.4)

# ---------------------------------------------------------------- sound design on the cues
# A — the hook
add(SFX, blip(mtof(88), .1, .05), H['dot'] + .01, send=.4)
add(SFX, sweep_tone(900, 3200, .6, .05), .05, p=-.3)                    # the construction lines
add(SFX, sweep_tone(700, 2400, .55, .04), .4, p=.3)
whoosh(1.4, .6, up=True, amp=.12, pf=0, pt=0, peak=.95, lo=500, hi=6000)
whoosh(H['through'][0], .6, up=True, amp=.3, pf=0, pt=0, peak=.92, lo=300, hi=7000)   # through the counter
add(SFX, sweep_tone(220, 1400, .6, .07, 2.2), H['through'][0])
n = int(1.4 * SR); t = tt(n)
add(HIT, np.sin(2 * np.pi * np.cumsum(40 + 26 * np.exp(-t / .05)) / SR) * np.exp(-t / .45) * np.minimum(1, t / .004) * .5, H['question'], send=.2)
chime(H['question'] + .02, [76, 83, 88], v=.06, gap=.05, send=.6)
add(SFX, noise_sweep(.42, 7000, 1800, .35, 2.2, .08), H['morph'])          # ? becomes a cursor
add(SFX, blip(mtof(95), .06, .04), H['morph'] + .38)
whoosh(3.78, .5, up=False, amp=.1, pf=.2, pt=-.3, peak=.5, lo=400, hi=2600)
add(SFX, tick(.3, 5200), H['click'])
add(SFX, sweep_tone(500, 1500, .62, .05), H['outline'][0] + .02, p=.2)     # the window is drawn
# B — anatomy
n = int(.9 * SR); t = tt(n)
add(BASS, np.sin(2 * np.pi * np.cumsum(41 + 14 * (t / .9)) / SR) * np.sin(np.pi * np.clip(t / .9, 0, 1)) * .45, P['explode'][0] - .1)   # depth
for i, t0 in enumerate(P['words']):
    chime(t0, [[83, 88], [85, 90], [88, 95]][i], v=.08, gap=.05, p=-.2 + .2 * i)
add(SFX, blip(mtof(90), .05, .02), 6.3, p=.3)                            # hover
whoosh(6.6, .3, up=True, amp=.06, pf=-.1, pt=.1, peak=.4, lo=800, hi=5000)   # card lift
for k in range(4):
    add(SFX, tick(.07, 3000 + rs.random() * 400), 7.0 + k * .07, p=-.3)   # typing
add(SFX, tick(.3, 4800), P['press'])
add(SFX, blip(mtof(76), .1, .05), P['press'] + .02, send=.3)
whoosh(P['intoButton'][0], .6, up=True, amp=.28, pf=0, pt=0, peak=.9, lo=300, hi=6000)
add(SFX, noise_sweep(.5, 600, 9000, .97, 2, .1, 1.5), 7.5, send=.3)
# C — the brand
add(DRUMS, kick(.9), B['morph'][0])
add(SFX, sweep_tone(300, 900, .8, .06), B['morph'][0], p=-.2)
for k in range(5):
    add(SFX, sweep_tone(900 + 200 * k, 2600 + 300 * k, .7, .025), B['draw'][0] + k * .12, p=-.5 + .25 * k)
for k in range(16):
    add(SFX, tick(.05 + .03 * (k % 4 == 0), 6000), 8.8 + k * S16 * 1.5, p=(k % 3 - 1) * .5)
add(SFX, noise_sweep(1.2, 400, 9000, .98, 2.2, .14, 1.5), 8.8, send=.3)
impact(B['accent'], 1.0)
chime(B['accent'], [76, 80, 83, 88, 92], v=.07, gap=.045, send=.65)
add(SFX, noise_sweep(1.0, 3500, 14000, .5, 2.4, .05), B['accent'] + .1, send=.5)
whoosh(B['out'][0], .95, up=False, amp=.2, pf=.5, pt=-.5, peak=.45, lo=250, hi=3200)
# D — the work: one transition sound per cut, an accent per service line
whoosh(13.6, .45, up=True, amp=.24, pf=0, pt=0, peak=.92, lo=400, hi=7000)
add(SFX, blip(mtof(88), .06, .04), 14.0)
whoosh(14.0, .6, up=False, amp=.14, pf=0, pt=0, peak=.15, lo=300, hi=5000)
add(SFX, noise_sweep(.5, 1500, 6000, .5, 1.6, .1), 15.3, p=.1)           # card flip
add(SFX, tick(.15, 3800), 15.55)
whoosh(16.6, .45, up=True, amp=.18, pf=-.5, pt=.6, peak=.9)             # edge slide
add(SFX, blip(mtof(83), .06, .05), 17.0, p=.4)
whoosh(18.2, .6, up=False, amp=.12, pf=.0, pt=.0, peak=.5, lo=600, hi=4000)   # scroll
for k in range(6):
    add(SFX, tick(.04, 3600), 18.25 + k * .07)
add(SFX, noise_sweep(.5, 400, 8000, .9, 2, .14, 1.3), 19.55, send=.4)    # the sky opens
add(HIT, kick(.6), 20.0)
chime(20.0, [80, 88], v=.07, gap=.05, send=.5)
for t0 in PJ['phrases']:
    add(DRUMS, snare(.22), t0, send=.35)
    add(SFX, blip(mtof(95), .03, .03), t0 + .01, send=.3)
# E — one system
n = int(1.0 * SR); t = tt(n)
add(BASS, np.sin(2 * np.pi * np.cumsum(55 - 18 * (t / 1.0)) / SR) * np.sin(np.pi * np.clip(t / 1.0, 0, 1)) * .55, VA['pullout'][0])   # dolly out
whoosh(VA['pullout'][0], .95, up=False, amp=.16, pf=0, pt=0, peak=.35)
for k in range(6):
    add(SFX, sweep_tone(1200 + 140 * k, 2400 + 200 * k, .5, .02), VA['lines'][0] + k * .05, p=-.6 + .24 * k)
add(DRUMS, snare(.5), VA['text1'], send=.45)
add(DRUMS, kick(.8), VA['text1'])
chime(VA['text2'], [80, 85, 88], v=.08, gap=.05, send=.6)
add(DRUMS, kick(.7), VA['text2'])
add(SFX, noise_sweep(1.4, 300, 9000, .985, 2.2, .2, 1.6), VA['riser'][0], send=.35)
add(SFX, sweep_tone(200, 1600, 1.4, .06, 2.4), VA['riser'][0])
roll = [25.0 + i * .25 for i in range(2)] + [25.5 + i * .125 for i in range(2)] + [25.75 + i * .0625 for i in range(4)]
for i, t0 in enumerate(roll):
    add(DRUMS, snare(.08 + .22 * i / len(roll)), t0, send=.3)
# F — the call
impact(E['accent'], .95)
chime(E['accent'] + .05, [64, 71, 76, 80, 83, 88], v=.06, gap=.05, send=.7)
add(SFX, noise_sweep(1.1, 3500, 14000, .5, 2.4, .05), 26.35, send=.5)
add(SFX, blip(mtof(88), .07, .05), E['button'], send=.3)
add(SFX, blip(mtof(95), .05, .07), E['button'] + .06, send=.3)
for k, m in enumerate((64, 71, 76, 80)):
    add(KEYS, keys(mtof(m), .32, 3.6), 28.0 + k * .04, send=.6)

# ---------------------------------------------------------------- mix
duck = np.ones(N)
for b in range(len(PROG)):
    for s in range(0, 16, 4):
        t0 = b * BAR + s * S16
        if groove(t0) == 'full':
            i = int(t0 * SR)
            k = np.arange(min(int(.25 * SR), N - i)) / SR
            duck[i:i + len(k)] = np.minimum(duck[i:i + len(k)], 1 - .4 * np.exp(-k / .09))
MUSIC = (PAD * .85 + ARP * 1.1 + KEYS * .55) * duck
irn = int(2.6 * SR)
ti = tt(irn)
ir = np.vstack([signal.sosfilt(lp(7000), rs.standard_normal(irn)) for _ in range(2)]) * np.exp(-ti / .55)
ir[:, :int(.018 * SR)] = 0
ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
wet = np.vstack([signal.fftconvolve(SEND[c], ir[c])[:N] for c in range(2)]) * .5
mix = MUSIC + DRUMS * .6 + BASS * .6 + SFX * .95 + HIT * .62 + wet * .9
mix = signal.sosfilt(hp(30), mix, axis=1)
mix = mix + .4 * signal.sosfilt(hp(6500), mix, axis=1)
mix = mix - .15 * signal.sosfilt(bp(250, 600), mix, axis=1)
peak = np.max(np.abs(mix))
mix = np.tanh(mix / peak * 1.3) / np.tanh(1.3)
fade = np.ones(N)
fi = int(29.0 * SR)
fade[fi:] = np.cos(np.linspace(0, np.pi / 2, N - fi)) ** 1.5
fade[:int(.006 * SR)] = np.linspace(0, 1, int(.006 * SR))
mix *= fade * .9
out = sys.argv[1] if len(sys.argv) > 1 else 'soundtrack3.wav'
wavfile.write(out, SR, (mix.T * 32767).astype(np.int16))
print('wrote', out, 'pre-peak', round(float(peak), 3))
