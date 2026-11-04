# B.S.Y DIGITAL — promo films

Motion-graphics ads in Hebrew, each in 16:9 and 9:16, built as frame-accurate GSAP timelines and rendered with headless Chromium.

## Films

- **`ad4.html` — "כל השירותים" (current, 47 s).** Bright blueprint look, 120 BPM, one continuous camera. Hook, the brand reveal, six real projects in match cuts, then one phone that walks through every service: payments on the site, an AI chat that books an appointment, organic Google ranking, paid campaigns that bring in leads, short videos for TikTok / Instagram / Facebook, and an illustrative leads dashboard; all services then gather around the phone. Beat map: `cues4.json`; score: `soundtrack4.py`.
- **`ad3.html` — launch film (30 s).** Same look, 120 BPM, logo built from the contact button. Beat map: `cues3.json`; score: `soundtrack3.py`.
- **`ad2.html` — "אותו לקוח" (45 s).** Bright studio look, 96 BPM. A customer searches for a barber, lands on a dated site and leaves; after the BSY build, the same search ends with a booking on Luma Hair Studio. Score: `soundtrack2.py`.
- **`ad.html` — first version (30 s).** Dark brand stage, 120 BPM. Score: `soundtrack.py`.

Open a film through a local server to watch it live: `ad4.html` (16:9) or `ad4.html?fmt=v` (9:16); `?t=12.5` freezes a frame.

Platform names (Google, TikTok, Instagram, Facebook) appear as plain text inside generic interfaces; no official logos are used. The dashboard is marked "להמחשה" (illustrative).

## Rendering

```sh
python3 soundtrack4.py out/soundtrack4.wav                       # original score, synced to cues4.json
PAGE=ad4.html DUR=47 node render.js frames h <dir> 30 4 4        # 4 sub-frames per frame (motion blur)
TUNE= ./encode.sh <dir> out/soundtrack4.wav out/BSY-Digital-Services-16x9.mp4 30 4 20   # blend, -14 LUFS, H.264
```

- `render.js` — `stills` or `frames`; `PAGE`, `DUR`, and `FROM`/`TO` (output-frame range) to re-render one passage.
- `encode.sh` — blends sub-frames, two-pass loudness normalisation, `AUDIO_ONLY=1` swaps the soundtrack without re-encoding the picture.
- `assets/` — fonts (Heebo incl. a variable face, Frank Ruhl Libre, Archivo, JetBrains Mono), GSAP 3.12.5 and 3.13 (MorphSVG, DrawSVG), logos, UI crops and project screenshots.
- `out/` — light renders and posters (full-quality masters are kept out of git).
