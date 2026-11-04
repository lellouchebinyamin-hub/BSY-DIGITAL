# B.S.Y DIGITAL — promo films

Motion-graphics ads in Hebrew, each in 16:9 and 9:16, built as frame-accurate GSAP timelines and rendered with headless Chromium.

## Films

- **`ad2.html` — "אותו לקוח" (current, 45 s).** Bright studio look, 96 BPM. A customer searches for a barber, lands on a dated site and leaves; after the BSY build, the same search ends with a booking on Luma Hair Studio. Score: `soundtrack2.py`.
- **`ad.html` — first version (30 s).** Dark brand stage, 120 BPM. Score: `soundtrack.py`.

Open a film through a local server to watch it live: `ad2.html` (16:9) or `ad2.html?fmt=v` (9:16); `?t=12.5` freezes a frame.

## Rendering

```sh
python3 soundtrack2.py out/soundtrack2.wav                       # original score, synced to the timeline
PAGE=ad2.html DUR=45 node render.js frames h <dir> 30 4 6        # 6 sub-frames per frame (motion blur)
./encode.sh <dir> out/soundtrack2.wav out/BSY-Digital-Promo2-16x9.mp4   # blend, -14 LUFS, H.264
```

- `render.js` — `stills` or `frames`; `PAGE`, `DUR`, and `FROM`/`TO` (output-frame range) to re-render one passage.
- `encode.sh` — blends sub-frames, two-pass loudness normalisation, `AUDIO_ONLY=1` swaps the soundtrack without re-encoding the picture.
- `assets/` — fonts (Heebo, Frank Ruhl Libre, Archivo, JetBrains Mono), GSAP 3.12.5, logos and project screenshots.
- `out/` — light renders and posters (full-quality masters are kept out of git).
