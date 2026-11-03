# B.S.Y DIGITAL — promo film

30-second motion-graphics ad (Hebrew), 16:9 and 9:16, built as a frame-accurate GSAP timeline and rendered with headless Chromium.

- `ad.html` — the film. Open `ad.html` (16:9) or `ad.html?fmt=v` (9:16) through a local server to watch it live; `?t=12.5` freezes a frame.
- `soundtrack.py` — original score and sound design (numpy/scipy), synced to the timeline. `python3 soundtrack.py out/soundtrack.wav`
- `render.js` — renders frames: `node render.js frames h <dir> 30 4 6` (6 sub-frames per frame for motion blur).
- `encode.sh` — blends sub-frames, normalises audio to -14 LUFS, writes the MP4: `./encode.sh <dir> out/soundtrack.wav out/BSY-Digital-Promo-16x9.mp4`
- `assets/` — fonts (Heebo, Frank Ruhl Libre, Archivo, JetBrains Mono), GSAP 3.12.5, logo and project screenshots.
