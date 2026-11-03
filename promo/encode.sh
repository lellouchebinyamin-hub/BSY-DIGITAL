#!/usr/bin/env bash
# Blend motion-blur sub-frames into the final film and mux the soundtrack.
#   ./encode.sh <framesDir> <soundtrack.wav> <out.mp4> [fps=30] [samples=6] [crf=18]
#   AUDIO_ONLY=1 ./encode.sh - <soundtrack.wav> <existing.mp4>   (swap the soundtrack, keep the picture)
set -euo pipefail
F=$1; A=$2; O=$3; FPS=${4:-30}; S=${5:-6}; CRF=${6:-18}
TMP=$(mktemp -d)

# two-pass loudness normalisation to -14 LUFS / -2 dBTP (social platforms)
ffmpeg -hide_banner -nostats -i "$A" -af loudnorm=I=-14:TP=-2:LRA=11:print_format=json -f null - 2>"$TMP/ln.txt"
read -r MI MTP MLRA MTH OFF < <(python3 - "$TMP/ln.txt" <<'EOF'
import json, sys, re
txt = open(sys.argv[1]).read()
j = json.loads(re.search(r'\{[^{}]*"input_i"[^{}]*\}', txt, re.S).group(0))
print(j['input_i'], j['input_tp'], j['input_lra'], j['input_thresh'], j['target_offset'])
EOF
)
ffmpeg -y -hide_banner -loglevel error -i "$A" \
  -af "loudnorm=I=-14:TP=-2:LRA=11:measured_I=$MI:measured_TP=$MTP:measured_LRA=$MLRA:measured_thresh=$MTH:offset=$OFF:linear=true,aresample=48000" \
  -c:a pcm_s16le "$TMP/norm.wav"

if [ "${AUDIO_ONLY:-0}" = 1 ]; then   # keep the existing picture, replace the soundtrack
  ffmpeg -y -hide_banner -loglevel error -i "$O" -i "$TMP/norm.wav" -map 0:v -map 1:a -c:v copy \
    -c:a aac -b:a 192k -ar 48000 -movflags +faststart -shortest "$TMP/remux.mp4"
  mv "$TMP/remux.mp4" "$O"; echo "$O"; exit 0
fi

ffmpeg -y -hide_banner -loglevel warning -nostats \
  -framerate $((FPS * S)) -i "$F/s%06d.jpg" -i "$TMP/norm.wav" \
  -filter_complex "[0:v]tmix=frames=$S,select='eq(mod(n\,$S)\,$((S - 1)))',setpts=N/($FPS*TB),format=yuv420p[v]" \
  -map "[v]" -map 1:a -r "$FPS" \
  -c:v libx264 -preset slow -crf "$CRF" -tune grain -profile:v high -pix_fmt yuv420p \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart -shortest "$O"
echo "$O"
