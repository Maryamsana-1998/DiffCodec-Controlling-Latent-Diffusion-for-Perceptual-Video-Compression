#!/bin/bash
# === User-configurable paths ===
VVC_HOME=$HOME/vvc_build
FFMPEG=$VVC_HOME/ffmpeg/bin/ffmpeg
VVDEC=$VVC_HOME/vvdec/bin/vvdecapp
INPUT_DIR=data/test_videos
OUT_DIR=vvc_results

mkdir -p "$OUT_DIR"

# === Ensure FFmpeg finds its libs ===
export LD_LIBRARY_PATH=$VVC_HOME/ffmpeg/lib:$LD_LIBRARY_PATH

# === Encoding parameters ===
GOP=8
BPP=0.01
FRAMES=97

# === Loop through all YUV files ===
for INPUT in "$INPUT_DIR"/*.yuv; do
  [ -e "$INPUT" ] || continue

  BASENAME=$(basename "$INPUT" .yuv)
  WIDTH=1920
  HEIGHT=1080
  FPS=120

  echo "==== Processing $BASENAME ===="

  # Compute bitrate in bits per second from bpp
  # bpp * width * height * fps
  BITRATE=$(echo "$BPP * $WIDTH * $HEIGHT * $FPS" | bc)
  BITRATE_INT=$(printf "%.0f" "$BITRATE")

  echo "Target bitrate: ${BITRATE_INT} bps"

  # Encode first 97 frames
  $FFMPEG -y \
    -f rawvideo -pix_fmt yuv420p -s:v ${WIDTH}x${HEIGHT} -r $FPS \
    -i "$INPUT" \
    -frames:v $FRAMES \
    -c:v libvvenc \
    -preset medium \
    -g $GOP \
    -b:v ${BITRATE_INT} \
    "$OUT_DIR/${BASENAME}_g${GOP}_bpp${BPP}.vvc"

  # Decode the bitstream back to YUV
  $VVDEC \
    "$OUT_DIR/${BASENAME}_g${GOP}_bpp${BPP}.vvc" \
    "$OUT_DIR/${BASENAME}_decoded.yuv"

  echo "Done: $BASENAME"
  echo
done
