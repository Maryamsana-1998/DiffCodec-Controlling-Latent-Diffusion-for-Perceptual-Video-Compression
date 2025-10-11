#!/bin/bash
# Usage: ./extract_all_vvc_frames_fixed.sh <vvc_folder> <yuv_folder> <width> <height>
# Example:
# ./extract_all_vvc_frames_fixed.sh /data/.../vvc_results /data/.../yuv_videos 1920 1080

set -e

VVC_FOLDER=$1
YUV_FOLDER=$2
WIDTH=$3
HEIGHT=$4

if [ -z "$VVC_FOLDER" ] || [ -z "$YUV_FOLDER" ] || [ -z "$WIDTH" ] || [ -z "$HEIGHT" ]; then
    echo "Usage: $0 <vvc_folder> <yuv_folder> <width> <height>"
    exit 1
fi

VVC_DECODER="/home/maryamsana_98/vvc_build/vvdec/bin/release-static/vvdecapp"

for VVC_FILE in "$VVC_FOLDER"/*.vvc; do
    echo "Processing $VVC_FILE ..."

    BASENAME=$(basename "$VVC_FILE" .vvc)
    VIDEO_NAME=$(echo "$BASENAME" | cut -d'_' -f2)
    BPP=$(echo "$BASENAME" | grep -oP 'bpp[0-9.]+')

    # YUV_FILE="$YUV_FOLDER/${VIDEO_NAME}_${WIDTH}x${HEIGHT}_50.yuv"
    # if [ ! -f "$YUV_FILE" ]; then
    #     echo "Warning: YUV file $YUV_FILE not found. Skipping..."
    #     continue
    # fi

    OUT_DIR="decoded_results/${VIDEO_NAME}/${BPP}"
    mkdir -p "$OUT_DIR"

    # Decode VVC to YUV and capture log for intra/inter analysis
    DECODER_LOG="$OUT_DIR/vvdec_log.txt"
    echo "Decoding $VVC_FILE ..."
    "$VVC_DECODER" -b "$VVC_FILE" -o "$OUT_DIR/output_decoded.yuv" > "$DECODER_LOG" 2>&1

    # Count decoded frames from the log
    FRAME_COUNT=$(grep -c "POC" "$DECODER_LOG")
    echo "Detected $FRAME_COUNT frames in decoder log"

    # Truncate YUV to exact number of frames to avoid over-extraction
    FRAME_SIZE=$((WIDTH * HEIGHT * 3 / 2))  # YUV420p
    truncate -s $((FRAME_COUNT * FRAME_SIZE)) "$OUT_DIR/output_decoded.yuv"

    # Convert YUV to PNG frames
    echo "Extracting frames from YUV..."
    ffmpeg -s:v ${WIDTH}x${HEIGHT} -pix_fmt yuv420p -i "$OUT_DIR/output_decoded.yuv" "$OUT_DIR/f%03d.png"

    # Compute intra vs inter storage from decoder log
    echo "Computing intra vs inter storage..."
    INTRA_BYTES=$(grep -Eo "POC.*\(I-SLICE.*\) \[DT[ ]+[0-9.]+" "$DECODER_LOG" \
                  | awk '{sum+=$NF} END{print sum}')
    INTER_BYTES=$(grep -Eo "POC.*\((B|P)-SLICE.*\) \[DT[ ]+[0-9.]+" "$DECODER_LOG" \
                  | awk '{sum+=$NF} END{print sum}')
    TOTAL_BYTES=$(awk "BEGIN{print $INTRA_BYTES + $INTER_BYTES}")

    cat <<EOT > "$OUT_DIR/intra_inter_storage.txt"
Intra bytes: $INTRA_BYTES
Inter bytes: $INTER_BYTES
Total bytes: $TOTAL_BYTES
EOT

    echo "✅ Finished $VIDEO_NAME / $BPP"
done

echo "All videos processed. Results are in decoded_results/"
