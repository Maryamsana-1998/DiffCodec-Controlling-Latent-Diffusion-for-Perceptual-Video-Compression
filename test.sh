#!/bin/bash

# Input folder
INPUT_DIR="data/test_videos"
# List of videos
VIDEOS=("Beauty" "Jockey" "Bosphorus" "ShakeNDry" "HoneyBee" "YachtRide" "ReadySteadyGo")
# VIDEOS=("HoneyBee" "YachtRide")
# VIDEOS=("MarketPlace" "RitualDance")
# VIDEOS=("BasketballDrive" "Cactus")
# VIDEOS=("BQTerrace")
# Parameters
WIDTH=1920
HEIGHT=1080
FPS=120
FRAMES=96
BPPS=(0.006 0.01 0.05)

# Loop over all videos
for VIDEO in "${VIDEOS[@]}"; do
    INPUT="${INPUT_DIR}/${VIDEO}_${WIDTH}x${HEIGHT}_${FPS}fps_420_8bit_YUV.yuv"
    # INPUT="${INPUT_DIR}/B_${VIDEO}_${WIDTH}x${HEIGHT}_${FPS}fps_10bit_420.yuv"

    for BPP in "${BPPS[@]}"; do
        # Compute bitrate in bps
        BITRATE=$(echo "$BPP * $WIDTH * $HEIGHT * $FPS" | bc)
        BITRATE_INT=$(printf "%.0f" $BITRATE)

        # Folder name
        OUT_DIR="benchmark_results/gop8_results/h264_uvg_gop8/${VIDEO}/bpp_${BPP}"
        mkdir -p "$OUT_DIR"

        echo "Encoding $VIDEO at bpp=$BPP (bitrate=$BITRATE_INT bps)"

        # Encode with HEVC (96 frames) # -pix_fmt yuv420p10le or yuv420p
        ffmpeg -s:v ${WIDTH}x${HEIGHT} -pix_fmt yuv420p -r $FPS -i "$INPUT" \
        -frames:v $FRAMES -c:v libx264 -preset fast -b:v ${BITRATE_INT} \
        -x265-params "keyint=8:min-keyint=8:scenecut=0" \
        "$OUT_DIR/output.mp4" -y

        # Extract per-frame sizes
        ffprobe -show_frames -select_streams v:0 -print_format csv -show_entries frame=pkt_size,pict_type "$OUT_DIR/output.mp4" \
        > "$OUT_DIR/frame_sizes.csv"

        # Compute intra vs inter storage
        awk -F',' '
        $3=="I"{intra+=$2}
        $3=="P"||$3=="B"{inter+=$2}
        END{
            print "Intra bytes: " intra > "intra_inter_storage.txt"
            print "Inter bytes: " inter >> "intra_inter_storage.txt"
            print "Total bytes: " intra+inter >> "intra_inter_storage.txt"
        }' "$OUT_DIR/frame_sizes.csv"

        mv intra_inter_storage.txt "$OUT_DIR/"

        # Save decoded frames as images
        ffmpeg -i "$OUT_DIR/output.mp4" "$OUT_DIR/frame_%03d.png" -y
    done
done
