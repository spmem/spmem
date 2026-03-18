
set -e
cd "$(dirname "$0")"

export PATH="/usr/local/cuda/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

python infer_stream.py \
    --output_dir outputs_stream_teaser\
    --input_video examples/000000000019.4_100/Vid.mp4 \
    --height 480 \
    --width 720 \
    --cond_frames 5 \
    --ref_frames 5 \
    --num_frames 49 \
    --speed 0.05 \
    --angle 0.5 \
    --da3_process_res 504 \
    # --naive_ref \
    # --force_ref_count \
    "$@"
