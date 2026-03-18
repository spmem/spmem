set -e
cd "$(dirname "$0")"

# Ensure PyCUDA can find nvcc (CUDA toolkit may exist but PATH not set).
export PATH="/usr/local/cuda/bin:$PATH"



python run_data.py \
    --input_dir ${name}.npz \
    --name "${name}" \
    --save_dir ${save_dir} \
    --render_ply \
    --n_imgs 49 \
    --iqr_filter
