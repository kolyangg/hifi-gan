#!/bin/bash
# Exit on error
set -e

# Create and activate conda environment (if not already created)
conda env list | grep hifigan || conda create -n hifigan python=3.12.7 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hifigan


# Download checkpoints

# Download a checkpoint using gdown
pip install gdown

mkdir -p checkpoints/hifigan
gdown --id 1fnkOteyRdPq4Gh2cfso3gqqrC6inLWsF -O checkpoints/hifigan/generator_v3.pt
gdown --id 1mke75axgO2sdJ41GL2HTrcb4KyAl0i45 -O checkpoints/hifigan/config.json

# Run inference (adjust paths as needed)
# python3 inference_NS.py --checkpoint_file generator_v3 \
#     --input_wavs_dir ../../open-universe/data/voicebank_demand/16k/test/noisy \
#     --output_dir ../../open-universe/results/voicebank_demand/16k/test/noisy/hifi-gan


# ./diff_inference.sh diffwave-ljspeech-22kHz-1000578.pt testing_files/noisy testing_results
