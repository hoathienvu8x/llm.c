#!/bin/bash
set -e

# Download pre-quantized OLMoE-1B-7B GGUF files from Hugging Face.
# Usage: ./download_gguf.sh [QUANT]
#   QUANT: quantization variant (default: q4_k_m)
#          Available: q4_k_m, q4_k_s, q5_k_m, q5_k_s, q6_k, q8_0

QUANT="${1:-q4_k_m}"
REPO="allenai/OLMoE-1B-7B-0924-Instruct-GGUF"
FILENAME="olmoe-1b-7b-0924-instruct-${QUANT}.gguf"
URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
OUTFILE="olmoe-1b-7b-${QUANT}.gguf"

echo "Downloading OLMoE-1B-7B ${QUANT} from ${REPO}..."
curl -fSL "${URL}" -o "${OUTFILE}"
echo "Done: ${OUTFILE}"
