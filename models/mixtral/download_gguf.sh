#!/bin/bash
set -e

# Download pre-quantized Mixtral 8x7B GGUF files from Hugging Face.
# Usage: ./download_gguf.sh [QUANT]
#   QUANT: quantization variant (default: Q4_K_M)

QUANT="${1:-Q4_K_M}"
REPO="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
FILENAME="mixtral-8x7b-instruct-v0.1.${QUANT}.gguf"
URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
OUTFILE="mixtral-8x7b-${QUANT}.gguf"

echo "Downloading Mixtral 8x7B ${QUANT} from ${REPO}..."
echo "WARNING: This file is ~24GB for Q4_K_M. Make sure you have enough disk space."
curl -fSL "${URL}" -o "${OUTFILE}"
echo "Done: ${OUTFILE}"
