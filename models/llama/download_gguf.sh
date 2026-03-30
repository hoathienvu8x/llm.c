#!/bin/bash
set -e

QUANT="${1:-Q4_K_M}"
REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
FILENAME="Llama-3.2-3B-Instruct-${QUANT}.gguf"
URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
OUTFILE="llama-3.2-3b-instruct-${QUANT}.gguf"

echo "Downloading Llama 3.2 3B Instruct ${QUANT} from ${REPO}..."
echo "File: ~2GB"
curl -fSL "${URL}" -o "${OUTFILE}"
echo "Done: ${OUTFILE}"
