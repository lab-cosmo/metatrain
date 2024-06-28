#!/bin/bash
set -e

echo "Generating data for testing..."

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd $ROOT_DIR

mtt train options.yaml -o model-32-bit.pt -r base_precision=32 > /dev/null
mtt train options.yaml -o model-64-bit.pt -r base_precision=64 > /dev/null
