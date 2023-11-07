#!/bin/sh
WARP_TILES_M=$1
WARP_TILES_N=$2
BLOCK_TILES_M=$3
BLOCK_TILES_N=$4
BLOCK_TILES_K=$5

shift 5
set -x
nvcc -Xptxas=-v -O3 -std=c++17 -arch=sm_80 matmul.cu main.cu goldenSeq.cpp -o main \
    -DWARP_TILES_M=$WARP_TILES_M \
    -DWARP_TILES_N=$WARP_TILES_N \
    -DBLOCK_TILES_M=$BLOCK_TILES_M \
    -DBLOCK_TILES_N=$BLOCK_TILES_N \
    -DBLOCK_TILES_K=$BLOCK_TILES_K "$@"
