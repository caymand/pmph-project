#!/bin/sh
WARP_TILES_M=$1
WARP_TILES_N=$2
WARP_TILES_K=$3
BLOCK_TILES_M=$4
BLOCK_TILES_N=$5
BLOCK_TILES_K=$6

shift 6
set -x
nvcc -Xptxas=-v -O3 -std=c++17 -arch=sm_80 matmul.cu main.cu goldenSeq.cpp -o main \
    -DWARP_TILES_M=$WARP_TILES_M \
    -DWARP_TILES_N=$WARP_TILES_N \
    -DWARP_TILES_K=$WARP_TILES_K \
    -DBLOCK_TILES_M=$BLOCK_TILES_M \
    -DBLOCK_TILES_N=$BLOCK_TILES_N \
    -DBLOCK_TILES_K=$BLOCK_TILES_K "$@"
