#!/bin/sh
FRAGS_M=$1
FRAGS_N=$2
FRAGS_K=$3
WARP_TILES_M=$4
WARP_TILES_N=$5
BLOCK_TILES_M=$6
BLOCK_TILES_N=$7
BLOCK_TILES_K=$8

shift 8
set -x
nvcc -Xptxas=-v -O3 -std=c++17 -lcublas -arch=sm_80 matmul.cu main.cu goldenSeq.cpp -o main \
    -DFRAGS_M=$FRAGS_M \
    -DFRAGS_N=$FRAGS_N \
    -DFRAGS_K=$FRAGS_K \
    -DWARP_TILES_M=$WARP_TILES_M \
    -DWARP_TILES_N=$WARP_TILES_N \
    -DBLOCK_TILES_M=$BLOCK_TILES_M \
    -DBLOCK_TILES_N=$BLOCK_TILES_N \
    -DBLOCK_TILES_K=$BLOCK_TILES_K "$@"
