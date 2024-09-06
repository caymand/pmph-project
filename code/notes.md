# TODO:
- Why is warp-frag-tiling slower with same config?
- Coalesced shared access, change stride of loaded fragments
- Bank conflicts, offset
- More clever access pattern
- Profile A100, nsight-compute
- Use `cudaMallocPitch()` for global memory alloc

- Double buffer registers
- Multiple register loads per thread spawn
  - Try C in reg, shared, and global
- Transpose on load to shared
- Producer threads, consumer threads

- Try doing swizzled loads in sequential phases


# Results
## Local
| Command                                                              | TFLOPS  |
|----------------------------------------------------------------------|---------|
| ./compile.sh 4 4 1 1 1 2 2 2 -DLOAD_TYPE=float4 -DNOUNROLL && ./main | 24077.9 |

## A100
| Command                                                          | TFLOPS |
|------------------------------------------------------------------|--------|
| ./compile.sh 4 4 1 2 2 2 -DLOAD_TYPE=float4 && ./main            | 133995 |
| ./compile.sh 4 4 2 2 2 2 -DLOAD_TYPE=float4 -DNOUNROLL && ./main | 140373 |
| ./compile.sh 2 4 2 4 2 2 -DLOAD_TYPE=float4 -DNOUNROLL && ./main | 145361 |
| ./compile.sh 4 2 2 2 4 2 -DLOAD_TYPE=float4 -DNOUNROLL && ./main | 149569 |
| ./compile.sh 4 2 4 1 1 1 4 4 -DLOAD_TYPE=float4 && ./main        | 145392 |

