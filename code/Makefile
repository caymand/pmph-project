# TODO: use other arch?
CXX = nvcc #nvcc -O3 -arch=sm_80 # g++
CXXFLAGS = -O3 -std=c++17 -arch=sm_80  #-Wall  -pedantic
CFILES = main.cu goldenSeq.cpp
CUFILES = matmul.cu # matmul-tensor.cu
HEADERS = matmul.cuh helpers.h helpers.tpp goldenSeq.h matmul-tensor.cuh
OPTIONS= -Xptxas=-v \
	-lcublas \
	-DWARP_TILES_M=1 \
	-DWARP_TILES_N=2 \
	-DBLOCK_TILES_M=8 \
	-DBLOCK_TILES_N=4 \
	-DBLOCK_TILES_K=2 \
	-DBLOCKS_PER_SM=2 \
	-DUNROLL \
	-DLOAD_TYPE=float2

MAIN = main
GOLDEN_CPU = golden_cpu.o

# depends on building the main file
all: $(MAIN)

run: $(MAIN)
	./$(MAIN)

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CUFILES) $(CFILES) $(OPTIONS) -o $(MAIN)

clean:
	- rm main