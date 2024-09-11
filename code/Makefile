CXX = nvcc
CXXFLAGS = -O3 -std=c++17 -arch=sm_80  #-Wall  -pedantic
CUFILES = main.cu
HEADERS = matmul.cuh helpers.h matmul-tensor-naive.cuh matmul-tensor.cuh
OPTIONS= -Xptxas=-v \
	--expt-relaxed-constexpr \
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
	$(CXX) $(CXXFLAGS) $(CUFILES) $(OPTIONS) -o $(MAIN)

clean:
	- rm main