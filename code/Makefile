# TODO: use other arch?
CXX = nvcc #nvcc -O3 -arch=sm_80 # g++
CXXFLAGS = -O3 -std=c++14 -arch=sm_80  #-Wall  -pedantic
CFILES = main.cu
CUFILES = matmul.cu # matmul-tensor.cu
HEADERS = matmul.cuh helpers.h helpers.tpp goldenSeq.h matmul-tensor.cuh

MAIN = main
GOLDEN_CPU = golden_cpu.o

# depends on building the main file
all: $(MAIN) matmul-tensor

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CUFILES) $(CFILES) -o $(MAIN)

$(GOLDEN_CPU): goldenSeq.h goldenSeq.cpp
	g++ -O3 -fopenmp -o $(GOLDEN_CPU) -c goldenSeq.cpp

matmul-tensor: matmul-tensor.cu
	$(CXX) -o matmul-tensor matmul-tensor.cu

clean:
	- rm main