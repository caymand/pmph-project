# TODO: use other arch?
CXX = nvcc #nvcc -O3 -arch=sm_80 # g++
CXXFLAGS = -O3 -std=c++14 -arch=sm_80  #-Wall  -pedantic
CFILES = main.cu
CUFILES = matmul.cu # matmul-tensor.cu
HEADERS = matmul.cuh helpers.h helpers.tpp goldenSeq.h matmul-tensor.cuh

MAIN = main

# depends on building the main file
all: $(MAIN) matmul-tensor

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CUFILES) $(CFILES) -o $(MAIN)

matmul-tensor: matmul-tensor.cu
	$(CXX) -o matmul-tensor matmul-tensor.cu

clean:
	- rm main