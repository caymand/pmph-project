# TODO: use other arch?
CXX = nvcc #nvcc -O3 -arch=sm_80 # g++
CXXFLAGS = -O2 -std=c++17 -arch=sm_80  #-Wall  -pedantic
CFILES = main.cu goldenSeq.cpp
CUFILES = matmul.cu # matmul-tensor.cu
HEADERS = matmul.cuh helpers.h helpers.tpp goldenSeq.h matmul-tensor.cuh

MAIN = main
GOLDEN_CPU = golden_cpu.o

# depends on building the main file
all: $(MAIN)

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CUFILES) $(CFILES) -o $(MAIN)

clean:
	- rm main