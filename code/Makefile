# TODO: use other arch?
CXX = nvcc #nvcc -O3 -arch=sm_80 # g++
CXXFLAGS = -O3 -std=c++14 -arch=sm_80  #-Wall  -pedantic
CFILES = main.cu
CUFILES = matmul.cu
HEADERS = matmul.cuh helpers.h helpers.tpp goldenSeq.h

MAIN = main

# depends on building the main file
all: $(MAIN)

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CUFILES) $(CFILES) -o $(MAIN)

clean:
	- rm main