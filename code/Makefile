# TODO: use other arch?
CXX = g++ #nvcc -O3 -arch=sm_80
CXXFLAGS = -O0 -std=c++20 
CFILES = helpers.cpp main.cpp
HEADERS = helpers.h

MAIN = main

# depends on building the main file
all: $(MAIN)

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(CFILES) -o $(MAIN)

clean:
	rm main