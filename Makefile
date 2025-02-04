# Compiler and flags
GXX = g++
NVCC = /usr/local/cuda/bin/nvcc
CPATH = /usr/local/cuda/include
CXXFLAGS = -std=c++23 -g -Og
NVCCFLAGS = -O3 -c
LDFLAGS = -Wl,-rpath,/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart -lnvinfer -lnvonnxparser -lcuda

all: build-simple infer-simple plugin.so

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Compile C++ sources
%.o: %.cpp
	$(GXX) $(CXXFLAGS) -c -o $@ $< -I$(CPATH)

# PIC object files for shared library
%.pic.o: %.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -o $@ $<

%.pic.o: %.cpp
	$(GXX) $(CXXFLAGS) -fPIC -c -o $@ $< -I$(CPATH)

build-simple: buildSimple.o addKernel.o addPlugin.o
	$(GXX) -o $@ $^ $(LDFLAGS)

infer-simple: inferSimple.o addKernel.o addPlugin.o
	$(GXX) -o $@ $^ $(LDFLAGS)

# Shared library target
plugin.so: addPlugin.pic.o addKernel.pic.o registerPlugin.pic.o
	$(GXX) -shared -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o *.pic.o build-simple infer-simple plugin.so
