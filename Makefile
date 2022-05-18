SHELL := /bin/bash

# Targets
TARGETS = .libtorch .opencv .mnist ml
PBASE=$(shell pwd)

SOURCES = $(wildcard *.h) $(wildcard *.cpp)

all: ${TARGETS}

.mnist:
	mkdir mnist && cd mnist && wget 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' && wget 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' && wget 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' && wget 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' && gunzip *.gz && cd .. && touch .mnist

.libtorch:
	wget 'https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip' && unzip libtorch-cxx11-abi-shared-with-deps-latest.zip && rm libtorch-cxx11-abi-shared-with-deps-latest.zip && touch .libtorch

.opencv:
	wget -O opencv.zip 'https://github.com/opencv/opencv/archive/4.5.5.zip' && unzip opencv.zip && rm opencv.zip && mkdir -p opencv && cd opencv && cmake ../opencv-* && cmake --build . -- -j 8 && cd .. && touch .opencv

.htslib:
	wget -O htslib.tar.bz2 'https://github.com/samtools/htslib/releases/download/1.15.1/htslib-1.15.1.tar.bz2' && bunzip2 htslib.tar.bz2 && tar -xf htslib.tar && rm htslib.tar && mv htslib-* htslib && cd htslib && autoheader && autoconf && ./configure --disable-s3 --disable-gcs --disable-libcurl --disable-plugins && make && make lib-static && cd ../ && touch .htslib

ml: .libtorch .opencv .htslib .mnist ${SOURCES}
	mkdir -p build && cd build && cmake -DCMAKE_PREFIX_PATH="${PBASE}/libtorch;${PBASE}/opencv" .. && make

clean:
	rm -rf build/

distclean:
	rm -rf $(TARGETS) $(TARGETS:=.o) build/ opencv* libtorch/ htslib/
