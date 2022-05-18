SHELL := /bin/bash

# Targets
TARGETS = .libtorch .opencv example
PBASE=$(shell pwd)

SOURCES = $(wildcard *.h) $(wildcard *.cpp)

all: ${TARGETS}

.libtorch:
	wget 'https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip' && unzip libtorch-cxx11-abi-shared-with-deps-latest.zip && rm libtorch-cxx11-abi-shared-with-deps-latest.zip && touch .libtorch

.opencv:
	wget -O opencv.zip 'https://github.com/opencv/opencv/archive/4.5.5.zip' && unzip opencv.zip && rm opencv.zip && mkdir -p opencv && cd opencv && cmake ../opencv-* && cmake --build . -- -j 8 && cd .. && touch .opencv

example: .libtorch .opencv ${SOURCES}
	mkdir -p build && cd build && cmake -DCMAKE_PREFIX_PATH="${PBASE}/libtorch;${PBASE}/opencv" .. && make

clean:
	rm -rf build/

distclean:
	rm -rf $(TARGETS) $(TARGETS:=.o) build/ opencv* libtorch/
