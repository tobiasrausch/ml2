SHELL := /bin/bash

# Flags for static compile 
ifeq (${STATIC}, 1)
	LIBTAG=static
	CVSHARED = "OFF"
else
	LIBTAG=shared
	CVSHARED = "ON"
endif

# Targets
TARGETS = .libtorch .opencv .mnist ml
PBASE=$(shell pwd)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	PYTORCHLIB="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip"
else
	PYTORCHLIB="https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-${LIBTAG}-with-deps-latest.zip"
endif
SOURCES = $(wildcard *.h) $(wildcard *.cpp)

all: ${TARGETS}

.mnist:
	mkdir mnist && cd mnist && wget 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' && wget 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' && wget 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' && wget 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' && gunzip *.gz && cd .. && touch .mnist

.libtorch:
	wget -O libtorch.zip ${PYTORCHLIB} && unzip libtorch.zip && rm libtorch.zip && touch .libtorch

.opencv:
	wget -O opencv.zip 'https://github.com/opencv/opencv/archive/4.5.5.zip' && unzip opencv.zip && rm opencv.zip && mkdir -p opencvbuild && cd opencvbuild && cmake ../opencv-* && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=${PBASE}/opencv -D BUILD_SHARED_LIBS=${CVSHARED} -DOPENCV_GENERATE_PKGCONFIG=ON -D BUILD_ZLIB=ON -D BUILD_PNG=ON -D WITH_OPENEXR=OFF -D WITH_JPEG=OFF -D WITH_JASPER=OFF -D WITH_TIFF=OFF -D WITH_WEBP=OFF -D WITH_OPENCL=OFF -D WITH_GTK=${CVSHARED} -D WITH_FFMPEG=OFF -D WITH_1394=OFF -D WITH_IPP=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_apps=OFF ../opencv-* &&  make -j 4 && make install && cd ../ && rm -rf opencvbuild/ && touch .opencv;

.htslib:
	wget -O htslib.tar.bz2 'https://github.com/samtools/htslib/releases/download/1.15.1/htslib-1.15.1.tar.bz2' && bunzip2 htslib.tar.bz2 && tar -xf htslib.tar && rm htslib.tar && mv htslib-* htslib && cd htslib && autoheader && autoconf && ./configure --disable-s3 --disable-gcs --disable-libcurl --disable-plugins && make && make lib-static && cd ../ && touch .htslib

ml: .libtorch .opencv .htslib .mnist ${SOURCES}
	mkdir -p build && cd build && cmake -DCMAKE_PREFIX_PATH="${PBASE}/libtorch;${PBASE}/opencv" .. && make

clean:
	rm -rf build/

distclean:
	rm -rf $(TARGETS) $(TARGETS:=.o) build/ opencv* libtorch/ htslib/
