SHELL := /bin/bash

# Targets
TARGETS = .libtorch example-app
PBASE=$(shell pwd)

SOURCES = $(wildcard *.h) $(wildcard *.cpp)

all: ${TARGETS}

.libtorch:
	wget 'https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip' && unzip libtorch-shared-with-deps-latest.zip && rm libtorch-shared-with-deps-latest.zip && touch .libtorch

build: .libtorch ${SOURCES}
	mkdir -p build && cd build && cmake -DCMAKE_PREFIX_PATH=${PBASE}/libtorch .. && cmake --build . --config Release

clean:
	rm -rf $(TARGETS) $(TARGETS:=.o) build/
