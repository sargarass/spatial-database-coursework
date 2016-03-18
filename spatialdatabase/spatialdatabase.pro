QMAKE_CC = gcc-4.9.3
QMAKE_CXX = g++-4.9.3
QMAKE_LINK_C       = $$QMAKE_CC
QMAKE_LINK_C_SHLIB = $$QMAKE_CC
QMAKE_LINK       = $$QMAKE_CXX
QMAKE_LINK_SHLIB = $$QMAKE_CXX

QT += core gui

CONFIG += c++11

TARGET = spatialdatabase
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app
NVCC_OPTIONS = --use_fast_math -m64
CUDA_DIR = /opt/cuda
CUDA_ARCH = sm_20
INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart
CUDA_SOURCES += cuda_stuff.cu
SOURCES += main.cpp
QMAKE_CFLAGS = -std=c++11 -m64
cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS -c -arch=$$CUDA_ARCH -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} 2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o
QMAKE_EXTRA_COMPILERS += cuda

HEADERS += \
    cuda_staff.h


