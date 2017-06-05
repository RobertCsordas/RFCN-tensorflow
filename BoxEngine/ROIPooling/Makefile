PYTHON?=$(shell which python)
TF_INC := $(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
USE_OLD_EABI?=1
USE_GPU?=1

ifeq (${USE_OLD_EABI}, 1)
	EABIFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
else
	EABIFLAGS=
endif

ifeq (${USE_GPU}, 1)
	LINKDEPS=roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o
	GPUFLAGS=-DUSE_GPU=1
else
	GPUFLAGS=-DUSE_GPU=0
	LINKDEPS=roi_pooling.o roi_pooling_grad.o
endif

all: roi_pooling.so

roi_pooling_grad.o: roi_pooling_grad.cc
	g++ -std=c++11 ${EABIFLAGS} ${GPUFLAGS} -c -o roi_pooling_grad.o roi_pooling_grad.cc -I $(TF_INC) -fPIC -O2

roi_pooling.o: roi_pooling.cc
	g++ -std=c++11 ${EABIFLAGS} ${GPUFLAGS} -c -o roi_pooling.o roi_pooling.cc -I $(TF_INC) -fPIC -O2

roi_pooling.cu.o: roi_pooling.cu.cc
	nvcc -std=c++11 -c -o roi_pooling.cu.o roi_pooling.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

roi_pooling_grad.cu.o: roi_pooling_grad.cu.cc
	nvcc -std=c++11 -c -o roi_pooling_grad.cu.o roi_pooling_grad.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

roi_pooling.so: ${LINKDEPS}
	g++ -std=c++11 ${EABIFLAGS} -shared -o roi_pooling.so ${LINKDEPS} -I $(TF_INC) -fPIC -O2


#roi_pooling.so: roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o
#	g++ -std=c++11  -D_GLIBCXX_USE_CXX11_ABI=0 -shared -o roi_pooling.so roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o -I $(TF_INC) -fPIC -O2


clean:
	-rm *.so
	-rm *.o
