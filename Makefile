
CC = mpicc 
VEC = -O3 -fomit-frame-pointer -march=armv8-a -ffast-math
OPT = $(VEC) -DALIGNED #-fopt-info-vec-optimized #-DuseINDEX
CFLAGS = -w -std=c99 $(OPT) -fopenmp # -Wall
LDFLAGS = -Wall -fopenmp -lm $(OPT)
LDLIBS = $(LDFLAGS)

targets = benchmark-naive benchmark-mpi benchmark-omp benchmark-hyb
objects = check.o benchmark.o stencil-naive.o stencil-mpi.o stencil-omp.o stencil-hyb.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : check.o benchmark.o stencil-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-mpi : check.o benchmark.o stencil-mpi.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-omp : check.o benchmark.o stencil-omp.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-hyb : check.o benchmark.o stencil-hyb.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)