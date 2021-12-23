#!/usr/bin/env bash
#SBATCH --nodes=1
#BATCH --ntasks-per-node=1

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <executable> <number of omp threads per process>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0
export OMP_NUM_THREADS=$2

DATAPATH=/storage/readonly/stencil_data

salloc -N 1 --exclusive --ntasks-per-node 1 mpirun $1 7 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_7_256x256x256_16steps
salloc -N 1 --exclusive --ntasks-per-node 1 mpirun $1 7 384 384 384 16 ${DATAPATH}/stencil_data_384x384x384 ${DATAPATH}/stencil_answer_7_384x384x384_16steps
salloc -N 1 --exclusive --ntasks-per-node 1 mpirun $1 7 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_7_512x512x512_16steps
salloc -N 1 --exclusive --ntasks-per-node 1 mpirun $1 27 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_27_256x256x256_16steps
salloc -N 1 --exclusive --ntasks-per-node 1 mpirun $1 27 384 384 384 16 ${DATAPATH}/stencil_data_384x384x384 ${DATAPATH}/stencil_answer_27_384x384x384_16steps
salloc -N 1 --exclusive --ntasks-per-node 1 mpirun $1 27 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_27_512x512x512_16steps
