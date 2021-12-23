#!/usr/bin/env bash
#SBATCH --nodes=1
#BATCH --ntasks-per-node=1

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <executable> <number of nodes> <number of processes per node>  <number of omp threads per process>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0
export OMP_NUM_THREADS=$4
export OMP_PROC_BIND=true

stride=$[128 / $3]
echo $stride
DATAPATH=/storage/readonly/stencil_data

salloc -N $2 --exclusive --ntasks-per-node $3 mpirun --map-by slot:PE=$stride --bind-to core $1 7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
salloc -N $2 --exclusive --ntasks-per-node $3 mpirun --map-by slot:PE=$stride --bind-to core $1 7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384
salloc -N $2 --exclusive --ntasks-per-node $3 mpirun --map-by slot:PE=$stride --bind-to core $1 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
salloc -N $2 --exclusive --ntasks-per-node $3 mpirun --map-by slot:PE=$stride --bind-to core $1 27 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
salloc -N $2 --exclusive --ntasks-per-node $3 mpirun --map-by slot:PE=$stride --bind-to core $1 27 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384
salloc -N $2 --exclusive --ntasks-per-node $3 mpirun --map-by slot:PE=$stride --bind-to core $1 27 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
