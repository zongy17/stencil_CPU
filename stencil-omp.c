#define _GNU_SOURCE             /* See feature_test_macros(7) */
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <mpi.h>
#include "common.h"

const char* version_name = "An omp version";

#ifndef SET_Y_SIZE
#define SET_Y_SIZE 8
#endif
#ifndef SET_X_SIZE
#define SET_X_SIZE 128
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

int mpisize, mpirank, namelen;
char processor_name[MPI_MAX_PROCESSOR_NAME] = {0};

void print_affinity() {
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);// 这个节点上on_line的有多少个cpu核
    printf("affinity_cpu=%02d of %d on %s ", sched_getcpu(), nproc, processor_name);// sched_getcpu()返回这个线程绑定的核心id
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_getaffinity error");
        exit(1);
    }
    printf("affinity_mask=");
    for (int i = 0; i < nproc; i++) printf("%d", CPU_ISSET(i, &mask));
}


void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

#define ACTIVATE_PARA long nproc=sysconf(_SC_NPROCESSORS_ONLN); cpu_set_t mask, get_mask; int pd; pd=omp_get_thread_num(); pd=pd%nproc; CPU_SET(pd, &mask); sched_setaffinity(0, sizeof(cpu_set_t), &mask);

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt, double * calc_time, double * comm_time) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Get_processor_name(processor_name, &namelen);
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);// 这个节点上on_line的有多少个cpu核
    // int proc_cpuid = sched_getcpu();
    // printf("process %d at affinity_cpu=%02d of %d on %s \n", grid_info->p_id, proc_cpuid, sysconf(_SC_NPROCESSORS_ONLN), processor_name);// sched_getcpu()返回这个线程绑定的核心id
    
    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];
        
        #pragma omp parallel
        {
            int ompsize = omp_get_num_threads();
            int omprank = omp_get_thread_num();// omp线程id
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(omprank % nproc, &mask);
            sched_setaffinity(0, sizeof(cpu_set_t), &mask);

            // if (t == 0){                
            //     #pragma omp critical
            //     {
            //         printf("%02d-%02d (%02d-%02d) ", mpirank, omprank, mpisize, ompsize);
            //         print_affinity();
            //         printf("\n");
            //     }
            // }

            #pragma omp for schedule(guided) collapse(2)
            for (int JJ = y_start; JJ < y_end; JJ += SET_Y_SIZE) {
                for (int II = x_start; II < x_end; II += SET_X_SIZE){
                    int REAL_Y_SIZE = MIN(SET_Y_SIZE, y_end - JJ);
                    int REAL_X_SIZE = MIN(SET_X_SIZE, x_end - II);
        
                    ptr_t a1_local = a1 + z_start*ldx*ldy + JJ*ldx + II;
                    cptr_t a0_local_Z = a0 + (a1_local - a1);
                    cptr_t a0_local_P = a0_local_Z + ldy*ldx;
                    cptr_t a0_local_N = a0_local_Z - ldy*ldx;

                    for(int z = z_start; z < z_end; ++z) {  
                        for(int y = 0; y < REAL_Y_SIZE; y++) {
                            #pragma unroll
                            for(int x = 0; x < REAL_X_SIZE; x++) {
                                a1_local[y*ldx+x] \
                                    = ALPHA_ZZZ * a0_local_Z[y*ldx+x]\
                                    + ALPHA_NZZ * a0_local_Z[y*ldx+x-1] \
                                    + ALPHA_PZZ * a0_local_Z[y*ldx+x+1] \
                                    + ALPHA_ZNZ * a0_local_Z[(y-1)*ldx+x] \
                                    + ALPHA_ZPZ * a0_local_Z[(y+1)*ldx+x] \
                                    + ALPHA_ZZN * a0_local_N[y*ldx+x] \
                                    + ALPHA_ZZP * a0_local_P[y*ldx+x];

                            }// x loop
                        }// y loop
                        a1_local = a1_local + ldx*ldy;
                        a0_local_N = a0_local_Z;
                        a0_local_Z = a0_local_P;
                        a0_local_P = a0_local_P + ldx*ldy;
                    }// z loop
                }
            }
        }
    }
    return buffer[nt % 2];
}

ptr_t stencil_27(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt, double * calc_time, double * comm_time) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Get_processor_name(processor_name, &namelen);
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);// 这个节点上on_line的有多少个cpu核

    // int proc_cpuid = sched_getcpu();
    // printf("process %d at affinity_cpu=%02d of %d on %s \n", grid_info->p_id, proc_cpuid, sysconf(_SC_NPROCESSORS_ONLN), processor_name);// sched_getcpu()返回这个线程绑定的核心id

    for(int t = 0; t < nt; ++t) {
        cptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];

        #pragma omp parallel 
        {
            int ompsize = omp_get_num_threads();
            int omprank = omp_get_thread_num();// omp线程id
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(omprank % nproc, &mask);
            sched_setaffinity(0, sizeof(cpu_set_t), &mask);

            // if (t == 0){                
            //     #pragma omp critical
            //     {
            //         printf("%02d-%02d (%02d-%02d) ", mpirank, omprank, mpisize, ompsize);
            //         print_affinity();
            //         printf("\n");
            //     }
            // }

            #pragma omp for schedule(guided) collapse(2)
            for (int JJ = y_start; JJ < y_end; JJ += SET_Y_SIZE) {
                for (int II = x_start; II < x_end; II += SET_X_SIZE){
                    int REAL_Y_SIZE = MIN(SET_Y_SIZE, y_end - JJ);
                    int REAL_X_SIZE = MIN(SET_X_SIZE, x_end - II);

                    ptr_t a1_local = a1 + z_start*ldx*ldy + JJ*ldx + II;
                    cptr_t a0_local_Z = a0 + (a1_local - a1);
                    cptr_t a0_local_P = a0_local_Z + ldy*ldx;
                    cptr_t a0_local_N = a0_local_Z - ldy*ldx;

                    for(int z = z_start; z < z_end; ++z) {
                        for(int y = 0; y < REAL_Y_SIZE; y++) {
                            #pragma unroll
                            for(int x = 0; x < REAL_X_SIZE; x++) {
                                a1_local[y*ldx+x] \
                                    = ALPHA_ZZZ * a0_local_Z[y*ldx+x] \
                                    + ALPHA_NZZ * a0_local_Z[y*ldx+x-1] \
                                    + ALPHA_PZZ * a0_local_Z[y*ldx+x+1] \
                                    + ALPHA_ZNZ * a0_local_Z[(y-1)*ldx+x] \
                                    + ALPHA_ZPZ * a0_local_Z[(y+1)*ldx+x] \
                                    + ALPHA_ZZN * a0_local_N[y*ldx+x] \
                                    + ALPHA_ZZP * a0_local_P[y*ldx+x] \
                                    + ALPHA_NNZ * a0_local_Z[(y-1)*ldx+x-1] \
                                    + ALPHA_PNZ * a0_local_Z[(y-1)*ldx+x+1] \
                                    + ALPHA_NPZ * a0_local_Z[(y+1)*ldx+x-1] \
                                    + ALPHA_PPZ * a0_local_Z[(y+1)*ldx+x+1] \
                                    + ALPHA_NZN * a0_local_N[y*ldx+x-1] \
                                    + ALPHA_PZN * a0_local_N[y*ldx+x+1] \
                                    + ALPHA_NZP * a0_local_P[y*ldx+x-1] \
                                    + ALPHA_PZP * a0_local_P[y*ldx+x+1] \
                                    + ALPHA_ZNN * a0_local_N[(y-1)*ldx+x] \
                                    + ALPHA_ZPN * a0_local_N[(y+1)*ldx+x] \
                                    + ALPHA_ZNP * a0_local_P[(y-1)*ldx+x] \
                                    + ALPHA_ZPP * a0_local_P[(y+1)*ldx+x] \
                                    + ALPHA_NNN * a0_local_N[(y-1)*ldx+x-1] \
                                    + ALPHA_PNN * a0_local_N[(y-1)*ldx+x+1] \
                                    + ALPHA_NPN * a0_local_N[(y+1)*ldx+x-1] \
                                    + ALPHA_PPN * a0_local_N[(y+1)*ldx+x+1] \
                                    + ALPHA_NNP * a0_local_P[(y-1)*ldx+x-1] \
                                    + ALPHA_PNP * a0_local_P[(y-1)*ldx+x+1] \
                                    + ALPHA_NPP * a0_local_P[(y+1)*ldx+x-1] \
                                    + ALPHA_PPP * a0_local_P[(y+1)*ldx+x+1];
                            }// x
                        }// y
                        a1_local = a1_local + ldx*ldy;
                        a0_local_N = a0_local_Z;
                        a0_local_Z = a0_local_P;
                        a0_local_P = a0_local_P + ldx*ldy;
                    }// z
                }
            }
        }
    }
    return buffer[nt % 2];
}
