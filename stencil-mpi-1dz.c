#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
const char* version_name = "A mpi version with 1D partition in z";
#include "common.h"

#ifndef SET_Y_SIZE
#define SET_Y_SIZE 8
#endif
#ifndef SET_X_SIZE
#define SET_X_SIZE 256
#endif
#define MIN(a,b) ((a) < (b) ? (a) : (b))

MPI_Comm cart_comm;
int up_ngb, down_ngb;// z小的为down
MPI_Datatype up_send_subarray, up_recv_subarray;
MPI_Datatype down_send_subarray, down_recv_subarray;
MPI_Status status;

// #define useINDEX

// 创建分布式网格：可以根据7点或27点类型做不同的划分
void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    // 一维划分 沿z轴切
    if (grid_info->p_id == 0)
        printf(" 1D partition: num_proc_z: %d\n", grid_info->p_num);
    grid_info->local_size_x = grid_info->global_size_x;
    grid_info->local_size_y = grid_info->global_size_y;
    if(grid_info->global_size_z % grid_info->p_num != 0) {
        if (grid_info->p_id == 0)
            printf(" Error: %d cannot divide %d!\n", grid_info->global_size_z, grid_info->p_num);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    grid_info->local_size_z = grid_info->global_size_z / grid_info->p_num;

    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = grid_info->local_size_z * grid_info->p_id;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;

    // printf("pid: %d    global: %d %d %d    local : %d %d %d    offset: %d %d %d\n", grid_info->p_id,\
    //     grid_info->global_size_z, grid_info->global_size_y, grid_info->global_size_x,\
    //     grid_info->local_size_z, grid_info->local_size_y, grid_info->local_size_x,\
    //     grid_info->offset_z, grid_info->offset_y, grid_info->offset_x);

    // 创建通信的拓扑
    int dims[1] = {grid_info->p_num};
    int periods = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, &periods, 0, &cart_comm);
    MPI_Cart_shift(cart_comm, 0, 1, &down_ngb, &up_ngb);
    // printf("pid: %d    down: %d    up: %d\n", grid_info->p_id, down_ngb, up_ngb);

    // 创建subarray
    int size[3] = { grid_info->local_size_z + 2*grid_info->halo_size_z,\
                    grid_info->local_size_y + 2*grid_info->halo_size_y,\
                    grid_info->local_size_x + 2*grid_info->halo_size_x};
    int subsize[3] = {  grid_info->halo_size_z, \
                        grid_info->local_size_y + 2*grid_info->halo_size_y,\
                        grid_info->local_size_x + 2*grid_info->halo_size_x}; 
    int start[3];
    // send to down_ngb
    start[0] = grid_info->halo_size_z; start[1] = 0; start[2] = 0;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, DATA_TYPE, &down_send_subarray);
    MPI_Type_commit(&down_send_subarray);
    // printf("pid: %d    down_send start: %d %d %d\n", grid_info->p_id, start[0], start[1], start[2]);

    // recv from down_ngb
    start[0] = 0; start[1] = 0; start[2] = 0;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, DATA_TYPE, &down_recv_subarray);
    MPI_Type_commit(&down_recv_subarray);
    // printf("pid: %d    down_recv start: %d %d %d\n", grid_info->p_id, start[0], start[1], start[2]);

    // send to up_ngb
    start[0] = grid_info->local_size_z; start[1] = 0; start[2] = 0;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, DATA_TYPE, &up_send_subarray);
    MPI_Type_commit(&up_send_subarray);
    // printf("pid: %d      up_send start: %d %d %d\n", grid_info->p_id, start[0], start[1], start[2]);

    // recv from up_ngb
    start[0] = grid_info->local_size_z + grid_info->halo_size_z; start[1] = 0; start[2] = 0;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, DATA_TYPE, &up_recv_subarray);
    MPI_Type_commit(&up_recv_subarray);
    // printf("pid: %d      up_recv start: %d %d %d\n", grid_info->p_id, start[0], start[1], start[2]);
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {
    for (int i = 1; i <= 8; i++) {
        if (send_subarray[i] != MPI_DATATYPE_NULL) 
            MPI_Type_free(&send_subarray[i]);
        if (recv_subarray[i] != MPI_DATATYPE_NULL)
            MPI_Type_free(&recv_subarray[i]);
    }
}

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt, double * calc_time, double * comm_time) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;
    double t_last, t_curr;

    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];

        // 通信同步(要让a0的边界是对的值！)
        t_curr = MPI_Wtime();
        MPI_Sendrecv(a0, 1, down_send_subarray, down_ngb, grid_info->p_id ^ down_ngb,\
                     a0, 1,   up_recv_subarray,   up_ngb, grid_info->p_id ^ up_ngb  , cart_comm, &status);
        MPI_Sendrecv(a0, 1,   up_send_subarray,   up_ngb, grid_info->p_id ^ up_ngb  ,\
                     a0, 1, down_recv_subarray, down_ngb, grid_info->p_id ^ down_ngb, cart_comm, &status);
        // MPI_Sendrecv(a0, 1, down_send_subarray, down_ngb, 10,\
        //              a0, 1,   up_recv_subarray,   up_ngb, 10, cart_comm, &status);
        // MPI_Sendrecv(a0, 1,   up_send_subarray,   up_ngb, 11,\
        //              a0, 1, down_recv_subarray, down_ngb, 11, cart_comm, &status);
        t_last = t_curr;
        t_curr = MPI_Wtime();
        *comm_time += t_curr - t_last;
        
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
        t_last = t_curr;
        t_curr = MPI_Wtime();
        *calc_time += t_curr - t_last;
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
    double t_last, t_curr;

    for(int t = 0; t < nt; ++t) {
        cptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];

        // 通信同步(要让a0的边界是对的值！)
        t_curr = MPI_Wtime();
        MPI_Sendrecv(a0, 1, down_send_subarray, down_ngb, grid_info->p_id ^ down_ngb,\
                     a0, 1,   up_recv_subarray,   up_ngb, grid_info->p_id ^ up_ngb  , cart_comm, &status);
        MPI_Sendrecv(a0, 1,   up_send_subarray,   up_ngb, grid_info->p_id ^ up_ngb  ,\
                     a0, 1, down_recv_subarray, down_ngb, grid_info->p_id ^ down_ngb, cart_comm, &status);
        // MPI_Sendrecv(a0, 1, down_send_subarray, down_ngb, 10,\
        //              a0, 1,   up_recv_subarray,   up_ngb, 10, cart_comm, &status);
        // MPI_Sendrecv(a0, 1,   up_send_subarray,   up_ngb, 11,\
        //              a0, 1, down_recv_subarray, down_ngb, 11, cart_comm, &status);
        t_last = t_curr;
        t_curr = MPI_Wtime();
        *comm_time += t_curr - t_last;

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
        t_last = t_curr;
        t_curr = MPI_Wtime();
        *calc_time += t_curr - t_last;
    }
    return buffer[nt % 2];
}
