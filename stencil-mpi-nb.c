#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
const char* version_name = "A mpi comp-comm overlapped";
#include "common.h"

#ifndef SET_Y_SIZE
#define SET_Y_SIZE 8
#endif
#ifndef SET_X_SIZE
#define SET_X_SIZE 256
#endif
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

MPI_Comm cart_comm;
int cart_ids[2];
//     6      2      5
// y(1)
//  ^  3      0      1
//  |
//  |  7      4      8
//  ------> z (0)
int oppo_idx[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
int ngbs[9];
MPI_Datatype send_subarray[9], recv_subarray[9];

static int ih_z_beg[9], ih_z_end[9], ih_y_beg[9], ih_y_end[9];

// 创建分布式网格：可以根据7点或27点类型做不同的划分
void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    // 二维划分 沿zy轴切
    int sqr_root = 1;
    int num_proc_y, num_proc_z;
    while (sqr_root*sqr_root < grid_info->p_num) sqr_root++;

    if (sqr_root*sqr_root == grid_info->p_num) {
        num_proc_z = num_proc_y = sqr_root;
    } else {
        int tmp = 1;
        while (grid_info->p_num%tmp==0 && grid_info->p_num/tmp>tmp) tmp *= 2;
        num_proc_z = grid_info->p_num / tmp;//跳出while时tmp>sqrt(p_num)
        num_proc_y = tmp;
        // num_proc_z = tmp;
        // num_proc_y = grid_info->p_num / tmp;
    }
    if (grid_info->p_id == 0)
        printf(" 2D partition: num_proc_y: %d, num_proc_z:%d\n", num_proc_y, num_proc_z);
    // x轴不切
    grid_info->local_size_x = grid_info->global_size_x;
    // y轴
    if (grid_info->global_size_y % num_proc_y != 0) {
        if (grid_info->p_id == 0)
            printf(" Error: %d cannot divide %d!\n", grid_info->global_size_y, num_proc_y);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    grid_info->local_size_y = grid_info->global_size_y / num_proc_y;
    // z轴
    if (grid_info->global_size_z % num_proc_z != 0) {
        if (grid_info->p_id == 0)
            printf(" Error: %d cannot divide %d!\n", grid_info->global_size_z, num_proc_z);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    grid_info->local_size_z = grid_info->global_size_z / num_proc_z;

    // printf("pid: %d    global: %d %d %d    local : %d %d %d    offset: %d %d %d\n", grid_info->p_id,\
    //     grid_info->global_size_z, grid_info->global_size_y, grid_info->global_size_x,\
    //     grid_info->local_size_z, grid_info->local_size_y, grid_info->local_size_x,\
    //     grid_info->offset_z, grid_info->offset_y, grid_info->offset_x);

    // 创建通信的拓扑
    ngbs[0] = grid_info->p_id;
    int dims[2] = {num_proc_z, num_proc_y};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, &periods, 0, &cart_comm);
    MPI_Cart_shift(cart_comm, 0, 1, &ngbs[3], &ngbs[1]);
    MPI_Cart_shift(cart_comm, 1, 1, &ngbs[4], &ngbs[2]);

    int dist_y = 1;
    if (ngbs[1]==MPI_PROC_NULL || ngbs[2]==MPI_PROC_NULL) ngbs[5] = MPI_PROC_NULL;
    else ngbs[5] = ngbs[1] + dist_y; 
    if (ngbs[1]==MPI_PROC_NULL || ngbs[4]==MPI_PROC_NULL) ngbs[8] = MPI_PROC_NULL;
    else ngbs[8] = ngbs[1] - dist_y; 
    if (ngbs[2]==MPI_PROC_NULL || ngbs[3]==MPI_PROC_NULL) ngbs[6] = MPI_PROC_NULL;
    else ngbs[6] = ngbs[3] + dist_y;
    if (ngbs[3]==MPI_PROC_NULL || ngbs[4]==MPI_PROC_NULL) ngbs[7] = MPI_PROC_NULL;
    else ngbs[7] = ngbs[3] - dist_y; 

    MPI_Cart_coords(cart_comm, grid_info->p_id, 2, &cart_ids);

    grid_info->offset_x = 0;
    grid_info->offset_y = grid_info->local_size_y * cart_ids[1];
    grid_info->offset_z = grid_info->local_size_z * cart_ids[0];
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;

    // printf("pid: %d    cart_id[0]: %d    cart_id[1]: %d\n %d %d %d %d %d %d %d %d %d\n offset_y:%d    offset_z:%d\n", grid_info->p_id, cart_ids[0], cart_ids[1],\
    //     ngbs[0],ngbs[1],ngbs[2],ngbs[3],ngbs[4],ngbs[5],ngbs[6],ngbs[7],ngbs[8], grid_info->offset_y, grid_info->offset_z);


    // 创建subarray
    int size[3] = { grid_info->local_size_z + 2*grid_info->halo_size_z,\
                    grid_info->local_size_y + 2*grid_info->halo_size_y,\
                    grid_info->local_size_x + 2*grid_info->halo_size_x};
    int subsize[3], send_start[3], recv_start[3];
    for (int i = 1; i <= 8; i++) {
        switch (i) {
        case 1:
            subsize[0] = grid_info->halo_size_z;
            subsize[1] = grid_info->local_size_y;// 注意是local_size 不带halo_width
            break;
        case 2:
            subsize[0] = grid_info->local_size_z;
            subsize[1] = grid_info->halo_size_y;
            break;
        case 3:
            subsize[0] = grid_info->halo_size_z;
            subsize[1] = grid_info->local_size_y;
            break;
        case 4:
            subsize[0] = grid_info->local_size_z;
            subsize[1] = grid_info->halo_size_y;
            break;
        default:// 5,6,7,8
            subsize[0] = grid_info->halo_size_z;
            subsize[1] = grid_info->halo_size_y;
            break;
        }
        subsize[2] = grid_info->local_size_x + 2*grid_info->halo_size_x;

        switch (i) {
        case 1:// up_z
            send_start[0] = grid_info->local_size_z;                          send_start[1] = grid_info->halo_size_y; send_start[2] = 0;
            recv_start[0] = grid_info->local_size_z + grid_info->halo_size_z; recv_start[1] = grid_info->halo_size_y; recv_start[2] = 0;
            break;
        case 3:// down_z
            send_start[0] = grid_info->halo_size_z; send_start[1] = grid_info->halo_size_y; send_start[2] = 0;
            recv_start[0] = 0;                      recv_start[1] = grid_info->halo_size_y; recv_start[2] = 0;
            break;
        case 2:// up_y
            send_start[0] = grid_info->halo_size_z; send_start[1] = grid_info->local_size_y;                          send_start[2] = 0;
            recv_start[0] = grid_info->halo_size_z; recv_start[1] = grid_info->local_size_y + grid_info->halo_size_y; recv_start[2] = 0;
            break;
        case 4:// down_y
            send_start[0] = grid_info->halo_size_z; send_start[1] = grid_info->halo_size_y; send_start[2] = 0;
            recv_start[0] = grid_info->halo_size_z; recv_start[1] = 0;                      recv_start[2] = 0;
            break;
        case 5:// up_z_up_y
            send_start[0] = grid_info->local_size_z;                          send_start[1] = grid_info->local_size_y;                          send_start[2] = 0;
            recv_start[0] = grid_info->local_size_z + grid_info->halo_size_z; recv_start[1] = grid_info->local_size_y + grid_info->halo_size_y; recv_start[2] = 0;
            break;
        case 6:// down_z_up_y
            send_start[0] = grid_info->halo_size_z; send_start[1] = grid_info->local_size_y;                          send_start[2] = 0;
            recv_start[0] = 0;                      recv_start[1] = grid_info->local_size_y + grid_info->halo_size_y; recv_start[2] = 0;
            break;
        case 7:// down_z_down_y
            send_start[0] = grid_info->halo_size_z; send_start[1] = grid_info->halo_size_y; send_start[2] = 0;
            recv_start[0] = 0;                      recv_start[1] = 0;                      recv_start[2] = 0;
            break;
        case 8:// up_z_down_y
            send_start[0] = grid_info->local_size_z;                          send_start[1] = grid_info->halo_size_y; send_start[2] = 0;
            recv_start[0] = grid_info->local_size_z + grid_info->halo_size_z; recv_start[1] = 0;                      recv_start[2] = 0;
            break;
        default:
            break;
        }
        MPI_Type_create_subarray(3, size, subsize, send_start, MPI_ORDER_C, DATA_TYPE, &send_subarray[i]);
        MPI_Type_commit(&send_subarray[i]);
        MPI_Type_create_subarray(3, size, subsize, recv_start, MPI_ORDER_C, DATA_TYPE, &recv_subarray[i]);
        MPI_Type_commit(&recv_subarray[i]);
    }

    // 记录计算通信重叠部分的内halo区(注意这是内halo！)
    for (int dir = 1; dir <= 4; dir++) {
        //
        //   ----------------------------
        //   |         2          |     |
        //   |--------------------|     |
        //   |     |              |     |
        //   |     |              |  1  |
        //   |     |              |     |
        // y |  3  |              |     |
        // ^ |     |              |     |
        // | |     |--------------------|
        // | |     |          4         |
        // | ----------------------------
        // O----> z
        switch (dir) {
        case 1:
            ih_z_beg[dir] = grid_info->local_size_z;
            ih_z_end[dir] = grid_info->halo_size_z + grid_info->local_size_z;
            ih_y_beg[dir] = grid_info->halo_size_y * 2;
            ih_y_end[dir] = grid_info->halo_size_y + grid_info->local_size_y;
            break;
        case 2:
            ih_z_beg[dir] = grid_info->halo_size_z;
            ih_z_end[dir] = grid_info->local_size_z;
            ih_y_beg[dir] = grid_info->local_size_y;
            ih_y_end[dir] = ih_y_beg[dir] + grid_info->halo_size_y;
            break;
        case 3:
            ih_z_beg[dir] = grid_info->halo_size_z;
            ih_z_end[dir] = grid_info->halo_size_z * 2;
            ih_y_beg[dir] = grid_info->halo_size_y;
            ih_y_end[dir] = grid_info->local_size_y;
            break;
        case 4:
            ih_z_beg[dir] = grid_info->halo_size_z * 2;
            ih_z_end[dir] = grid_info->halo_size_z + grid_info->local_size_z;
            ih_y_beg[dir] = grid_info->halo_size_y;
            ih_y_end[dir] = grid_info->halo_size_y * 2;
            break;
        default:
            break;
        }
        // printf("pid %d, dir %d, %d %d %d %d\n", grid_info->p_id, dir, ih_z_beg[dir], ih_z_end[dir], ih_y_beg[dir], ih_y_end[dir]);
    }
    
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
    MPI_Request req_send[4], req_recv[4];
    MPI_Status status[4];

    // for boundary conditions
    for (int i = 1; i <= 4; i++) {// 7点stencil只需要通信4个
        int oppo = oppo_idx[i];
        MPI_Sendrecv(buffer[0], 1, send_subarray[i]   , ngbs[i]   , grid_info->p_id ^ ngbs[i]   ,\
                     buffer[0], 1, recv_subarray[oppo], ngbs[oppo], grid_info->p_id ^ ngbs[oppo], cart_comm, &status[i-1]);
    }

    // y_start += grid_info->halo_size_y; y_end -= grid_info->halo_size_y;
    // z_start += grid_info->halo_size_z; z_end -= grid_info->halo_size_z;

    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];

        // 先计算内halo区
        for (int dir = 1; dir <= 4; dir++) {
            ptr_t a1_local = a1 + ih_z_beg[dir]*ldx*ldy + ih_y_beg[dir]*ldx;
            cptr_t a0_local_Z = a0 + (a1_local - a1);
            cptr_t a0_local_P = a0_local_Z + ldy*ldx;
            cptr_t a0_local_N = a0_local_Z - ldy*ldx;
            int yy_end = ih_y_end[dir]-ih_y_beg[dir];
            for (int z = ih_z_beg[dir]; z < ih_z_end[dir]; z++){
                for (int y = 0; y < yy_end; y++)
                    for (int x = x_start; x < x_end; x++)
                        a1_local[y*ldx+x] \
                            = ALPHA_ZZZ * a0_local_Z[y*ldx+x]\
                            + ALPHA_NZZ * a0_local_Z[y*ldx+x-1] \
                            + ALPHA_PZZ * a0_local_Z[y*ldx+x+1] \
                            + ALPHA_ZNZ * a0_local_Z[(y-1)*ldx+x] \
                            + ALPHA_ZPZ * a0_local_Z[(y+1)*ldx+x] \
                            + ALPHA_ZZN * a0_local_N[y*ldx+x] \
                            + ALPHA_ZZP * a0_local_P[y*ldx+x];
                a1_local = a1_local + ldx*ldy;
                a0_local_N = a0_local_Z;
                a0_local_Z = a0_local_P;
                a0_local_P = a0_local_P + ldx*ldy;
            }
        }
        // 隐藏非阻塞通信
        for (int dir = 1; dir <= 4; dir++) {
            int oppo = oppo_idx[dir];
            MPI_Isend(a1, 1, send_subarray[dir],  ngbs[dir],  grid_info->p_id ^ ngbs[dir],  cart_comm, &req_send[dir-1]);
            MPI_Irecv(a1, 1, recv_subarray[oppo], ngbs[oppo], grid_info->p_id ^ ngbs[oppo], cart_comm, &req_recv[oppo-1]);
        }

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

        MPI_Waitall(4, req_recv, status);
        MPI_Waitall(4, req_send, status);
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
    MPI_Request req_send[8], req_recv[8];
    MPI_Status status[8];

    // for boundary conditions
    for (int i = 1; i <= 8; i++) {// 27点stencil需要通信8个
        int oppo = oppo_idx[i];
        MPI_Sendrecv(buffer[0], 1, send_subarray[i]   , ngbs[i]   , grid_info->p_id ^ ngbs[i]   ,\
                     buffer[0], 1, recv_subarray[oppo], ngbs[oppo], grid_info->p_id ^ ngbs[oppo], cart_comm, &status[i-1]);
    }

    // 刨去外周一圈，导致对不齐（或者别的原因？），速度骤降，比重新再多算一遍外周还慢得多！！！
    // y_start += grid_info->halo_size_y; y_end -= grid_info->halo_size_y;
    // z_start += grid_info->halo_size_z; z_end -= grid_info->halo_size_z;

    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = buffer[t % 2];
        ptr_t a1 = buffer[(t + 1) % 2];

        // 先计算内halo区
        for (int dir = 1; dir <= 4; dir++) {
            ptr_t a1_local = a1 + ih_z_beg[dir]*ldx*ldy + ih_y_beg[dir]*ldx;
            cptr_t a0_local_Z = a0 + (a1_local - a1);
            cptr_t a0_local_P = a0_local_Z + ldy*ldx;
            cptr_t a0_local_N = a0_local_Z - ldy*ldx;
            int yy_end = ih_y_end[dir]-ih_y_beg[dir];

            for (int z = ih_z_beg[dir]; z < ih_z_end[dir]; z++){
                for (int y = 0; y < yy_end; y++)
                    for (int x = x_start; x < x_end; x++)
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
                a1_local = a1_local + ldx*ldy;
                a0_local_N = a0_local_Z;
                a0_local_Z = a0_local_P;
                a0_local_P = a0_local_P + ldx*ldy;
            }
        }
        // 隐藏非阻塞通信
        for (int dir = 1; dir <= 8; dir++) {
            int oppo = oppo_idx[dir];
            MPI_Isend(a1, 1, send_subarray[dir],  ngbs[dir],  grid_info->p_id ^ ngbs[dir],  cart_comm, &req_send[dir-1]);
            MPI_Irecv(a1, 1, recv_subarray[oppo], ngbs[oppo], grid_info->p_id ^ ngbs[oppo], cart_comm, &req_recv[oppo-1]);
        }
        
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

        MPI_Waitall(8, req_recv, status);
        MPI_Waitall(8, req_send, status);
    }
    return buffer[nt % 2];
}
