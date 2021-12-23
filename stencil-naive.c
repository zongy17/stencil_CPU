#include "common.h"

const char* version_name = "A naive base-line";

#ifndef SET_Y_SIZE
#define SET_Y_SIZE 8
#endif
#ifndef SET_X_SIZE
#define SET_X_SIZE 256
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))

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

ptr_t stencil_7(ptr_t grid, ptr_t aux, const dist_grid_info_t *grid_info, int nt, double * calc_time, double * comm_time) {
    ptr_t buffer[2] = {grid, aux};
    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    for(int t = 0; t < nt; ++t) {
        cptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];
        
        for (int JJ = y_start; JJ < y_end; JJ += SET_Y_SIZE) {
            int REAL_Y_SIZE = MIN(SET_Y_SIZE, y_end - JJ);
            for (int II = x_start; II < x_end; II += SET_X_SIZE){
                int REAL_X_SIZE = MIN(SET_X_SIZE, x_end - II);

                ptr_t a1_local = a1 + z_start*ldx*ldy + JJ*ldx + II;
                ptr_t a0_local_Z = a0 + z_start*ldx*ldy + JJ*ldx + II;
                ptr_t a0_local_P = a0 + (z_start+1)*ldx*ldy + JJ*ldx + II;
                ptr_t a0_local_N = a0 - (z_start-1)*ldx*ldy + JJ*ldx + II;

                for(int z = z_start; z < z_end; ++z) {
                    for (int y = 0; y < REAL_Y_SIZE; y++){
                        #pragma unroll
                        for (int x = 0; x < REAL_X_SIZE; x++){
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
    for(int t = 0; t < nt; ++t) {
        cptr_t restrict a0 = buffer[t % 2];
        ptr_t restrict a1 = buffer[(t + 1) % 2];

        for (int JJ = y_start; JJ < y_end; JJ += SET_Y_SIZE) {
            int REAL_Y_SIZE = MIN(SET_Y_SIZE, y_end - JJ);
            for (int II = x_start; II < x_end; II += SET_X_SIZE){
                int REAL_X_SIZE = MIN(SET_X_SIZE, x_end - II);

                ptr_t a1_local = a1 + z_start*ldx*ldy + JJ*ldx + II;
                cptr_t a0_local_Z = a0 + z_start*ldx*ldy + JJ*ldx + II;
                cptr_t a0_local_P = a0 + (z_start+1)*ldx*ldy + JJ*ldx + II;
                cptr_t a0_local_N = a0 - (z_start-1)*ldx*ldy + JJ*ldx + II;     

                for(int z = z_start; z < z_end; ++z) {
                    for (int y = 0; y < REAL_Y_SIZE; y++){
                        #pragma unroll
                        for (int x = 0; x < REAL_X_SIZE; x++){
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
                }
            }
        }
    }
    return buffer[nt % 2];
}
