
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define dtype float

#define SIGMOID(inp) (1.0f / (1 + half_exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)
#define BLOCK_SIZE 32
#define WPT 8
#define RBLOCK_SIZE (BLOCK_SIZE/WPT)

__kernel __attribute__((reqd_work_group_size(RBLOCK_SIZE, RBLOCK_SIZE, 1)))
void GEMM(
      __global dtype* restrict A,
      __global dtype* restrict B,
      __global dtype* restrict C,
      __const int M,
      __const int N,
      __const int P,
      __const int M_,
      __const int N_,
      __const int P_)
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int m = BLOCK_SIZE*get_group_id(0) + row;
    const int p = BLOCK_SIZE*get_group_id(1) + col;
    __local float A_local[BLOCK_SIZE][BLOCK_SIZE];
    __local float B_local[BLOCK_SIZE][BLOCK_SIZE];

    float Areg;
    float Breg[WPT];
    float acc[WPT][WPT];
    for(int wm=0; wm<WPT; wm++){
        for(int wn=0; wn<WPT; wn++){
            acc[wm][wn] = 0.0f;
        }
    }
    const int numTiles = N_/BLOCK_SIZE;
    #pragma unroll
    for (int t=0; t<numTiles; t++) {
        for (int wm=0; wm<WPT; wm++){
            for (int wn=0; wn<WPT; wn++){
                const int r = BLOCK_SIZE*t + row;
                const int c = BLOCK_SIZE*t + col;
                if(((m + wm*RBLOCK_SIZE) < M) && ((c + wn*RBLOCK_SIZE) < N)){
                    A_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = A[(m + wm*RBLOCK_SIZE)*N + (c + wn*RBLOCK_SIZE)];
                } else {
                    A_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = 0.0;
                }

                if(((p + wn*RBLOCK_SIZE) < P) && ((r + wm*RBLOCK_SIZE) < N)){
                    B_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = B[(r + wm*RBLOCK_SIZE)*P + (p + wn*RBLOCK_SIZE)];
                } else {
                    B_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = 0.0;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll BLOCK_SIZE
        for (int k=0; k<BLOCK_SIZE; k++){
            for (int wn=0; wn<WPT; wn++){
                Breg[wn] = B_local[k][col + wn*RBLOCK_SIZE];
            }
            for (int wm=0; wm<WPT; wm++){
                Areg = A_local[row + wm*RBLOCK_SIZE][k];
                for (int wn=0; wn<WPT; wn++){
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int wm=0; wm<WPT; wm++){
        for (int wn=0; wn<WPT; wn++){
            if(((m + wm*RBLOCK_SIZE) < M) && ((p + wn*RBLOCK_SIZE) < P)){
                C[(m + wm*RBLOCK_SIZE)*P + (p + wn*RBLOCK_SIZE)] = acc[wm][wn];
            }

        }
    }
}

__attribute__((num_simd_work_items(8)))
__attribute__((uses_global_work_offset(0)))
__attribute__((reqd_work_group_size(RBLOCK_SIZE, RBLOCK_SIZE, 1)))
__kernel void linear(
      __global dtype* restrict A,
      __global dtype* restrict B,
      __global dtype* restrict bias,
      __global dtype* restrict C,
      __const int M,
      __const int N,
      __const int P,
      __const int M_,
      __const int N_,
      __const int P_,
      __const int activation){

        const int row = get_local_id(0);
        const int col = get_local_id(1);
        const int m = BLOCK_SIZE*get_group_id(0) + row;
        const int p = BLOCK_SIZE*get_group_id(1) + col;
        __local float A_local[BLOCK_SIZE][BLOCK_SIZE];
        __local float B_local[BLOCK_SIZE][BLOCK_SIZE];

        float Areg;
        float Breg[WPT];
        float acc[WPT][WPT];
        for(int wm=0; wm<WPT; wm++){
            for(int wn=0; wn<WPT; wn++){
                acc[wm][wn] = 0.0f;
            }
        }
        const int numTiles = N_/BLOCK_SIZE;
        #pragma unroll
        for (int t=0; t<numTiles; t++) {
            for (int wm=0; wm<WPT; wm++){
                for (int wn=0; wn<WPT; wn++){
                    const int r = BLOCK_SIZE*t + row;
                    const int c = BLOCK_SIZE*t + col;
                    if(((m + wm*RBLOCK_SIZE) < M) && ((c + wn*RBLOCK_SIZE) < N)){
                        A_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = A[(m + wm*RBLOCK_SIZE)*N + (c + wn*RBLOCK_SIZE)];
                    } else {
                        A_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = 0.0;
                    }

                    if(((p + wn*RBLOCK_SIZE) < P) && ((r + wm*RBLOCK_SIZE) < N)){
                        B_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = B[(r + wm*RBLOCK_SIZE)*P + (p + wn*RBLOCK_SIZE)];
                    } else {
                        B_local[row + wm*RBLOCK_SIZE][col + wn*RBLOCK_SIZE] = 0.0;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll BLOCK_SIZE
            for (int k=0; k<BLOCK_SIZE; k++){
                for (int wn=0; wn<WPT; wn++){
                    Breg[wn] = B_local[k][col + wn*RBLOCK_SIZE];
                }
                for (int wm=0; wm<WPT; wm++){
                    Areg = A_local[row + wm*RBLOCK_SIZE][k];
                    for (int wn=0; wn<WPT; wn++){
                        acc[wm][wn] += Areg * Breg[wn];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int wm=0; wm<WPT; wm++){
            for (int wn=0; wn<WPT; wn++){
                if(((m + wm*RBLOCK_SIZE) < M) && ((p + wn*RBLOCK_SIZE) < P)){
                    if(((m + wm*RBLOCK_SIZE) < M) && ((p + wn*RBLOCK_SIZE) < P)){
                        C[(m + wm*RBLOCK_SIZE)*P + (p + wn*RBLOCK_SIZE)] = acc[wm][wn];
                        if(activation == 1){
                          C[(m + wm*RBLOCK_SIZE) * P + (p + wn*RBLOCK_SIZE)] = RELU( C[(m + wm*RBLOCK_SIZE) * P + (p + wn*RBLOCK_SIZE)] + bias[(p + wn*RBLOCK_SIZE)] );
                        } else if(activation == 2){
                          C[(m + wm*RBLOCK_SIZE) * P + (p + wn*RBLOCK_SIZE)] = SIGMOID( C[(m + wm*RBLOCK_SIZE) * P + (p + wn*RBLOCK_SIZE)] + bias[(p + wn*RBLOCK_SIZE)] );
                        } else {
                          C[(m + wm*RBLOCK_SIZE) * P + (p + wn*RBLOCK_SIZE)] += bias[(p + wn*RBLOCK_SIZE)];
                        }
                    }

                }

            }
        }
}

__kernel
__attribute__((uses_global_work_offset(0)))
void interaction_cat(__global dtype* restrict sender,
                              __global dtype* restrict receiver,
                              __global dtype* restrict ri,
                              __global dtype* restrict out,
                              int m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    if(x_idx < 3){
        out[x] = sender[x];
    } else if(x_idx < 6){
        out[x] = receiver[x - (3 * m)];
    } else {
        out[x] = ri[x - (6 * m)];
    }
}

__kernel
__attribute__((uses_global_work_offset(0)))
void transpose(__global dtype* restrict a_t,
                        __global dtype* restrict a,
                        int m, int n)
{
    // global space:  a.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    a_t[global_idx(y_idx, x_idx, m)] = a[global_idx(x_idx, y_idx, n)];
}


__attribute__((uses_global_work_offset(0)))
__kernel void aggregate_cat(__global dtype* restrict obj_t,
                            __global dtype* restrict effect_receiver,
                            __global dtype* restrict out,
                            int m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    if(x_idx < 3){
        out[x] = obj_t[x];
    } else {
        out[x] = effect_receiver[x - (3 * m)];
    }
}
