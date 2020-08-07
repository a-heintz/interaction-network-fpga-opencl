#define SIGMOID(inp) (1.0f / (1 + native_exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)

__kernel void pad(__global float* restrict a_unpad,
                  __global float* restrict a_pad,
                  __const ushort M,
                  __const ushort N,
                  __const ushort M_pad,
                  __const ushort N_pad){
  // global size: M, N
  // a_pad must be a zeros scoped_array
  int i = get_global_id(0);
  int j = get_global_id(1);

  a_pad[i*N_pad + j] = a_unpad[i*N + j];
}

__kernel void matrixMul(
      __global float* restrict A,
      __global float* restrict B,
      __global float* restrict C,
      __const ushort M,
      __const ushort K,
      __const ushort N){
  int m = get_global_id(0);
  int n = get_global_id(1);
  float acc = 0.0f;
  #pragma unroll 4
  for(int k = 0; k < K; ++k){
      acc += A[m*K + k] * B[k*N + n];
  }
  C[m*N + n] = acc;
}

__attribute__((num_simd_work_items(8)))
__attribute__((uses_global_work_offset(0)))
__kernel void linear(
      __global float* restrict A,
      __global float* restrict B,
      __global float* restrict bias,
      __global float* restrict C,
      __const ushort M,
      __const ushort N,
      __const ushort P,
      __const int BLOCK_SIZE,
      __const int activation){

  matrixMul(A, B, C, M, N, P);
  int m = get_global_id(0);
  int p = get_global_id(1);

  if(activation == 1){
    C[m * P + p] = RELU(C[m * P + p] + bias[p]);
  } else if(activation == 2){
    C[m * P + p] = SIGMOID(C[m * P + p] + bias[p]);
  } else {
    C[m * P + p] += bias[p];
  }
}

__attribute__((uses_global_work_offset(0)))
__kernel void transpose(__global float* restrict a_t,
                        __global float* restrict a,
                        ushort n, ushort m)
{
    // global space:  a.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    a_t[global_idx(y_idx, x_idx, n)] = a[global_idx(x_idx, y_idx, m)];
}

__attribute__((uses_global_work_offset(0)))
__kernel void interaction_cat(__global float* restrict sender,
                              __global float* restrict receiver,
                              __global float* restrict ri,
                              __global float* restrict out,
                              ushort m)
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

__attribute__((uses_global_work_offset(0)))
__kernel void aggregate_cat(__global float* restrict obj_t,
                            __global float* restrict effect_receiver,
                            __global float* restrict out,
                            ushort m)
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
