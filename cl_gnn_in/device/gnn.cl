#define SIGMOID(inp) (1.0f / (1 + native_exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)

#define A_local(i, j, BLOCK) A_local[j + i * BLOCK]
#define B_local(i, j, BLOCK) B_local[j + i * BLOCK]


__attribute__((uses_global_work_offset(0)))
__kernel void add_bias(__global float *inp,
                       __global float *bias,
                       __global float *out,
                       ushort m)
{
    // global space:  inp.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    out[x] = inp[x] + bias[y_idx];
}

__kernel void matmul( __global float* A,
                      __global float* B,
                      __global float* C,
                      __local float* A_local,
                      __local float* B_local,
                      ushort A_width,
                      ushort B_width,
                      int BLOCK_SIZE){

    // Block index
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    // Local ID index (offset within a block)
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    // Compute loop bounds
    int a_start = A_width * BLOCK_SIZE * block_y;
    int a_end   = a_start + A_width - 1;
    int b_start = BLOCK_SIZE * block_x;

    float running_sum = 0.0f;

    for (int a = a_start, b = b_start; a <= a_end; a += BLOCK_SIZE, b += (BLOCK_SIZE * B_width))
    {

      A_local(local_y, local_x, BLOCK_SIZE) = A[a + A_width * local_y + local_x];
      B_local(local_x, local_y, BLOCK_SIZE) = B[b + B_width * local_y + local_x];

      barrier(CLK_LOCAL_MEM_FENCE);

      #pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k)
      {
          running_sum += A_local(local_y, k, 1) * B_local(local_x, k, 1);
      }

      barrier(CLK_LOCAL_MEM_FENCE);

    }
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}
/*
__kernel void matrixMul(
      __global float* A,
      __global float* B,
      __global float* C,
      __const ushort M,
      __const ushort K,
      __const ushort N){
  int m = get_global_id(0);
  int n = get_global_id(1);
  float acc = 0.0f;
  for(int k = 0; k < K; ++k){
    acc += A[m*K + k] * B[k*N + n];
  }
  C[m*N + n] = acc;
}
*/
__kernel void matrixMul(
      __global float* A,
      __global float* B,
      __global float* C,
      __const ushort M,
      __const ushort K,
      __const ushort N){
  int m = get_global_id(0);
  int n = get_global_id(1);
  float acc = 0.0f;
  #pragma unroll
  for(int k = 0; k < K; ++k){
    acc += A[m*K + k] * B[k*N + n];
  }
  C[m*N + n] = acc;
}

__kernel void linear(
      __global float* A,
      __global float* B,
      __global float* bias,
      __global float* C,
      __local float* A_local,
      __local float* B_local,
      __const ushort A_width,
      __const ushort B_width,
      __const int BLOCK_SIZE){

  int m = get_global_id(0);
  int n = get_global_id(1);
  matmul(A, B, C, A_local, B_local, A_width, B_width, BLOCK_SIZE);
  C[n * get_global_size(0) + m] += bias[m];
}

__kernel void linear_relu(
      __global float* A,
      __global float* B,
      __global float* bias,
      __global float* C,
      __local float* A_local,
      __local float* B_local,
      __const ushort A_width,
      __const ushort B_width,
      __const int BLOCK_SIZE){

  int m = get_global_id(0);
  int n = get_global_id(1);

  matmul(A, B, C, A_local, B_local, A_width, B_width, BLOCK_SIZE);
  C[n * get_global_size(0) + m] = RELU(C[n * get_global_size(0) + m] + bias[m]);
}

__kernel void linear_sigmoid(
      __global float* A,
      __global float* B,
      __global float* bias,
      __global float* C,
      __local float* A_local,
      __local float* B_local,
      __const ushort A_width,
      __const ushort B_width,
      __const int BLOCK_SIZE){

  int m = get_global_id(0);
  int n = get_global_id(1);
  matmul(A, B, C, A_local, B_local, A_width, B_width, BLOCK_SIZE);
  C[n * get_global_size(0) + m] = SIGMOID(C[n * get_global_size(0) + m] + bias[m]);
}

__attribute__((uses_global_work_offset(0)))
__kernel void transpose(__global float *a_t, __global float *a, ushort n, ushort m)
{
    // global space:  a.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    a_t[global_idx(y_idx, x_idx, n)] = a[global_idx(x_idx, y_idx, m)];
}

__attribute__((uses_global_work_offset(0)))
__kernel void relu(__global float *inp, __global float *out, ushort m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    out[x] = RELU(inp[x]);
}

__attribute__((uses_global_work_offset(0)))
__kernel void sigmoid(__global float *inp, __global float *out, ushort m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    out[x] = SIGMOID(inp[x]);
}

__attribute__((uses_global_work_offset(0)))
__kernel void interaction_cat(__global float *sender, __global float *receiver, __global float *ri, __global float *out, ushort m)
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
__kernel void aggregate_cat(__global float *obj_t, __global float *effect_receiver, __global float *out, ushort m)
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
