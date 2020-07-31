#define SIGMOID(inp) (1.0f / (1 + exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)

__attribute__((uses_global_work_offset(0)))
__kernel void add_bias(__global float *inp, __global float *bias, __global float *out, ushort m)
{
    // global space:  inp.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    out[x] = inp[x] + bias[y_idx];
}

__attribute__((uses_global_work_offset(0)))
__kernel void matMul(__global const float* a, __global const float* b, __global float* result,
  const ushort M, const ushort N, const ushort P)
{
    int idx = get_global_id(0);
    int k = 0;
    float temp = 0.0f;
    int i = idx / P;
    int j = idx % P;
    #pragma unroll
    for( k = 0; k < N; k++)
        temp += a[ i*N + k ] * b[ k*P + j ];
    result[idx] = temp;
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
