#define SIGMOID(inp) (1.0f / (1 + native_exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)
#define RM_w0 250
#define RM_w2 250
#define RM_w4 250
#define RM_w6 1
#define OM_w0 200
#define OM_w2 200
#define OM_w4 3
/*
__kernel void swi_matrixMul(__global float* A,
                            __global float* B,
                            __global float* C,
                            __const int M,
                            __const int K,
                            __const int N){
  for(int m = 0; m < M; m++){
    for(int n = 0; n < N; n++){
      float acc = 0.0f;
      //#pragma unroll 4
      for(int k = 0; k < K; ++k){
          acc += A[m*K + k] * B[k*N + n];
      }
      C[m*N + n] = acc;
    }
  }
}

__kernel void swi_interaction_cat(__global float* sender,
                                  __global float* receiver,
                                  __global float* ri,
                                  __global float* out,
                                  int m, int n)
{
    for(int x_idx = 0; x_idx < m; x_idx++){
      for(int y_idx = 0; y_idx < n; y_idx++){
        int x = global_idx(x_idx, y_idx, n);
        if(x_idx < 3){
            out[x] = sender[x];
        } else if(x_idx < 6){
            out[x] = receiver[x - (3 * n)];
        } else {
            out[x] = ri[x - (6 * n)];
        }
      }
    }
}

__kernel void swi_transpose(__global float* a_t,
                            __global float* a,
                            int m, int n) {
  for(int x_idx = 0; x_idx < m; x_idx++)
    for(int y_idx = 0; y_idx < n; y_idx++)
      a_t[global_idx(y_idx, x_idx, n)] = a[global_idx(x_idx, y_idx, m)];
}

__kernel void swi_aggregate_cat(__global float* obj_t,
                                __global float* effect_receiver,
                                __global float* out,
                                int obj_t_w, int effect_receiver_w, int obj_t_h)
{
  int term_w = obj_t_w + effect_receiver_w;
	int term_h = obj_t_h;
  for(int x_idx = 0; x_idx < term_w; x_idx++){
    for(int y_idx = 0; y_idx < term_h; y_idx++){
        int x = global_idx(x_idx, y_idx, term_h);
        if(x_idx < 3){
          out[x] = obj_t[x];
        }else{
          out[x] = effect_receiver[x - (3 * term_w)];
        }
    }
  }
}

__kernel void swi_linear(
      __global float* A,
      __global float* C,
      __global float* B,
      __global float* bias,
      __const int M,
      __const int N,
      __const int P,
      __const int activation){
  swi_matrixMul(A, B, C, M, N, P);
  for(int m = 0; m < M; m++){
    for(int p = 0; p < P; p++){
      if(activation == 1){
        C[m * P + p] = RELU(C[m * P + p] + bias[p]);
      } else if(activation == 2){
        C[m * P + p] = SIGMOID(C[m * P + p] + bias[p]);
      } else {
        C[m * P + p] += bias[p];
      }
    }
  }
}

__kernel void swi_relational_model(
      __global float* x_arr,
      __global float* x_arr_t,
      __global float* effect_arr,
      __global float* x1,
      __global float* x2,
      __global float* x4,
      __global float* RM_WEIGHT_0,
      __global float* RM_WEIGHT_2,
      __global float* RM_WEIGHT_4,
      __global float* RM_WEIGHT_6,
      __global float* RM_BIAS_0,
      __global float* RM_BIAS_2,
      __global float* RM_BIAS_4,
      __global float* RM_BIAS_6,
      __const int term_w,
      __const int term_h,
      __const int w0,
      __const int w2,
      __const int w4,
      __const int w6
      ){
  printf("RM 0! \n");
  swi_transpose(x_arr_t, x_arr, term_w, term_h);
  printf("RM 1! \n");
  swi_linear(x_arr_t, x1, RM_WEIGHT_0, RM_BIAS_0, term_h, term_w, w0, 1); // relu
  printf("RM 2! \n");
  swi_linear(x1, x2, RM_WEIGHT_2, RM_BIAS_2, term_h, w0, w2, 1); // relu
  printf("RM 3! \n");
	swi_linear(x2, x4, RM_WEIGHT_4, RM_BIAS_4, term_h, w2, w4, 1); // relu
  printf("RM 4! \n");
	swi_linear(x4, effect_arr, RM_WEIGHT_6, RM_BIAS_6, term_h, w4, w6, 2); // sigmoid
}

__kernel void swi_object_model(
      __global float* x_arr,
      __global float* x_arr_t,
      __global float* inf_arr,
      __global float* x1,
      __global float* x2,
      __global float* OM_WEIGHT_0,
      __global float* OM_WEIGHT_2,
      __global float* OM_WEIGHT_4,
      __global float* OM_BIAS_0,
      __global float* OM_BIAS_2,
      __global float* OM_BIAS_4,
      __const int aggregate_w,
      __const int aggregate_h,
      __const int w0,
      __const int w2,
      __const int w4
      ){
  printf("OM 0! \n");
  swi_transpose(x_arr_t, x_arr, aggregate_w, aggregate_h);
  printf("OM 1! \n");
  swi_linear(x_arr_t, x1, OM_WEIGHT_0, OM_BIAS_0, aggregate_h, aggregate_w, w0, 1); // relu
  printf("OM 2! \n");
  swi_linear(x1, x2, OM_WEIGHT_2, OM_BIAS_2, aggregate_h, w0, w2, 1); // relu
	printf("OM 3! \n");
  swi_linear(x2, inf_arr, OM_WEIGHT_4, OM_BIAS_4, aggregate_h, w2, w4, 3); // None

}

__kernel void single_work_item_IN(__global float* obj_arr,
                                  __global float* sr_arr,
                                  __global float* rr_arr,
                                  __global float* ri_arr,
                                  __global float* obj_arr_t,
                                  __global float* sender_arr,
                                  __global float* receiver_arr,
                                  __global float* interaction_term_arr,
                                  __global float* interaction_term_arr_t,
                                  __global float* effect_arr,
                                  __global float* effect_receiver_arr,
                                  __global float* effect_receiver_arr_t,
                                  __global float* pred_arr,
                                  __global float* predict_arr,
                                  __global float* agg_arr,
                                  __global float* agg_arr_t,
                                  __global float* inf_arr,
                                  __global float* RM_WEIGHT_0,
                                  __global float* RM_WEIGHT_2,
                                  __global float* RM_WEIGHT_4,
                                  __global float* RM_WEIGHT_6,
                                  __global float* OM_WEIGHT_0,
                                  __global float* OM_WEIGHT_2,
                                  __global float* OM_WEIGHT_4,
                                  __global float* RM_BIAS_0,
                                  __global float* RM_BIAS_2,
                                  __global float* RM_BIAS_4,
                                  __global float* RM_BIAS_6,
                                  __global float* OM_BIAS_0,
                                  __global float* OM_BIAS_2,
                                  __global float* OM_BIAS_4,
                                  __global float* RM_x1,
                                  __global float* RM_x2,
                                  __global float* RM_x4,
                                  __global float* OM_x1,
                                  __global float* OM_x2,
                                  __const int obj_w,
                                  __const int obj_h,
                                  __const int ri_w,
                                  __const int ri_h,
                                  __const int rr_w,
                                  __const int rr_h,
                                  __const int sender_w,
                                  __const int sender_h,
                                  __const int receiver_w,
                                  __const int receiver_h,
                                  __const int term_w,
                                  __const int term_h,
                                  __const int effect_w,
                                  __const int effect_h,
                                  __const int effect_receiver_w,
                                  __const int effect_receiver_h,
                                  __const int aggregate_w,
                                  __const int aggregate_h,
                                  __const int pred_w,
                                  __const int pred_h){
  printf("hi 1!\n");
  swi_transpose(obj_arr_t, obj_arr, obj_w, obj_h);
  printf("hi 2!\n");
  swi_matrixMul(obj_arr_t, sr_arr, sender_arr, obj_h, obj_w, sender_h);
  printf("hi 3!\n");
	swi_matrixMul(obj_arr_t, rr_arr, receiver_arr, obj_h, obj_w, receiver_h);
  printf("hi 4!\n");
  swi_interaction_cat(sender_arr, receiver_arr, ri_arr, interaction_term_arr, term_w, term_h);
  printf("hi 5!\n");
  swi_relational_model(interaction_term_arr,interaction_term_arr_t,effect_arr,RM_x1,RM_x2,RM_x4,
    RM_WEIGHT_0,RM_WEIGHT_2,RM_WEIGHT_4,RM_WEIGHT_6,RM_BIAS_0,RM_BIAS_2,RM_BIAS_4,RM_BIAS_6,
    term_w,term_h,RM_w0,RM_w2,RM_w4,RM_w6);
  printf("hi 6!\n");
  swi_matrixMul(rr_arr, effect_arr, effect_receiver_arr, rr_w, rr_h, effect_h);
  printf("hi 7!\n");
  swi_transpose(effect_receiver_arr_t, effect_receiver_arr, effect_receiver_w, effect_receiver_h);
  printf("hi 8!\n");
  swi_aggregate_cat(obj_arr_t, effect_receiver_arr, agg_arr, obj_h, effect_receiver_w, obj_w);
  printf("hi 9!\n");
  swi_object_model( agg_arr, agg_arr_t, inf_arr, OM_x1, OM_x2,
    OM_WEIGHT_0, OM_WEIGHT_2, OM_WEIGHT_4, OM_BIAS_0, OM_BIAS_2, OM_BIAS_4,
    aggregate_w, aggregate_h, OM_w0, OM_w2, OM_w4);
  printf("hi 10!\n");
  swi_transpose(predict_arr, inf_arr, pred_h, pred_w);
  printf("hi 11!\n");
  swi_matrixMul(predict_arr, sr_arr, sender_arr, pred_w, pred_h, sender_h);
  printf("hi 12!\n");
	swi_matrixMul(predict_arr, rr_arr, receiver_arr, pred_w, pred_h, receiver_h);
  printf("hi 13!\n");
  swi_interaction_cat(sender_arr, receiver_arr, ri_arr, interaction_term_arr, term_w, term_h);
  printf("hi 14!\n");
  swi_relational_model(interaction_term_arr,interaction_term_arr_t,pred_arr,RM_x1,RM_x2,RM_x4,
    RM_WEIGHT_0,RM_WEIGHT_2,RM_WEIGHT_4,RM_WEIGHT_6,RM_BIAS_0,RM_BIAS_2,RM_BIAS_4,RM_BIAS_6,
    term_w,term_h,RM_w0,RM_w2,RM_w4,RM_w6);
  printf("hi 15!\n");
}


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
*/
__kernel void matrixMul(
      __global float* restrict A,
      __global float* restrict B,
      __global float* restrict C,
      __const int M,
      __const int K,
      __const int N){
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
      __const int M,
      __const int N,
      __const int P,
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
                        int n, int m)
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

__attribute__((uses_global_work_offset(0)))
__kernel void aggregate_cat(__global float* restrict obj_t,
                            __global float* restrict effect_receiver,
                            __global float* restrict out,
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
