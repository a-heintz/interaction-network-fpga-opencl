#define SIGMOID(inp) (1.0f / (1 + native_exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)

/*
__kernel void swi_matrixMul(
      __global float* restrict A,
      __global float* restrict B,
      __global float* restrict C,
      __const int M,
      __const int K,
      __const int N){
  for(int m = 0; m < M; m++){
    for(int n = 0; n < N; n++){
      float acc = 0.0f;
      #pragma unroll 4
      for(int k = 0; k < K; ++k){
          acc += A[m*K + k] * B[k*N + n];
      }
      C[m*N + n] = acc;
    }
  }
}

__kernel void swi_transpose(__global float* restrict a,
                            __global float* restrict a_t,
                            int m, int n) {
  for(int x_idx = 0; x_idx < m; x_idx++){
    for(int y_idx = 0; y_idx < n; y_idx++){
      a_t[global_idx(y_idx, x_idx, m)] = a[global_idx(x_idx, y_idx, n)];
    }
  }
}

__attribute__((uses_global_work_offset(0)))
__kernel void swi_interaction_cat(__global float* restrict sender,
                              __global float* restrict receiver,
                              __global float* restrict ri,
                              __global float* restrict out,
                              int n, int m)
{
  for(int x_idx = 0; x_idx < n; x_idx++){
    for(int y_idx = 0; y_idx < m; y_idx++){
      int x = global_idx(x_idx, y_idx, m);
      if(x_idx < 3){
          out[x] = sender[x];
      } else if(x_idx < 6){
          out[x] = receiver[x - (3 * m)];
      } else {
          out[x] = ri[x - (6 * m)];
      }
    }
  }
}


__kernel void swi_aggregate_cat(__global float* restrict obj_t, __global float* restrict effect_receiver, __global float* restrict out,
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
        out[x] = effect_receiver[x - (3 * term_h)];
      }
    }
  }
}

__kernel void swi_linear(
      __global float* restrict A,
      __global float* restrict B,
      __global float* restrict bias,
      __global float* restrict C,
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

__kernel void swi_relational_model(__global float* restrict BUF_x_arr, __global float* restrict BUF_x_out, __const int x_w,
  __const int x_h, __const int RM_w0, __const int RM_w2, __const int RM_w4, __const int RM_w6,
	__global float* restrict BUF_RM_WEIGHT_0, __global float* restrict BUF_RM_WEIGHT_2, __global float* restrict BUF_RM_WEIGHT_4,
  __global float* restrict BUF_RM_WEIGHT_6,
	__global float* restrict BUF_RM_BIAS_0, __global float* restrict BUF_RM_BIAS_2, __global float* restrict BUF_RM_BIAS_4,
  __global float* restrict BUF_RM_BIAS_6,
	__global float* restrict BUF_RM_x1, __global float* restrict BUF_RM_x2, __global float* restrict BUF_RM_x4,
  __global float* restrict BUF_RM_x_arr_t)
{
	swi_transpose(BUF_x_arr, BUF_RM_x_arr_t, x_w, x_h);
  swi_linear(BUF_RM_x_arr_t, BUF_RM_WEIGHT_0, BUF_RM_BIAS_0, BUF_RM_x1, x_h, x_w, RM_w0, 1);
	swi_linear(BUF_RM_x1, BUF_RM_WEIGHT_2, BUF_RM_BIAS_2, BUF_RM_x2, x_h, RM_w0, RM_w2, 1);
	swi_linear(BUF_RM_x2, BUF_RM_WEIGHT_4, BUF_RM_BIAS_4, BUF_RM_x4, x_h, RM_w2, RM_w4, 1);
	swi_linear(BUF_RM_x4, BUF_RM_WEIGHT_6, BUF_RM_BIAS_6, BUF_x_out, x_h, RM_w4, RM_w6, 2);
}

__kernel void swi_object_model(__global float* restrict BUF_x_arr, __global float* restrict BUF_x_out,
  __const int x_w, __const int x_h, __const int OM_w0, __const int OM_w2, __const int OM_w4,
									__global float* restrict BUF_OM_WEIGHT_0, __global float* restrict BUF_OM_WEIGHT_2, __global float* restrict BUF_OM_WEIGHT_4,
									__global float* restrict BUF_OM_BIAS_0, __global float* restrict BUF_OM_BIAS_2, __global float* restrict BUF_OM_BIAS_4,
									__global float* restrict BUF_OM_x_arr_t, __global float* restrict BUF_OM_x1, __global float* restrict BUF_OM_x2)
{
	swi_transpose(BUF_x_arr, BUF_OM_x_arr_t, x_w, x_h);
	swi_linear(BUF_OM_x_arr_t, BUF_OM_WEIGHT_0, BUF_OM_BIAS_0, BUF_OM_x1, x_h, x_w, OM_w0, 1);
	swi_linear(BUF_OM_x1, BUF_OM_WEIGHT_2, BUF_OM_BIAS_2, BUF_OM_x2, x_h, OM_w0, OM_w2, 1);
	swi_linear(BUF_OM_x2, BUF_OM_WEIGHT_4, BUF_OM_BIAS_4, BUF_x_out, x_h, OM_w2, OM_w4, 3);
}

__kernel void swi_IN( __global float* restrict BUF_obj_arr, __global float* restrict BUF_obj_arr_t, __global float* restrict BUF_sr_arr,
  __global float* restrict BUF_sender_arr, __global float* restrict BUF_rr_arr, __global float* restrict BUF_ri_arr, __global float* restrict BUF_receiver_arr,
  __global float* restrict BUF_effect_receiver_arr, __global float* restrict BUF_effect_receiver_arr_t, __global float* restrict BUF_predict_arr,
  __global float* restrict BUF_agg_arr, __global float* restrict BUF_inf_arr, __global float* restrict BUF_interaction_term_arr, __global float* restrict BUF_effect_arr,
  __global float* restrict BUF_pred_arr, __global float* restrict BUF_RM_x1, __global float* restrict BUF_RM_x2, __global float* restrict BUF_RM_x4,
  __global float* restrict BUF_RM_x_arr_t, __global float* restrict BUF_OM_x1, __global float* restrict BUF_OM_x2, __global float* restrict BUF_OM_x_arr_t,
  __global float* restrict BUF_RM_WEIGHT_0, __global float* restrict BUF_RM_WEIGHT_2, __global float* restrict BUF_RM_WEIGHT_4, __global float* restrict BUF_RM_WEIGHT_6,
  __global float* restrict BUF_OM_WEIGHT_0, __global float* restrict BUF_OM_WEIGHT_2, __global float* restrict BUF_OM_WEIGHT_4, __global float* restrict BUF_RM_BIAS_0,
  __global float* restrict BUF_RM_BIAS_2, __global float* restrict BUF_RM_BIAS_4, __global float* restrict BUF_RM_BIAS_6, __global float* restrict BUF_OM_BIAS_0,
  __global float* restrict BUF_OM_BIAS_2, __global float* restrict BUF_OM_BIAS_4,
  __const int obj_w, __const int obj_h, __const int sr_w, __const int sr_h, __const int ri_w, __const int ri_h,
  __const int rr_w, __const int rr_h, __const int obj_t_w, __const int obj_t_h, __const int sender_w,
  __const int sender_h, __const int receiver_w, __const int receiver_h, __const int term_w,
  __const int term_h, __const int effect_w, __const int effect_h, __const int effect_receiver_w,
  __const int effect_receiver_h, __const int aggregate_w, __const int aggregate_h, __const int pred_w,
  __const int pred_h, __const int out_w, __const int out_h, __const int RM_w0, __const int RM_w2, __const int RM_w4,
  __const int RM_w6, __const int OM_w0, __const int OM_w2, __const int OM_w4)
{
  swi_transpose(BUF_obj_arr, BUF_obj_arr_t, obj_w, obj_h);
  swi_matrixMul(BUF_obj_arr_t, BUF_sr_arr, BUF_sender_arr, obj_t_w, obj_t_h, sender_h);
  swi_matrixMul(BUF_obj_arr_t, BUF_rr_arr, BUF_receiver_arr, obj_t_w, obj_t_h, receiver_h);
  swi_interaction_cat(BUF_sender_arr, BUF_receiver_arr, BUF_ri_arr, BUF_interaction_term_arr, term_w, term_h);
  //printf("%i %i \n ------------------ \n", term_w, term_h);
  //for(int k = 0; k < term_w*term_h; k++){
  //    printf("%f \n", BUF_interaction_term_arr[k]);
  //}
  //printf("\n ------------------ \n");
  swi_relational_model(BUF_interaction_term_arr, BUF_effect_arr, term_w, term_h, RM_w0, RM_w2, RM_w4, RM_w6,
		BUF_RM_WEIGHT_0, BUF_RM_WEIGHT_2, BUF_RM_WEIGHT_4, BUF_RM_WEIGHT_6,
		BUF_RM_BIAS_0, BUF_RM_BIAS_2, BUF_RM_BIAS_4, BUF_RM_BIAS_6,
		BUF_RM_x1, BUF_RM_x2, BUF_RM_x4, BUF_RM_x_arr_t);
  swi_matrixMul(BUF_rr_arr, BUF_effect_arr, BUF_effect_receiver_arr, rr_w, rr_h, effect_h);
	swi_transpose(BUF_effect_receiver_arr, BUF_effect_receiver_arr_t, effect_receiver_w, effect_receiver_h);
  swi_aggregate_cat(BUF_obj_arr_t, BUF_effect_receiver_arr_t, BUF_agg_arr, obj_t_w, effect_receiver_w, obj_t_h);
	swi_object_model(BUF_agg_arr, BUF_inf_arr, aggregate_w, aggregate_h, OM_w0, OM_w2, OM_w4,
		BUF_OM_WEIGHT_0, BUF_OM_WEIGHT_2, BUF_OM_WEIGHT_4, BUF_OM_BIAS_0, BUF_OM_BIAS_2, BUF_OM_BIAS_4,
		BUF_OM_x_arr_t, BUF_OM_x1, BUF_OM_x2);
	swi_transpose(BUF_inf_arr, BUF_predict_arr, pred_h, pred_w);
	swi_matrixMul(BUF_predict_arr, BUF_sr_arr, BUF_sender_arr, pred_w, pred_h, sender_h);
	swi_matrixMul(BUF_predict_arr, BUF_rr_arr, BUF_receiver_arr, pred_w, pred_h, receiver_h);
  swi_interaction_cat(BUF_sender_arr, BUF_receiver_arr, BUF_ri_arr, BUF_interaction_term_arr, term_w, term_h);
	swi_relational_model(BUF_interaction_term_arr, BUF_pred_arr, term_w, term_h, RM_w0, RM_w2, RM_w4, RM_w6,
		BUF_RM_WEIGHT_0, BUF_RM_WEIGHT_2, BUF_RM_WEIGHT_4, BUF_RM_WEIGHT_6,
		BUF_RM_BIAS_0, BUF_RM_BIAS_2, BUF_RM_BIAS_4, BUF_RM_BIAS_6,
		BUF_RM_x1, BUF_RM_x2, BUF_RM_x4, BUF_RM_x_arr_t);
} */

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
__kernel void transpose(__global float* restrict a_t,
                        __global float* restrict a,
                        int m, int n)
{
    // global space:  a.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    a_t[global_idx(y_idx, x_idx, m)] = a[global_idx(x_idx, y_idx, n)];
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
