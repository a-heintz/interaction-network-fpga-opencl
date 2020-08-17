#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "utility.h"
#include "error_handling.hpp"
using namespace aocl_utils;
using namespace std;
using namespace std::chrono;
#define global_idx(x_idx, y_idx, x_size) (x_idx * x_size + y_idx)

int shrRoundUp(int K, int N)
{
    int rem = (N + K) % K;
    if (rem == 0)
        return N;
    else
        return N + K - rem;
}

cl_mem create_input_buffer_1d(vector<float> inp, int size, cl_int status){
  float inp_arr[size];
  for (int i = 0; i < size; i++) {
    inp_arr[i] = inp[i];
  }
  cl_mem inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inp_arr, &status);
  checkError(status, "Creating buffer inp_buf");
  return inp_buf;
}

cl_mem create_input_buffer_2d(vector<vector<float>> inp, int size, cl_int status){
  vector<float> flat_inp = flatten(inp);
  float inp_arr[size];
  for (int i = 0; i < size; i++) {
    inp_arr[i] = flat_inp[i];
  }
  cl_mem inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inp_arr, &status);
  checkError(status, "Creating buffer inp_buf");
  return inp_buf;
}

cl_mem create_output_buffer(int size, cl_int status){
  cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &status);
  checkError(status, "Creating buffer out_buf");
  return out_buf;
}

cl_mem create_intermediate_buffer(float* inp, int size, cl_int status){
  cl_mem int_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inp, &status);
  checkError(status, "Creating buffer out_buf");
  return int_buf;
}

void run_kernel(const size_t* global, cl_kernel kernel, cl_int status){
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueuing kernel");
  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");
}

void read_out_buffer(cl_mem out_buf, float* out, int size, cl_int status){
  //read_out_buffer(out_buf, a_t, size, status);
  status = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(float) * size, out, 0, NULL, NULL);
  checkError(status, "Reading back buffer out_buf");
}

cl_mem create_input_buffer_from_arr(float *inp, int size, cl_int status){
  cl_mem inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inp, &status);
  checkError(status, "Creating buffer inp_buf");
  return inp_buf;
}

void linear(cl_mem a, cl_mem b, cl_mem bias, cl_mem out, int m, int n, int p, char* activation)
{
  cl_int status;
  cl_kernel kernel = kernels["linear"];
  int activation_int;
  if(activation == "relu"){
    activation_int = 1;
  } else if(activation == "sigmoid"){
    activation_int = 2;
  } else {
    activation_int = 3;
  }
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias);
  status |=  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 4, sizeof(int), &m);
  status |=  clSetKernelArg(kernel, 5, sizeof(int), &n);
  status |=  clSetKernelArg(kernel, 6, sizeof(int), &p);
  status |=  clSetKernelArg(kernel, 7, sizeof(int), &activation_int);
  checkError(status, "Setting kernel arguments");
  size_t global[2] = {m, p};
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueuing kernel");
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");
}

void interaction_cat(int term_w, int term_h, int sender_w, int sender_h, int receiver_w, int receiver_h,
                     int ri_w, int ri_h, cl_mem sender, cl_mem receiver, cl_mem ri, cl_mem out)
{
  cl_kernel kernel = kernels["interaction_cat"];
  cl_int status;
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &sender);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &receiver);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &ri);
  status |=  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 4, sizeof(int), &term_h);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {term_w, term_h};
  run_kernel(global, kernel, status);
}

void aggregate_cat(cl_mem obj_t, cl_mem effect_receiver, cl_mem out,
                   int obj_t_w, int obj_t_h,
                   int effect_receiver_w, int effect_receiver_h)
{
  cl_kernel kernel = kernels["aggregate_cat"];
  cl_int status;
  int term_w = obj_t_w + effect_receiver_w;
	int term_h = obj_t_h;
  // Set the kernel argument (argument 0)
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &obj_t);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &effect_receiver);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 3, sizeof(int), &term_h);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {term_w, term_h};
  run_kernel(global, kernel, status);
}

void transpose(cl_mem in, cl_mem out, int m, int n)
{
    cl_kernel kernel = kernels["transpose"];
    cl_int status;
    status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
    status |=  clSetKernelArg(kernel, 2, sizeof(int), &m);
    status |=  clSetKernelArg(kernel, 3, sizeof(int), &n);
    checkError(status, "Setting kernel arguments");
    const size_t global[2] = {m, n};
    run_kernel(global, kernel, status);
}

void buf_fastMatMul(cl_mem a, cl_mem b, cl_mem out, int m, int n, int p)
{
  cl_kernel kernel = kernels["matrixMul"];
  cl_int status;
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 3, sizeof(int), &m);
  status |=  clSetKernelArg(kernel, 4, sizeof(int), &n);
  status |=  clSetKernelArg(kernel, 5, sizeof(int), &p);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {m, p};
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueuing kernel");
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");
}

void swi_IN( cl_mem BUF_obj_arr, cl_mem BUF_obj_arr_t, cl_mem BUF_sr_arr,
  cl_mem BUF_sender_arr, cl_mem BUF_rr_arr, cl_mem BUF_ri_arr, cl_mem BUF_receiver_arr,
  cl_mem BUF_effect_receiver_arr, cl_mem BUF_effect_receiver_arr_t, cl_mem BUF_predict_arr,
  cl_mem BUF_agg_arr, cl_mem BUF_inf_arr, cl_mem BUF_interaction_term_arr, cl_mem BUF_effect_arr,
  cl_mem BUF_pred_arr, cl_mem BUF_RM_x1, cl_mem BUF_RM_x2, cl_mem BUF_RM_x4,
  cl_mem BUF_RM_x_arr_t, cl_mem BUF_OM_x1, cl_mem BUF_OM_x2, cl_mem BUF_OM_x_arr_t,
  cl_mem BUF_RM_WEIGHT_0, cl_mem BUF_RM_WEIGHT_2, cl_mem BUF_RM_WEIGHT_4, cl_mem BUF_RM_WEIGHT_6,
  cl_mem BUF_OM_WEIGHT_0, cl_mem BUF_OM_WEIGHT_2, cl_mem BUF_OM_WEIGHT_4, cl_mem BUF_RM_BIAS_0,
  cl_mem BUF_RM_BIAS_2, cl_mem BUF_RM_BIAS_4, cl_mem BUF_RM_BIAS_6, cl_mem BUF_OM_BIAS_0,
  cl_mem BUF_OM_BIAS_2, cl_mem BUF_OM_BIAS_4,
  int obj_w, int obj_h, int sr_w, int sr_h, int ri_w, int ri_h,
  int rr_w, int rr_h, int obj_t_w, int obj_t_h, int sender_w,
  int sender_h, int receiver_w, int receiver_h, int term_w,
  int term_h, int effect_w, int effect_h, int effect_receiver_w,
  int effect_receiver_h, int aggregate_w, int aggregate_h, int pred_w,
  int pred_h, int out_w, int out_h, int RM_w0, int RM_w2, int RM_w4,
  int RM_w6, int OM_w0, int OM_w2, int OM_w4)
{
  cl_kernel kernel = kernels["swi_IN"];
  cl_int status;

  status  =  clSetKernelArg(kernel,  0, sizeof(cl_mem), &BUF_obj_arr);
  status |=  clSetKernelArg(kernel,  1, sizeof(cl_mem), &BUF_obj_arr_t);
  status |=  clSetKernelArg(kernel,  2, sizeof(cl_mem), &BUF_sr_arr);
  status |=  clSetKernelArg(kernel,  3, sizeof(cl_mem), &BUF_sender_arr);
  status |=  clSetKernelArg(kernel,  4, sizeof(cl_mem), &BUF_rr_arr);
  status |=  clSetKernelArg(kernel,  5, sizeof(cl_mem), &BUF_ri_arr);
  status |=  clSetKernelArg(kernel,  6, sizeof(cl_mem), &BUF_receiver_arr);
  status |=  clSetKernelArg(kernel,  7, sizeof(cl_mem), &BUF_effect_receiver_arr);
  status |=  clSetKernelArg(kernel,  8, sizeof(cl_mem), &BUF_effect_receiver_arr_t);
  status |=  clSetKernelArg(kernel,  9, sizeof(cl_mem), &BUF_predict_arr);
  status |=  clSetKernelArg(kernel, 10, sizeof(cl_mem), &BUF_agg_arr);
  status |=  clSetKernelArg(kernel, 11, sizeof(cl_mem), &BUF_inf_arr);
  status |=  clSetKernelArg(kernel, 12, sizeof(cl_mem), &BUF_interaction_term_arr);
  status |=  clSetKernelArg(kernel, 13, sizeof(cl_mem), &BUF_effect_arr);
  status |=  clSetKernelArg(kernel, 14, sizeof(cl_mem), &BUF_pred_arr);
  status |=  clSetKernelArg(kernel, 15, sizeof(cl_mem), &BUF_RM_x1);
  status |=  clSetKernelArg(kernel, 16, sizeof(cl_mem), &BUF_RM_x2);
  status |=  clSetKernelArg(kernel, 17, sizeof(cl_mem), &BUF_RM_x4);
  status |=  clSetKernelArg(kernel, 18, sizeof(cl_mem), &BUF_RM_x_arr_t);
  status |=  clSetKernelArg(kernel, 19, sizeof(cl_mem), &BUF_OM_x1);
  status |=  clSetKernelArg(kernel, 20, sizeof(cl_mem), &BUF_OM_x2);
  status |=  clSetKernelArg(kernel, 21, sizeof(cl_mem), &BUF_OM_x_arr_t);
  status |=  clSetKernelArg(kernel, 22, sizeof(cl_mem), &BUF_RM_WEIGHT_0);
  status |=  clSetKernelArg(kernel, 23, sizeof(cl_mem), &BUF_RM_WEIGHT_2);
  status |=  clSetKernelArg(kernel, 24, sizeof(cl_mem), &BUF_RM_WEIGHT_4);
  status |=  clSetKernelArg(kernel, 25, sizeof(cl_mem), &BUF_RM_WEIGHT_6);
  status |=  clSetKernelArg(kernel, 26, sizeof(cl_mem), &BUF_OM_WEIGHT_0);
  status |=  clSetKernelArg(kernel, 27, sizeof(cl_mem), &BUF_OM_WEIGHT_2);
  status |=  clSetKernelArg(kernel, 28, sizeof(cl_mem), &BUF_OM_WEIGHT_4);
  status |=  clSetKernelArg(kernel, 29, sizeof(cl_mem), &BUF_RM_BIAS_0);
  status |=  clSetKernelArg(kernel, 30, sizeof(cl_mem), &BUF_RM_BIAS_2);
  status |=  clSetKernelArg(kernel, 31, sizeof(cl_mem), &BUF_RM_BIAS_4);
  status |=  clSetKernelArg(kernel, 32, sizeof(cl_mem), &BUF_RM_BIAS_6);
  status |=  clSetKernelArg(kernel, 33, sizeof(cl_mem), &BUF_OM_BIAS_0);
  status |=  clSetKernelArg(kernel, 34, sizeof(cl_mem), &BUF_OM_BIAS_2);
  status |=  clSetKernelArg(kernel, 35, sizeof(cl_mem), &BUF_OM_BIAS_4);
  status |=  clSetKernelArg(kernel, 36, sizeof(int), &obj_w);
  status |=  clSetKernelArg(kernel, 37, sizeof(int), &obj_h);
  status |=  clSetKernelArg(kernel, 38, sizeof(int), &sr_w);
  status |=  clSetKernelArg(kernel, 39, sizeof(int), &sr_h);
  status |=  clSetKernelArg(kernel, 40, sizeof(int), &ri_w);
  status |=  clSetKernelArg(kernel, 41, sizeof(int), &ri_h);
  status |=  clSetKernelArg(kernel, 42, sizeof(int), &rr_w);
  status |=  clSetKernelArg(kernel, 43, sizeof(int), &rr_h);
  status |=  clSetKernelArg(kernel, 44, sizeof(int), &obj_t_w);
  status |=  clSetKernelArg(kernel, 45, sizeof(int), &obj_t_h);
  status |=  clSetKernelArg(kernel, 46, sizeof(int), &sender_w);
  status |=  clSetKernelArg(kernel, 47, sizeof(int), &sender_h);
  status |=  clSetKernelArg(kernel, 48, sizeof(int), &receiver_w);
  status |=  clSetKernelArg(kernel, 49, sizeof(int), &receiver_h);
  status |=  clSetKernelArg(kernel, 50, sizeof(int), &term_w);
  status |=  clSetKernelArg(kernel, 51, sizeof(int), &term_h);
  status |=  clSetKernelArg(kernel, 52, sizeof(int), &effect_w);
  status |=  clSetKernelArg(kernel, 53, sizeof(int), &effect_h);
  status |=  clSetKernelArg(kernel, 54, sizeof(int), &effect_receiver_w);
  status |=  clSetKernelArg(kernel, 55, sizeof(int), &effect_receiver_h);
  status |=  clSetKernelArg(kernel, 56, sizeof(int), &aggregate_w);
  status |=  clSetKernelArg(kernel, 57, sizeof(int), &aggregate_h);
  status |=  clSetKernelArg(kernel, 58, sizeof(int), &pred_w);
  status |=  clSetKernelArg(kernel, 59, sizeof(int), &pred_h);
  status |=  clSetKernelArg(kernel, 60, sizeof(int), &out_w);
  status |=  clSetKernelArg(kernel, 61, sizeof(int), &out_h);
  status |=  clSetKernelArg(kernel, 62, sizeof(int), &RM_w0);
  status |=  clSetKernelArg(kernel, 63, sizeof(int), &RM_w2);
  status |=  clSetKernelArg(kernel, 64, sizeof(int), &RM_w4);
  status |=  clSetKernelArg(kernel, 65, sizeof(int), &RM_w6);
  status |=  clSetKernelArg(kernel, 66, sizeof(int), &OM_w0);
  status |=  clSetKernelArg(kernel, 67, sizeof(int), &OM_w2);
  status |=  clSetKernelArg(kernel, 68, sizeof(int), &OM_w4);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {1, 1};
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueuing kernel");
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");
}
