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







void cl_linear(cl_mem a, cl_mem b, cl_mem bias, cl_mem out, int m, int n, int p, char* activation)
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
