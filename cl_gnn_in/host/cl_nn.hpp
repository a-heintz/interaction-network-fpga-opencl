#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "utility.h"
#include "error_handling.hpp"
using namespace aocl_utils;
using namespace std;
using namespace std::chrono;
#define global_idx(x_idx, y_idx, x_size) (x_idx * x_size + y_idx)

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

void transpose(float *inp, float* out, int m, int n)
{
    // get kernel to execute
    cl_kernel kernel = kernels["transpose"];
    cl_int status;
    const ushort m_ = (ushort) m;
    const ushort n_ = (ushort) n;
    int size = m * n;
    // create buffers
    cl_mem inp_buf = create_input_buffer_from_arr(inp, size, status);
    cl_mem out_buf = create_output_buffer(size, status);
    // Set the kernel argument (argument 0)
    status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_buf);
    status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inp_buf);
    status |=  clSetKernelArg(kernel, 2, sizeof(ushort), &m_);
    status |=  clSetKernelArg(kernel, 3, sizeof(ushort), &n_);
    checkError(status, "Setting kernel arguments");
    // execute kernel
    const size_t global[2] = {m, n};
    run_kernel(global, kernel, status);
    // read buffer to host
    read_out_buffer(out_buf, out, size, status);
    // clean up -- destroy buffers and free up memory on device
    clReleaseMemObject(inp_buf);
    clReleaseMemObject(out_buf);

}

void sigmoid(float* inp, float* out, int m_, int n_){
  // get kernel to execute
  cl_kernel kernel = kernels["sigmoid"];
  cl_int status;
  const ushort m = m_; //inp.size();
  const ushort n = n_; //inp[0].size();
  int size = (int) m * n;
  // create buffers
  cl_mem inp_buf = create_input_buffer_from_arr(inp, size, status);
  cl_mem out_buf = create_output_buffer(size, status);
  // Set the kernel argument (argument 0)
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inp_buf);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 2, sizeof(ushort), &n);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  const size_t global[2] = {m, n};
  run_kernel(global, kernel, status);
  // read buffer to host
  read_out_buffer(out_buf, out, size, status);

  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);
}

void relu(float* inp, float* out, int m_, int n_){
  // get kernel to execute
  cl_kernel kernel = kernels["relu"];
  cl_int status;
  const ushort m = m_; //inp.size();
  const ushort n = n_; //inp[0].size();
  int size = (int) m * n;
  // create buffers
  cl_mem inp_buf = create_input_buffer_from_arr(inp, size, status);
  cl_mem out_buf = create_output_buffer(size, status);
  // Set the kernel argument (argument 0)
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inp_buf);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 2, sizeof(ushort), &n);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  const size_t global[2] = {m, n};
  run_kernel(global, kernel, status);
  // read buffer to host
  read_out_buffer(out_buf, out, size, status);
  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);
}

void add_bias(float* inp, float* bias, float* out, int m_, int n_){
  // get kernel to execute
  cl_kernel kernel = kernels["add_bias"];
  cl_int status;
  const ushort m = (ushort) m_; //inp.size();
  const ushort n = (ushort) n_; //inp[0].size();
  int size = (int) m * n;
  // create buffers
  cl_mem inp_buf = create_input_buffer_from_arr(inp, size, status);
  cl_mem bias_buf = create_input_buffer_from_arr(bias, n, status);
  cl_mem out_buf = create_output_buffer(size, status);
  // Set the kernel argument (argument 0)
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inp_buf);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &bias_buf);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 3, sizeof(ushort), &n);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  const size_t global[2] = {m, n};
  run_kernel(global, kernel, status);
  // read buffer to host
  read_out_buffer(out_buf, out, size, status);

  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);
  clReleaseMemObject(bias_buf);
}

void interaction_cat(int term_w,
                     int term_h,
                     int sender_w,
                     int sender_h,
                     int receiver_w,
                     int receiver_h,
                     int ri_w,
                     int ri_h,
                     float* sender,
                     float* receiver,
                     float* ri,
                     float* out){
  // get kernel to execute
  cl_kernel kernel = kernels["interaction_cat"];
  cl_int status;

  // create buffers
  int size_sender = sender_w * sender_h;
  cl_mem sender_buf = create_input_buffer_from_arr(sender, size_sender, status);
  int size_receiver = receiver_w * receiver_h;
  cl_mem receiver_buf = create_input_buffer_from_arr(receiver, size_receiver, status);
  int size_ri = ri_w * ri_h;
  cl_mem ri_buf = create_input_buffer_from_arr(ri, size_ri, status);
  int size_out = term_w * term_h;
  cl_mem out_buf = create_output_buffer(size_out, status);
  // Set the kernel argument (argument 0)
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &sender_buf);
  status  =  clSetKernelArg(kernel, 1, sizeof(cl_mem), &receiver_buf);
  status  =  clSetKernelArg(kernel, 2, sizeof(cl_mem), &ri_buf);
  status  =  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 4, sizeof(ushort), &term_h);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  ushort t_w = (ushort) term_w;
  ushort t_h = (ushort) term_h;
  const size_t global[2] = {term_w, term_h};
  run_kernel(global, kernel, status);
  // read buffer to host
  read_out_buffer(out_buf, out, size_out, status);

  clReleaseMemObject(sender_buf);
  clReleaseMemObject(receiver_buf);
  clReleaseMemObject(ri_buf);
  clReleaseMemObject(out_buf);
}

void aggregate_cat(float* obj_t,
                                    float* effect_receiver,
                                    float* out,
                                    int obj_t_w,
                                    int obj_t_h,
                                    int effect_receiver_w,
                                    int effect_receiver_h){
  int term_w = obj_t_w + effect_receiver_w;
	int term_h = obj_t_h;
  // get kernel to execute
  cl_kernel kernel = kernels["aggregate_cat"];
  cl_int status;

  // create buffers
  int size_obj_t = obj_t_w * obj_t_h;
  cl_mem obj_t_buf = create_input_buffer_from_arr(obj_t, size_obj_t, status);
  int size_effect_receiver = effect_receiver_w * effect_receiver_h;
  cl_mem effect_receiver_buf = create_input_buffer_from_arr(effect_receiver, size_effect_receiver, status);
  int size_out = term_w * term_h;
  cl_mem out_buf = create_output_buffer(size_out, status);
  // Set the kernel argument (argument 0)
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &obj_t_buf);
  status  =  clSetKernelArg(kernel, 1, sizeof(cl_mem), &effect_receiver_buf);
  status  =  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 3, sizeof(ushort), &term_h);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  ushort t_w = (ushort) term_w;
  ushort t_h = (ushort) term_h;
  const size_t global[2] = {term_w, term_h};
  run_kernel(global, kernel, status);
  // read buffer to host
  read_out_buffer(out_buf, out, size_out, status);

  clReleaseMemObject(obj_t_buf);
  clReleaseMemObject(effect_receiver_buf);
  clReleaseMemObject(out_buf);

}

void matmul(float* a, float* b, float* out, int m_, int n_, int p_){
  // get kernel to execute
  cl_kernel kernel = kernels["matMul"];
  cl_int status;

  const ushort m = (ushort) m_; // a.size();
	const ushort n = (ushort) n_; // a[0].size();
	const ushort p = (ushort) p_; // b[0].size();

  int a_size = (int) m * n;
  int b_size = (int) n * p;
  int out_size = (int) m * p;

  // create buffers
  cl_mem a_buf = create_input_buffer_from_arr(a, a_size, status);
  cl_mem b_buf = create_input_buffer_from_arr(b, b_size, status);
  cl_mem out_buf = create_output_buffer(out_size, status);
  // Set the kernel argument (argument 0)

  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 3, sizeof(ushort), &m);
  status |=  clSetKernelArg(kernel, 4, sizeof(ushort), &n);
  status |=  clSetKernelArg(kernel, 5, sizeof(ushort), &p);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  //const size_t global[2] = {wC, hC};
  const size_t worksize = m * p;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &worksize, NULL, 0, NULL, NULL);
  checkError(status, "Enqueuing kernel");
  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");
  // read buffer to host
  read_out_buffer(out_buf, out, out_size, status);

  clReleaseMemObject(a_buf);
  clReleaseMemObject(b_buf);
  clReleaseMemObject(out_buf);
}
