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

vector<vector<float>> transpose(vector<vector<float>> &inp)
{
    // get kernel to execute
    cl_kernel kernel = kernels["transpose"];
    cl_int status;
    const ushort m = inp.size();
    const ushort n = inp[0].size();
    int size = (int) m * n;
    // create buffers
    cl_mem inp_buf = create_input_buffer_2d(inp, size, status);
    cl_mem out_buf = create_output_buffer(size, status);
    // Set the kernel argument (argument 0)
    status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_buf);
    status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inp_buf);
    status |=  clSetKernelArg(kernel, 2, sizeof(ushort), &m);
    status |=  clSetKernelArg(kernel, 3, sizeof(ushort), &n);
    checkError(status, "Setting kernel arguments");
    // execute kernel
    const size_t global[2] = {m, n};
    run_kernel(global, kernel, status);
    // read buffer to host
    float a_t[size];
    read_out_buffer(out_buf, a_t, size, status);
    // reshape to 2d
    vector<vector<float>> a_t_2d(n, vector<float>(m));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        a_t_2d[i][j] = a_t[global_idx(i, j, m)];
      }
    }
    // clean up -- destroy buffers and free up memory on device
    clReleaseMemObject(inp_buf);
    clReleaseMemObject(out_buf);

    return a_t_2d;
}

vector<vector<float>> sigmoid(vector<vector<float>> inp){
  // get kernel to execute
  cl_kernel kernel = kernels["sigmoid"];
  cl_int status;
  const ushort m = inp.size();
  const ushort n = inp[0].size();
  int size = (int) m * n;
  // create buffers
  cl_mem inp_buf = create_input_buffer_2d(inp, size, status);
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
  float out[size];
  read_out_buffer(out_buf, out, size, status);
  vector<vector<float>> x(m, vector<float>(n));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      x[i][j] = out[global_idx(i, j, n)];
    }
  }

  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);

  return x;
}

vector<vector<float>> relu(vector<vector<float>> inp){
  // get kernel to execute
  cl_kernel kernel = kernels["relu"];
  cl_int status;
  const ushort m = inp.size();
  const ushort n = inp[0].size();
  int size = (int) m * n;
  // create buffers
  cl_mem inp_buf = create_input_buffer_2d(inp, size, status);
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
  float out[size];
  read_out_buffer(out_buf, out, size, status);
  vector<vector<float>> x(m, vector<float>(n));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      x[i][j] = out[global_idx(i, j, n)];
    }
  }
  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);

  return x;
}

vector<vector<float>> add_bias(vector<vector<float>> inp, vector<float> bias){
  // get kernel to execute
  cl_kernel kernel = kernels["add_bias"];
  cl_int status;
  const ushort m = inp.size();
  const ushort n = inp[0].size();
  int size = (int) m * n;
  // create buffers
  cl_mem inp_buf = create_input_buffer_2d(inp, size, status);
  cl_mem bias_buf = create_input_buffer_1d(bias, n, status);
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
  float out[size];
  read_out_buffer(out_buf, out, size, status);
  vector<vector<float>> x(m, vector<float>(n));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      x[i][j] = out[global_idx(i, j, n)];
    }
  }

  clReleaseMemObject(inp_buf);
  clReleaseMemObject(out_buf);
  clReleaseMemObject(bias_buf);

  return x;
}

vector<vector<float>> interaction_cat(int term_w, int term_h, vector<vector<float>> sender, vector<vector<float>> receiver, vector<vector<float>> ri){
  // get kernel to execute
  cl_kernel kernel = kernels["interaction_cat"];
  cl_int status;

  // create buffers
  const ushort w_sender = sender.size();
  const ushort h_sender = sender[0].size();
  int size_sender = (int) w_sender * h_sender;
  cl_mem sender_buf = create_input_buffer_2d(sender, size_sender, status);
  const ushort w_receiver = receiver.size();
  const ushort h_receiver = receiver[0].size();
  int size_receiver = (int) w_receiver * h_receiver;
  cl_mem receiver_buf = create_input_buffer_2d(receiver, size_receiver, status);
  const ushort w_ri = ri.size();
  const ushort h_ri = ri[0].size();
  int size_ri = (int) w_ri * h_ri;
  cl_mem ri_buf = create_input_buffer_2d(ri, size_ri, status);
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
  float out[size_out];
  read_out_buffer(out_buf, out, size_out, status);
  vector<vector<float>> x(term_w, vector<float>(term_h));
  for (int i = 0; i < term_w; i++) {
    for (int j = 0; j < term_h; j++) {
      x[i][j] = out[global_idx(i, j, term_h)];
    }
  }

  clReleaseMemObject(sender_buf);
  clReleaseMemObject(receiver_buf);
  clReleaseMemObject(ri_buf);
  clReleaseMemObject(out_buf);

  return x;
}

vector<vector<float>> aggregate_cat(vector<vector<float>> obj_t, vector<vector<float>> effect_receiver){
  int term_w = obj_t.size() + effect_receiver.size();
	int term_h = obj_t[0].size();
  // get kernel to execute
  cl_kernel kernel = kernels["aggregate_cat"];
  cl_int status;

  // create buffers
  const ushort w_obj_t = obj_t.size();
  const ushort h_obj_t = obj_t[0].size();
  int size_obj_t = (int) w_obj_t * h_obj_t;
  cl_mem obj_t_buf = create_input_buffer_2d(obj_t, size_obj_t, status);
  const ushort w_effect_receiver = effect_receiver.size();
  const ushort h_effect_receiver = effect_receiver[0].size();
  int size_effect_receiver = (int) w_effect_receiver * h_effect_receiver;
  cl_mem effect_receiver_buf = create_input_buffer_2d(effect_receiver, size_effect_receiver, status);
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
  float out[size_out];
  read_out_buffer(out_buf, out, size_out, status);
  vector<vector<float>> x(term_w, vector<float>(term_h));
  for (int i = 0; i < term_w; i++) {
    for (int j = 0; j < term_h; j++) {
      x[i][j] = out[global_idx(i, j, term_h)];
    }
  }

  clReleaseMemObject(obj_t_buf);
  clReleaseMemObject(effect_receiver_buf);
  clReleaseMemObject(out_buf);

  return x;
}
/*
vector<vector<float>> matmul(vector<vector<float>> a, vector<vector<float>> b){
  // get kernel to execute
  cl_kernel kernel = kernels["matrixMul"];
  cl_int status;

  const ushort m = a.size();
	const ushort n = a[0].size();
	const ushort p = b[0].size();

  int a_size = (int) m * n;
  int b_size = (int) n * p;
  int out_size = (int) m * p;

  // create buffers
  cl_mem a_buf = create_input_buffer_2d(a, a_size, status);
  cl_mem b_buf = create_input_buffer_2d(b, b_size, status);
  cl_mem out_buf = create_output_buffer(out_size, status);
  // Set the kernel argument (argument 0)

  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf);
  status |=  clSetKernelArg(kernel, 3, sizeof(ushort), &m);
  status |=  clSetKernelArg(kernel, 4, sizeof(ushort), &p);
  checkError(status, "Setting kernel arguments");
  // execute kernel
  const size_t global[2] = {m, p};
  const size_t worksize = m * p;
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
  checkError(status, "Enqueuing kernel");
  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");
  // read buffer to host
  float out[out_size];
  read_out_buffer(out_buf, out, out_size, status);
  vector<vector<float>> c(m, vector<float>(p));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      c[i][j] = out[global_idx(i, j, p)];
    }
  }
  return c;
}
*/
vector<vector<float>> matmul(vector<vector<float>> a, vector<vector<float>> b){
  // get kernel to execute
  cl_kernel kernel = kernels["matMul"];
  cl_int status;

  const ushort m = a.size();
	const ushort n = a[0].size();
	const ushort p = b[0].size();

  int a_size = (int) m * n;
  int b_size = (int) n * p;
  int out_size = (int) m * p;

  // create buffers
  cl_mem a_buf = create_input_buffer_2d(a, a_size, status);
  cl_mem b_buf = create_input_buffer_2d(b, b_size, status);
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
  float out[out_size];
  read_out_buffer(out_buf, out, out_size, status);
  vector<vector<float>> c(m, vector<float>(p));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      c[i][j] = out[global_idx(i, j, p)];
    }
  }
  clReleaseMemObject(a_buf);
  clReleaseMemObject(b_buf);
  clReleaseMemObject(out_buf);

  return c;
}
