#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "utility.h"
#include "error_handling.hpp"
using namespace aocl_utils;
using namespace std;
using namespace std::chrono;
#define global_idx(x_idx, y_idx, x_size) (x_idx * x_size + y_idx)

int RoundUp(int N, int K)
{
    int rem = (N + K) % K;
    if (rem == 0)
        return N;
    else
        return N + K - rem;
}

cl_mem create_input_buffer_1d(vector<dtype> inp, int size, cl_int status){
  dtype inp_arr[size];
  for (int i = 0; i < size; i++) {
    inp_arr[i] = inp[i];
  }
  cl_mem inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(dtype) * size, inp_arr, &status);
  checkError(status, "Creating buffer inp_buf");
  return inp_buf;
}

cl_mem create_input_buffer_2d(vector<vector<dtype>> inp, int size, cl_int status){
  vector<dtype> flat_inp = flatten(inp);
  dtype inp_arr[size];
  for (int i = 0; i < size; i++) {
    inp_arr[i] = flat_inp[i];
  }
  cl_mem inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(dtype) * size, inp_arr, &status);
  checkError(status, "Creating buffer inp_buf");
  return inp_buf;
}

cl_mem create_output_buffer(int size, cl_int status){
  cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(dtype) * size, NULL, &status);
  checkError(status, "Creating buffer out_buf");
  return out_buf;
}

cl_mem create_intermediate_buffer(dtype* inp, int size, cl_int status){
  cl_mem int_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(dtype) * size, inp, &status);
  checkError(status, "Creating buffer out_buf");
  return int_buf;
}

cl_mem create_intermediate_buffer_no_inp(int size, cl_int status){
  dtype inp[size];
  cl_mem int_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(dtype) * size, inp, &status);
  checkError(status, "Creating buffer out_buf");
  return int_buf;
}

void run_kernel(cl_command_queue queue, const size_t* global, cl_kernel kernel, cl_event event, cl_int status){
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
  checkError(status, "Enqueuing kernel");
  // Wait for command queue to complete pending events
  //status = clFinish(queue);
  //checkError(status, "Waiting for queue to finish");
}

void read_out_buffer(cl_command_queue queue, cl_mem out_buf, dtype* out, int size, cl_int status){
  //read_out_buffer(out_buf, a_t, size, status);
  status = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(dtype) * size, out, 0, NULL, NULL);
  checkError(status, "Reading back buffer out_buf");
}

cl_mem create_input_buffer_from_arr(dtype *inp, int size, cl_int status){
  cl_mem inp_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(dtype) * size, inp, &status);
  checkError(status, "Creating buffer inp_buf");
  return inp_buf;
}

float linear(cl_command_queue queue, cl_mem a, cl_mem b, cl_mem bias, cl_mem out, int m, int n, int p, char* activation)
{
  auto start = high_resolution_clock::now();
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
  int m_pad = RoundUp(m, BLOCK_SIZE);
  int n_pad = RoundUp(n, BLOCK_SIZE);
  int p_pad = RoundUp(p, BLOCK_SIZE);
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias);
  status |=  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 4, sizeof(int), &m);
  status |=  clSetKernelArg(kernel, 5, sizeof(int), &n);
  status |=  clSetKernelArg(kernel, 6, sizeof(int), &p);
  status |=  clSetKernelArg(kernel, 7, sizeof(int), &m_pad);
  status |=  clSetKernelArg(kernel, 8, sizeof(int), &n_pad);
  status |=  clSetKernelArg(kernel, 9, sizeof(int), &p_pad);
  status |=  clSetKernelArg(kernel, 10, sizeof(int), &activation_int);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {m_pad / WPT, p_pad / WPT};
  const size_t local[2] = {BLOCK_SIZE / WPT, BLOCK_SIZE / WPT};
  cl_event event;
  auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//cout << ">> Time taken CPU Overhead: "
	//		 << duration.count() << " microsecs \n";
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
  checkError(status, "Enqueuing kernel");
  clWaitForEvents(1, &event);
  cl_ulong time_start;
  cl_ulong time_end;
  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

  float nanoSeconds = time_end-time_start;
  //cout << "linear: OpenCl Execution time is: "<< nanoSeconds / 1000.0 <<" microseconds" << "\n";
  return nanoSeconds;
}

float interaction_cat(cl_command_queue queue, int term_w, int term_h, int sender_w, int sender_h, int receiver_w, int receiver_h,
                     int ri_w, int ri_h, cl_mem sender, cl_mem receiver, cl_mem ri, cl_mem out)
{
  auto start = high_resolution_clock::now();
  cl_kernel kernel = kernels["interaction_cat"];
  cl_int status;
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &sender);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &receiver);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &ri);
  status |=  clSetKernelArg(kernel, 3, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 4, sizeof(int), &term_h);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {term_w, term_h};
  cl_event event;
  auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//cout << ">> Time taken CPU Overhead: "
	//		 << duration.count() << " microsecs \n";
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
  checkError(status, "Enqueuing kernel");
  clWaitForEvents(1, &event);
  cl_ulong time_start;
  cl_ulong time_end;
  status = clFinish(queue);

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

  float nanoSeconds = time_end-time_start;
  //cout << "interaction_cat: OpenCl Execution time is: "<< nanoSeconds / 1000.0 <<" microseconds" << "\n";
  checkError(status, "Waiting for queue to finish");
  return nanoSeconds;
}

float aggregate_cat(cl_command_queue queue, cl_mem obj_t, cl_mem effect_receiver, cl_mem out,
                   int obj_t_w, int obj_t_h,
                   int effect_receiver_w, int effect_receiver_h)
{
  auto start = high_resolution_clock::now();
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
  cl_event event;
  auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//cout << ">> Time taken CPU Overhead: "
	//		 << duration.count() << " microsecs \n";
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
  checkError(status, "Enqueuing kernel");
  clWaitForEvents(1, &event);
  cl_ulong time_start;
  cl_ulong time_end;
  status = clFinish(queue);

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

  float nanoSeconds = time_end-time_start;
  //cout << "aggregate_cat: OpenCl Execution time is: "<< nanoSeconds / 1000.0 <<" microseconds" << "\n";
  checkError(status, "Waiting for queue to finish");
  return nanoSeconds;
}

float transpose(cl_command_queue queue, cl_mem in, cl_mem out, int m, int n)
{
    auto start = high_resolution_clock::now();
    cl_kernel kernel = kernels["transpose"];
    cl_int status;
    status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &in);
    status |=  clSetKernelArg(kernel, 2, sizeof(int), &m);
    status |=  clSetKernelArg(kernel, 3, sizeof(int), &n);
    checkError(status, "Setting kernel arguments");
    const size_t global[2] = {m, n};

    cl_event event;
    auto stop = high_resolution_clock::now();
  	auto duration = duration_cast<microseconds>(stop - start);
  	//cout << ">> Time taken CPU Overhead: "
  	//		 << duration.count() << " microsecs \n";
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
    checkError(status, "Enqueuing kernel");
    clWaitForEvents(1, &event);
    cl_ulong time_start;
    cl_ulong time_end;
    status = clFinish(queue);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    float nanoSeconds = time_end-time_start;
    //cout << "transpose: OpenCl Execution time is: "<< nanoSeconds / 1000.0 <<" microseconds" << "\n";
    checkError(status, "Waiting for queue to finish");
    return nanoSeconds;
}

float buf_fastMatMul(cl_command_queue queue, cl_mem a, cl_mem b, cl_mem out, int m, int n, int p)
{
  auto start = high_resolution_clock::now();
  cl_kernel kernel = kernels["GEMM"];
  cl_int status;
  int m_pad = RoundUp(m, BLOCK_SIZE);
  int n_pad = RoundUp(n, BLOCK_SIZE);
  int p_pad = RoundUp(p, BLOCK_SIZE);
  status  =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
  status |=  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
  status |=  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);
  status |=  clSetKernelArg(kernel, 3, sizeof(int), &m);
  status |=  clSetKernelArg(kernel, 4, sizeof(int), &n);
  status |=  clSetKernelArg(kernel, 5, sizeof(int), &p);
  status |=  clSetKernelArg(kernel, 6, sizeof(int), &m_pad);
  status |=  clSetKernelArg(kernel, 7, sizeof(int), &n_pad);
  status |=  clSetKernelArg(kernel, 8, sizeof(int), &p_pad);
  checkError(status, "Setting kernel arguments");
  const size_t global[2] = {m_pad / WPT, p_pad / WPT};
  const size_t local[2] = {BLOCK_SIZE / WPT, BLOCK_SIZE / WPT};
  cl_event event;
  auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	//cout << ">> Time taken CPU Overhead: "
	//		 << duration.count() << " microsecs \n";
  status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
  checkError(status, "Enqueuing kernel");
  clWaitForEvents(1, &event);
  cl_ulong time_start;
  cl_ulong time_end;

  status = clFinish(queue);
  checkError(status, "Waiting for queue to finish");

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

  float nanoSeconds = time_end-time_start;
  //cout << "buf_fastMatMul: OpenCl Execution time is: "<< nanoSeconds / 1000.0 <<" microseconds" << "\n";

  return nanoSeconds;
}
