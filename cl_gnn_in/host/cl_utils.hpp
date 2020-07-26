#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "utility.h"
#include "error_handling.hpp"
using namespace aocl_utils;
using namespace std;
using namespace std::chrono;

#define NUM_CMD_QUEUES    16

cl_platform_id platform;                        // OpenCL platform
cl_device_id device;                            // device ID
cl_context context;                             // context
cl_command_queue queues[NUM_CMD_QUEUES];        // command queue
cl_program program;                             // program
map<string, cl_kernel> kernels;                 // kernels

void createKernel(string kernel_name){
    // error code returned from api calls
    cl_int err;
    // create kernel and check for errors
    kernels[kernel_name] = clCreateKernel(program, kernel_name.c_str(), &err);
    checkErr(err, __LINE__);
}

void initializeOpenCLParameters(){
    cl_int err;

    if(!setCwdToExeDir()) {
	exit(1);
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA");
    if(platform == NULL) {
      std::cerr << "ERROR: FPGA platform not found." << std::endl;
      exit(1);
    }
    // Query the available OpenCL devices.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &err);
    checkErr(err, __LINE__);

    // Create all of the command queues
    for (unsigned int i = 0; i < NUM_CMD_QUEUES; i++) {
        queues[i] = clCreateCommandQueue(context, device, 0, &err);
        checkErr(err, __LINE__);
    }

    // Create the program.
    std::string cl_program_name = "gnn";
    std::string binary_file = getBoardBinaryFile(cl_program_name.c_str(), device);

    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkErr(err, __LINE__);

    createKernel("add_bias_helper");
    createKernel("add_bias");
    createKernel("matMul_helper");
    createKernel("matMul");
    createKernel("transpose_helper");
    createKernel("transpose");
    createKernel("relu");
    createKernel("sigmoid");
    createKernel("interaction_cat");
    createKernel("aggregate_cat");
}

void cleanup(){
    for(int i = 0; i < NUM_CMD_QUEUES; i++){
        if (queues[i] != 0)
            clReleaseCommandQueue(queues[i]);
    }

    for (const auto & kernel : kernels) {
        if (kernel.second != 0)
            clReleaseKernel(kernel.second);
    }

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}
