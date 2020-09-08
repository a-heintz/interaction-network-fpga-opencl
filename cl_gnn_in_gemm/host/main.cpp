#include "nn.hpp"
#include "cl_utils.hpp"
using namespace std;
using namespace std::chrono;

// all of the following is for the first graph only
// 14703417 microsec on emulator --  all 2d
// 687033 microsec on emulator -- all 1d
// 1269062 microsec on fpga -- all 2d
// 684228 microsec on emulator -- 1d, 1d matmul
// 676311 microsec on emulator -- 1d, 2d matmul
// 465900 microsec on emulator -- create linear kernel
// 254952 microsec on emulator -- create linear + relu kernel and linear + sigmoid kernel
// >> this implementation lost precision more significantly than the most recent ones
// 64071 microsec on fpga -- 1d, 2d matmul, linear + relu + sigmoid integrated kernel

// 26230 microsec on fpga -- added unrolling
// 4287288 microsec on fpga -- data_len = 100

// 51009930 microsec on emulator -- data_len = 100
// 85884402 microsec on fpga -- data_len = 1000

// 259761 microsec -- ndr emulator -- data_len = 1

// 33300 microsec -- swi emulator -- data_len = 1
// 397324 microsec -- swi fpga -- data_len = 1
// 25306 microsec -- ndr fpga -- data_len = 1
// 3582400 microsec -- ndr fpga -- data_len = 100
// 79918738 microsec -- swi fpga -- data_len = 100


// 262906 microsec -- ndr emulator -- data_len = 1 -- added copy to local in matrixMul
// 287858 microsec -- ndr emulator -- data_len = 1 -- added double buffering
// 3393314 microsec -- ndr emulator -- data_len = 1 -- added padding & GEMM to linear
// 22585 microsec -- ndr fpga -- data_len = 1 -- added padding & GEMM to linear
// 25306 microsec -- ndr fpga -- data_len = 1 -- old


// ------ emulator runtime breakdown -------
// Total (microsecs): 91117
//FPGA (microsecs): 79481.603
//CPU (microsecs): 1572

float avg(float* in, int size){
	float out = 0.0f;
	for(int i = 0; i<size; i++){
		out += in[i];
	}
	out /= size;
	return out;
}

float var(float* in, int size){
	float mean = avg(in, size);
	float out = 0.0f;
	for(int i = 0; i<size; i++){
		out += (in[i] - mean) * (in[i] - mean);
	}
	out /= size;
	return out;
}

int main(int argc, char** argv)
{
		MODEL_FILE = argv[1];
		DATA_FILE = argv[2];
		data_len = atoi(argv[3]);

		//Options options(argc, argv);
		initializeOpenCLParameters();
		// load model
		load_model();
		// load data

		float elapsed_times[data_len];
		float durations[data_len];
		float elapsed_time;
		for(int i = 0; i < data_len; i++){
			string i_char = string(DATA_FILE) + "/" + to_string(i) + ".hdf5";
			//cout << i_char << "\n \n \n \n";
			const char* DATA_FILE_i = i_char.c_str();
			//cout << DATA_FILE_i << "\n \n \n \n";
			const hid_t data_file = H5Fopen(DATA_FILE_i, H5F_ACC_RDONLY, H5P_DEFAULT);
			string sec = "obj";
			vector<vector<dtype>> obj = load_data(data_file, sec, data_len);
			sec = "sr";
			vector<vector<dtype>> sr = load_data(data_file, sec, data_len);
			sec = "rr";
			vector<vector<dtype>> rr = load_data(data_file, sec, data_len);
			sec = "ri";
			vector<vector<dtype>> ri = load_data(data_file, sec, data_len);

			cl_int status;
			// forward operation
			elapsed_time = 0.0f;
			auto start = high_resolution_clock::now();
			vector<vector<dtype>> predicted = ndr_forward(obj, sr, rr, ri, &elapsed_time);
			auto stop = high_resolution_clock::now();
			float duration = duration_cast<microseconds>(stop - start).count();
			elapsed_times[i] = elapsed_time;
			durations[i] = duration;
		}
		float avg_duration = avg(durations, data_len);
		float var_duration = var(durations, data_len);
		float avg_elapsed_time = avg(elapsed_times, data_len);
		float var_elapsed_time = var(elapsed_times, data_len);
		cout << "Time taken by device (avg): Total -- (per event) == "
		     << avg_duration << " microsecs \n";
		cout << "Time taken by device (var): Total -- (per event) == "
		  	 << var_duration << " microsecs \n";
		cout << "Time taken by device (avg): FPGA  -- (per event) == "
			   << avg_elapsed_time << " microsecs \n";
	 	cout << "Time taken by device (var): FPGA  -- (per event) == "
			   << var_elapsed_time << " microsecs \n";

		// cleanup
		cleanup();
		// free all resources in host
		return 0;
}
