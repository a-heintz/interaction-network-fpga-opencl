#include "nn.hpp"
#include "cl_utils.hpp"
using namespace std;
using namespace std::chrono;

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
		print_bool = atoi(argv[4]);

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
		if(print_bool == 0){
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
		}
		// cleanup
		cleanup();
		// free all resources in host
		return 0;
}
