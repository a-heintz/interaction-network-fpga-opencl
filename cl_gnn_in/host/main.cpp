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
int main(int argc, char **argv)
{
		Options options(argc, argv);
		initializeOpenCLParameters();
		// load data
		const hid_t data_file = H5Fopen(DATA_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
		string sec = "obj_";
		vector<vector<vector<float>>> obj = load_data(data_file, sec, data_len);
		sec = "sr_";
		vector<vector<vector<float>>> sr = load_data(data_file, sec, data_len);
		sec = "rr_";
		vector<vector<vector<float>>> rr = load_data(data_file, sec, data_len);
		sec = "ri_";
		vector<vector<vector<float>>> ri = load_data(data_file, sec, data_len);
		// load model
		load_model();
		// forward operation
		auto start = high_resolution_clock::now();
		vector<vector<vector<float>>> predicted = ndr_forward(obj, sr, rr, ri);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		cout << "Time taken by function SWI: "
		     << duration.count() << " microsecs \n";


		//start = high_resolution_clock::now();
		//predicted = ndr_forward(obj, sr, rr, ri);
		//stop = high_resolution_clock::now();
		//duration = duration_cast<microseconds>(stop - start);
		//cout << "Time taken by function NDRange: "
		//		<< duration.count() << " microsecs \n";



		// cleanup
		cleanup();
		// free all resources in host
		return 0;
}
