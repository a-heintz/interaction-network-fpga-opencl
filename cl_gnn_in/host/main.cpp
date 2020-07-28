#include "nn.hpp"
#include "cl_utils.hpp"
using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
	Options options(argc, argv);

	initializeOpenCLParameters();
	// create variables for all dimensions
	// calculate length of all arrays
	// create buffers for everything...
	// allocate space for everything in host
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
	vector<vector<vector<float>>> predicted = forward(obj, sr, rr, ri);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by function: "
         << duration.count() << " microsecs \n";
	// cleanup
	cleanup();
	// free all resources in host








	return 0;
}
