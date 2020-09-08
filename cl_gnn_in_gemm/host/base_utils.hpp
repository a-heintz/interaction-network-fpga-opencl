#include <iostream>
#include <typeinfo>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <iterator>
#include <math.h>
#include <vector>
#include <chrono>
#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "hdf5.h"
using namespace std;
using namespace std::chrono;

/* char emulator
Time taken by device (avg): Total -- (per event) == 312566 microsecs
Time taken by device (var): Total -- (per event) == 5.5345e+10 microsecs
Time taken by device (avg): FPGA  -- (per event) == 307007 microsecs
Time taken by device (var): FPGA  -- (per event) == 5.5166e+10 microsecs
*/

/* half emulator
Time taken by device (avg): Total -- (per event) == 312566 microsecs
Time taken by device (var): Total -- (per event) == 5.5345e+10 microsecs
Time taken by device (avg): FPGA  -- (per event) == 307007 microsecs
Time taken by device (var): FPGA  -- (per event) == 5.5166e+10 microsecs
*/

/* float emulator
Time taken by device (avg): Total -- (per event) == 316233 microsecs
Time taken by device (var): Total -- (per event) == 5.66119e+10 microsecs
Time taken by device (avg): FPGA  -- (per event) == 310141 microsecs
Time taken by device (var): FPGA  -- (per event) == 5.62709e+10 microsecs
*/

#define dtype char

char* MODEL_FILE;
char* DATA_FILE;

int obj_w;
int obj_h;
int sr_w;
int sr_h;
int ri_w;
int ri_h;
int rr_w;
int rr_h;
int obj_t_w;
int obj_t_h;
int sender_w;
int sender_h;
int receiver_w;
int receiver_h;
int term_w;
int term_h;
int effect_w;
int effect_h;
int effect_receiver_w;
int effect_receiver_h;
int aggregate_w;
int aggregate_h;
int pred_w;
int pred_h;
int out_w;
int out_h;

vector<vector<dtype>> RM_WEIGHT_0_VEC(7, vector<dtype>(250));
vector<vector<dtype>> RM_WEIGHT_2_VEC(250, vector<dtype>(250));
vector<vector<dtype>> RM_WEIGHT_4_VEC(250, vector<dtype>(250));
vector<vector<dtype>> RM_WEIGHT_6_VEC(250, vector<dtype>(1));
vector<vector<dtype>> OM_WEIGHT_0_VEC(4, vector<dtype>(200));
vector<vector<dtype>> OM_WEIGHT_2_VEC(200, vector<dtype>(200));
vector<vector<dtype>> OM_WEIGHT_4_VEC(200, vector<dtype>(3));

vector<dtype> RM_BIAS_0_VEC(250);
vector<dtype> RM_BIAS_2_VEC(250);
vector<dtype> RM_BIAS_4_VEC(250);
vector<dtype> RM_BIAS_6_VEC(1);
vector<dtype> OM_BIAS_0_VEC(200);
vector<dtype> OM_BIAS_2_VEC(200);
vector<dtype> OM_BIAS_4_VEC(3);

dtype RM_WEIGHT_0[7*250];
dtype RM_WEIGHT_2[250*250];
dtype RM_WEIGHT_4[250*250];
dtype RM_WEIGHT_6[250*1];
dtype OM_WEIGHT_0[4*200];
dtype OM_WEIGHT_2[200*200];
dtype OM_WEIGHT_4[200*3];
dtype RM_BIAS_0[250];
dtype RM_BIAS_2[250];
dtype RM_BIAS_4[250];
dtype RM_BIAS_6[1];
dtype OM_BIAS_0[200];
dtype OM_BIAS_2[200];
dtype OM_BIAS_4[3];

int RM_w0 = RM_WEIGHT_0_VEC[0].size();
int RM_w2 = RM_WEIGHT_2_VEC[0].size();
int RM_w4 = RM_WEIGHT_4_VEC[0].size();
int RM_w6 = RM_WEIGHT_6_VEC[0].size();
int OM_w0 = OM_WEIGHT_0_VEC[0].size();
int OM_w2 = OM_WEIGHT_2_VEC[0].size();
int OM_w4 = OM_WEIGHT_4_VEC[0].size();

int data_len;
int data_idx_m[1];
int data_idx_n[1];

cl_mem BUF_obj_arr;
cl_mem BUF_obj_arr_t;
cl_mem BUF_sr_arr;
cl_mem BUF_sender_arr;
cl_mem BUF_rr_arr;
cl_mem BUF_ri_arr;
cl_mem BUF_receiver_arr;
cl_mem BUF_effect_receiver_arr;
cl_mem BUF_effect_receiver_arr_t;
cl_mem BUF_predict_arr;
cl_mem BUF_agg_arr;
cl_mem BUF_inf_arr;
cl_mem BUF_interaction_term_arr;
cl_mem BUF_effect_arr;
cl_mem BUF_pred_arr;
cl_mem BUF_RM_x1;
cl_mem BUF_RM_x2;
cl_mem BUF_RM_x4;
cl_mem BUF_RM_x_arr_t;
cl_mem BUF_OM_x1;
cl_mem BUF_OM_x2;
cl_mem BUF_OM_x_arr_t;

vector<dtype> flatten(const vector<vector<dtype>> &orig)
{
    vector<dtype> ret;
    for(const auto &v: orig)
        ret.insert(ret.end(), v.begin(), v.end());
    return ret;
}

void flatten2dvec2array(vector<vector<dtype>> inp, dtype *inp_arr){
	vector<dtype> flat_inp = flatten(inp);
	int size = flat_inp.size();
	for (int i = 0; i < size; i++) {
		inp_arr[i] = flat_inp[i];
  }
}

void flatten1dvec2array(vector<dtype> inp, dtype *inp_arr){
	int size = inp.size();
	for (int i = 0; i < size; i++) {
		inp_arr[i] = inp[i];
  }
}

string toString(int &i){
   stringstream ss;
   ss << i;
   return ss.str();
}
// THIS IS WHERE I WANT TO CAST float to dtype
vector<vector<dtype>> readH5_2_vec_2d(vector<vector<dtype>> vec, const char* str, hid_t model_file){
	int m = vec.size();
	int n = vec[0].size();
	float x[m][n];
	const hid_t id = H5Dopen2(model_file, str, H5P_DEFAULT);
	H5Dread(id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
	H5Dclose(id);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			vec[i][j] = x[i][j];
		}
	}
	return vec;
}
// THIS IS WHERE I WANT TO CAST float to dtype
vector<dtype> readH5_2_vec_1d(vector<dtype> vec, const char* str, hid_t model_file){
	int m = vec.size();
	float x[m];
	const hid_t id = H5Dopen2(model_file, str, H5P_DEFAULT);
	H5Dread(id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x);
	H5Dclose(id);
	for(int i = 0; i < m; i++){
			vec[i] = x[i];
	}
	return vec;
}

void load_model() {
	const hid_t model_file = H5Fopen(MODEL_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);

	RM_WEIGHT_0_VEC = readH5_2_vec_2d(RM_WEIGHT_0_VEC, "relational_model.layers.0.weight", model_file);
	RM_WEIGHT_2_VEC = readH5_2_vec_2d(RM_WEIGHT_2_VEC, "relational_model.layers.2.weight", model_file);
	RM_WEIGHT_4_VEC = readH5_2_vec_2d(RM_WEIGHT_4_VEC, "relational_model.layers.4.weight", model_file);
	RM_WEIGHT_6_VEC = readH5_2_vec_2d(RM_WEIGHT_6_VEC, "relational_model.layers.6.weight", model_file);

	OM_WEIGHT_0_VEC = readH5_2_vec_2d(OM_WEIGHT_0_VEC, "object_model.layers.0.weight", model_file);
	OM_WEIGHT_2_VEC = readH5_2_vec_2d(OM_WEIGHT_2_VEC, "object_model.layers.2.weight", model_file);
	OM_WEIGHT_4_VEC = readH5_2_vec_2d(OM_WEIGHT_4_VEC, "object_model.layers.4.weight", model_file);

	RM_BIAS_0_VEC = readH5_2_vec_1d(RM_BIAS_0_VEC, "relational_model.layers.0.bias", model_file);
	RM_BIAS_2_VEC = readH5_2_vec_1d(RM_BIAS_2_VEC, "relational_model.layers.2.bias", model_file);
	RM_BIAS_4_VEC = readH5_2_vec_1d(RM_BIAS_4_VEC, "relational_model.layers.4.bias", model_file);
	RM_BIAS_6_VEC = readH5_2_vec_1d(RM_BIAS_6_VEC, "relational_model.layers.6.bias", model_file);
	OM_BIAS_0_VEC = readH5_2_vec_1d(OM_BIAS_0_VEC, "object_model.layers.0.bias", model_file);
	OM_BIAS_2_VEC = readH5_2_vec_1d(OM_BIAS_2_VEC, "object_model.layers.2.bias", model_file);
	OM_BIAS_4_VEC = readH5_2_vec_1d(OM_BIAS_4_VEC, "object_model.layers.4.bias", model_file);

	H5Fclose(model_file);

  flatten2dvec2array(RM_WEIGHT_0_VEC, RM_WEIGHT_0);
	flatten2dvec2array(RM_WEIGHT_2_VEC, RM_WEIGHT_2);
	flatten2dvec2array(RM_WEIGHT_4_VEC, RM_WEIGHT_4);
	flatten2dvec2array(RM_WEIGHT_6_VEC, RM_WEIGHT_6);
	flatten2dvec2array(OM_WEIGHT_0_VEC, OM_WEIGHT_0);
	flatten2dvec2array(OM_WEIGHT_2_VEC, OM_WEIGHT_2);
	flatten2dvec2array(OM_WEIGHT_4_VEC, OM_WEIGHT_4);
	flatten1dvec2array(RM_BIAS_0_VEC, RM_BIAS_0);
	flatten1dvec2array(RM_BIAS_2_VEC, RM_BIAS_2);
	flatten1dvec2array(RM_BIAS_4_VEC, RM_BIAS_4);
	flatten1dvec2array(RM_BIAS_6_VEC, RM_BIAS_6);
	flatten1dvec2array(OM_BIAS_0_VEC, OM_BIAS_0);
	flatten1dvec2array(OM_BIAS_2_VEC, OM_BIAS_2);
	flatten1dvec2array(OM_BIAS_4_VEC, OM_BIAS_4);
}
// THIS IS WHERE I WANT TO CAST float to dtype
vector<vector<dtype>> load_data(hid_t data_file, string sec, int data_len)
{
    //cout << "hi 1 \n \n";
		const hid_t id0 = H5Dopen2(data_file, (sec + "_shape_0").c_str(), H5P_DEFAULT);
		H5Dread(id0, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_idx_m);
		H5Dclose(id0);
    //cout << "hi 2 \n \n";
		const hid_t id1 = H5Dopen2(data_file, (sec + "_shape_1").c_str(), H5P_DEFAULT);
		H5Dread(id1, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_idx_n);
		H5Dclose(id1);
    //cout << "hi 3 \n \n";
		vector<vector<dtype>> dat;
		int m = data_idx_m[0];
		int n = data_idx_n[0];
		dat.resize(m);
    //cout << "hi 4 \n \n";
		float dat_temp[m][n];
		string text = sec;
		const hid_t id = H5Dopen2(data_file, text.c_str(), H5P_DEFAULT);
		H5Dread(id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dat_temp);
		H5Dclose(id);
    //cout << "hi 5 \n \n";
		for(int j = 0; j < m; j++){
			dat[j].resize(n);
			for(int k = 0; k < n; k++){
				dat[j][k] = dat_temp[j][k];
			}
		}
    //cout << "hi 6 \n \n";
		return dat;
}





/* half data type
transpose: OpenCl Execution time is: 117.929 microseconds
buf_fastMatMul: OpenCl Execution time is: 126.003 microseconds
buf_fastMatMul: OpenCl Execution time is: 138.79 microseconds
interaction_cat: OpenCl Execution time is: 115.233 microseconds
transpose: OpenCl Execution time is: 119.818 microseconds
linear: OpenCl Execution time is: 115.347 microseconds
linear: OpenCl Execution time is: 237.285 microseconds
linear: OpenCl Execution time is: 221.846 microseconds
linear: OpenCl Execution time is: 119.779 microseconds
buf_fastMatMul: OpenCl Execution time is: 115.032 microseconds
transpose: OpenCl Execution time is: 115.482 microseconds
aggregate_cat: OpenCl Execution time is: 115.007 microseconds
transpose: OpenCl Execution time is: 133.232 microseconds
linear: OpenCl Execution time is: 221.794 microseconds
linear: OpenCl Execution time is: 222.134 microseconds
linear: OpenCl Execution time is: 114.957 microseconds
transpose: OpenCl Execution time is: 114.983 microseconds
buf_fastMatMul: OpenCl Execution time is: 134.073 microseconds
buf_fastMatMul: OpenCl Execution time is: 98.183 microseconds
interaction_cat: OpenCl Execution time is: 64.134 microseconds
transpose: OpenCl Execution time is: 64.492 microseconds
linear: OpenCl Execution time is: 171.289 microseconds
linear: OpenCl Execution time is: 180.391 microseconds
linear: OpenCl Execution time is: 168.435 microseconds
linear: OpenCl Execution time is: 138.04 microseconds
Time taken by function SWI: 12190 microsecs
kernelL 3483.688 microsecs
*/

/* float data type
transpose: OpenCl Execution time is: 122.484 microseconds
buf_fastMatMul: OpenCl Execution time is: 125.318 microseconds
buf_fastMatMul: OpenCl Execution time is: 113.989 microseconds
interaction_cat: OpenCl Execution time is: 76.097 microseconds
transpose: OpenCl Execution time is: 115.332 microseconds
linear: OpenCl Execution time is: 114.392 microseconds
linear: OpenCl Execution time is: 230.642 microseconds
linear: OpenCl Execution time is: 221.25 microseconds
linear: OpenCl Execution time is: 134.062 microseconds
buf_fastMatMul: OpenCl Execution time is: 126.51 microseconds
transpose: OpenCl Execution time is: 124.887 microseconds
aggregate_cat: OpenCl Execution time is: 115.148 microseconds
transpose: OpenCl Execution time is: 115.037 microseconds
linear: OpenCl Execution time is: 115.238 microseconds
linear: OpenCl Execution time is: 140.317 microseconds
linear: OpenCl Execution time is: 115.136 microseconds
transpose: OpenCl Execution time is: 117.358 microseconds
buf_fastMatMul: OpenCl Execution time is: 80.76 microseconds
buf_fastMatMul: OpenCl Execution time is: 62.413 microseconds
interaction_cat: OpenCl Execution time is: 63.281 microseconds
transpose: OpenCl Execution time is: 63.317 microseconds
linear: OpenCl Execution time is: 170.823 microseconds
linear: OpenCl Execution time is: 180.1 microseconds
linear: OpenCl Execution time is: 179.478 microseconds
linear: OpenCl Execution time is: 138.337 microseconds
Time taken by function SWI: 13251 microsecs
kernel: 3161.706 microsecs
*/
