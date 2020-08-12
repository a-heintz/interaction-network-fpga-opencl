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
#include "hdf5.h"
using namespace std;
using namespace std::chrono;

#define MODEL_FILE    "../data/model_weights_LP_5.hdf5"
#define DATA_FILE     "../data/test_LP_5.hdf5"

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

vector<vector<float>> RM_WEIGHT_0_VEC(7, vector<float>(250));
vector<vector<float>> RM_WEIGHT_2_VEC(250, vector<float>(250));
vector<vector<float>> RM_WEIGHT_4_VEC(250, vector<float>(250));
vector<vector<float>> RM_WEIGHT_6_VEC(250, vector<float>(1));
vector<vector<float>> OM_WEIGHT_0_VEC(4, vector<float>(200));
vector<vector<float>> OM_WEIGHT_2_VEC(200, vector<float>(200));
vector<vector<float>> OM_WEIGHT_4_VEC(200, vector<float>(3));

vector<float> RM_BIAS_0_VEC(250);
vector<float> RM_BIAS_2_VEC(250);
vector<float> RM_BIAS_4_VEC(250);
vector<float> RM_BIAS_6_VEC(1);
vector<float> OM_BIAS_0_VEC(200);
vector<float> OM_BIAS_2_VEC(200);
vector<float> OM_BIAS_4_VEC(3);

float RM_WEIGHT_0[7*250];
float RM_WEIGHT_2[250*250];
float RM_WEIGHT_4[250*250];
float RM_WEIGHT_6[250*1];
float OM_WEIGHT_0[4*200];
float OM_WEIGHT_2[200*200];
float OM_WEIGHT_4[200*3];
float RM_BIAS_0[250];
float RM_BIAS_2[250];
float RM_BIAS_4[250];
float RM_BIAS_6[1];
float OM_BIAS_0[200];
float OM_BIAS_2[200];
float OM_BIAS_4[3];

int data_len = 100;
int data_idx_m[1000];
int data_idx_n[1000];

vector<float> flatten(const vector<vector<float>> &orig)
{
    vector<float> ret;
    for(const auto &v: orig)
        ret.insert(ret.end(), v.begin(), v.end());
    return ret;
}

void flatten2dvec2array(vector<vector<float>> inp, float *inp_arr){
	vector<float> flat_inp = flatten(inp);
	int size = flat_inp.size();
	for (int i = 0; i < size; i++) {
		inp_arr[i] = flat_inp[i];
  }
}

void flatten1dvec2array(vector<float> inp, float *inp_arr){
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

vector<vector<float>> readH5_2_vec_2d(vector<vector<float>> vec, const char* str, hid_t model_file){
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

vector<float> readH5_2_vec_1d(vector<float> vec, const char* str, hid_t model_file){
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

vector<vector<vector<float>>> load_data(hid_t data_file, string sec, int data_len)
{
		const hid_t id0 = H5Dopen2(data_file, (sec + "shape_0_i").c_str(), H5P_DEFAULT);
		H5Dread(id0, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_idx_m);
		H5Dclose(id0);

		const hid_t id1 = H5Dopen2(data_file, (sec + "shape_1_i").c_str(), H5P_DEFAULT);
		H5Dread(id1, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_idx_n);
		H5Dclose(id1);

		vector<vector<vector<float>>> dat;
		dat.resize(data_len);
		for(int i = 0; i < data_len; i++){
			int m = data_idx_m[i];
			int n = data_idx_n[i];
			dat[i].resize(m);

			float dat_temp[m][n];
			string text = sec + toString(i);
			const hid_t id = H5Dopen2(data_file, text.c_str(), H5P_DEFAULT);
			H5Dread(id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dat_temp);
			H5Dclose(id);

			for(int j = 0; j < m; j++){
				dat[i][j].resize(n);
				for(int k = 0; k < n; k++){
					dat[i][j][k] = dat_temp[j][k];
				}
			}
		}
		return dat;
}
