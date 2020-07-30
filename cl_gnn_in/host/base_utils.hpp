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

#define MODEL_FILE    "../data/model_weights.hdf5"
#define DATA_FILE     "../data/test.hdf5"

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

vector<vector<float>> RM_WEIGHT_0(7, vector<float>(250));
vector<vector<float>> RM_WEIGHT_2(250, vector<float>(250));
vector<vector<float>> RM_WEIGHT_4(250, vector<float>(250));
vector<vector<float>> RM_WEIGHT_6(250, vector<float>(1));
vector<vector<float>> OM_WEIGHT_0(4, vector<float>(200));
vector<vector<float>> OM_WEIGHT_2(200, vector<float>(200));
vector<vector<float>> OM_WEIGHT_4(200, vector<float>(3));

vector<float> RM_BIAS_0(250);
vector<float> RM_BIAS_2(250);
vector<float> RM_BIAS_4(250);
vector<float> RM_BIAS_6(1);
vector<float> OM_BIAS_0(200);
vector<float> OM_BIAS_2(200);
vector<float> OM_BIAS_4(3);

int data_len = 10;
int data_idx_m[100];
int data_idx_n[100];

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

	RM_WEIGHT_0 = readH5_2_vec_2d(RM_WEIGHT_0, "relational_model.layers.0.weight", model_file);
	RM_WEIGHT_2 = readH5_2_vec_2d(RM_WEIGHT_2, "relational_model.layers.2.weight", model_file);
	RM_WEIGHT_4 = readH5_2_vec_2d(RM_WEIGHT_4, "relational_model.layers.4.weight", model_file);
	RM_WEIGHT_6 = readH5_2_vec_2d(RM_WEIGHT_6, "relational_model.layers.6.weight", model_file);

	OM_WEIGHT_0 = readH5_2_vec_2d(OM_WEIGHT_0, "object_model.layers.0.weight", model_file);
	OM_WEIGHT_2 = readH5_2_vec_2d(OM_WEIGHT_2, "object_model.layers.2.weight", model_file);
	OM_WEIGHT_4 = readH5_2_vec_2d(OM_WEIGHT_4, "object_model.layers.4.weight", model_file);

	RM_BIAS_0 = readH5_2_vec_1d(RM_BIAS_0, "relational_model.layers.0.bias", model_file);
	RM_BIAS_2 = readH5_2_vec_1d(RM_BIAS_2, "relational_model.layers.2.bias", model_file);
	RM_BIAS_4 = readH5_2_vec_1d(RM_BIAS_4, "relational_model.layers.4.bias", model_file);
	RM_BIAS_6 = readH5_2_vec_1d(RM_BIAS_6, "relational_model.layers.6.bias", model_file);
	OM_BIAS_0 = readH5_2_vec_1d(OM_BIAS_0, "object_model.layers.0.bias", model_file);
	OM_BIAS_2 = readH5_2_vec_1d(OM_BIAS_2, "object_model.layers.2.bias", model_file);
	OM_BIAS_4 = readH5_2_vec_1d(OM_BIAS_4, "object_model.layers.4.bias", model_file);

	H5Fclose(model_file);
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
