#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>
#include <iterator>
#include <math.h>
#include <vector>
#include <chrono>
#include "hdf5.h"
#include "base_utils.hpp"
//#include "nn_utils.hpp"
#include "shared_utils.hpp"
#include "cl_nn.hpp"
using namespace std;
using namespace std::chrono;
//#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)

vector<vector<float>> array2_2dvec(float* inp, int m, int n){
	vector<vector<float>> x(m, vector<float>(n));
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        x[i][j] = inp[global_idx(i, j, n)];
      }
    }
		return x;
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

void linear(float* x_arr, float* out_bias_arr, vector<vector<float>> weight, vector<float> bias, int x_w, int x_h, int weight_h, char* activation){
	int m = x_w;
	int n = x_h;
	int p = weight_h;
	float weight_arr[n*p];
	float bias_arr[p];
	flatten2dvec2array(weight, weight_arr);
	flatten1dvec2array(bias, bias_arr);
	cl_linear(x_arr, weight_arr, bias_arr, out_bias_arr, m, n, p, activation);
}

void relational_model(float* x_arr, float* x_out, int x_w, int x_h){
	// transpose
	float x_arr_t[x_w*x_h];
	transpose(x_arr, x_arr_t, x_w, x_h);
	// fc layers
	int w0 = RM_WEIGHT_0[0].size();
	int w2 = RM_WEIGHT_2[0].size();
	int w4 = RM_WEIGHT_4[0].size();
	int w6 = RM_WEIGHT_6[0].size();
	float x1[x_h*w0];
	float x2[x_h*w2];
	float x4[x_h*w4];
	linear(x_arr_t, x1, RM_WEIGHT_0, RM_BIAS_0, x_h, x_w, w0, "relu");
	linear(x1, x2, RM_WEIGHT_2, RM_BIAS_2, x_h, w0, w2, "relu");
	linear(x2, x4, RM_WEIGHT_4, RM_BIAS_4, x_h, w2, w4, "relu");
	linear(x4, x_out, RM_WEIGHT_6, RM_BIAS_6, x_h, w4, w6, "sigmoid");
}

void object_model(float* x_arr, float* x_out, int x_w, int x_h){
	// transpose
	float x_arr_t[x_w*x_h];
	transpose(x_arr, x_arr_t, x_w, x_h);
	// fc layers
	int w0 = OM_WEIGHT_0[0].size();
	int w2 = OM_WEIGHT_2[0].size();
	int w4 = OM_WEIGHT_4[0].size();
	float x1[x_h*w0];
	float x2[x_h*w2];
	linear(x_arr_t, x1, OM_WEIGHT_0, OM_BIAS_0, x_h, x_w, w0, "relu");
	linear(x1, x2, OM_WEIGHT_2, OM_BIAS_2, x_h, w0, w2, "relu");
	linear(x2, x_out, OM_WEIGHT_4, OM_BIAS_4, x_h, w2, w4, "None");
}

vector<vector<vector<float>>> forward(vector<vector<vector<float>>> obj,
																			vector<vector<vector<float>>> sr,
																			vector<vector<vector<float>>> rr,
																			vector<vector<vector<float>>> ri){
	vector<vector<float>> predict;
	vector<vector<vector<float>>> predicted;
	int N = data_len;
	//int i = 0;
	for(int i = 0; i < N; i++){

	obj_w = obj[i].size();
	obj_h = obj[i][0].size();
	sr_w = sr[i].size();
	sr_h = sr[i][0].size();
	ri_w = ri[i].size();
	ri_h = ri[i][0].size();
	rr_w = rr[i].size();
	rr_h = rr[i][0].size();
	obj_t_w = obj_h;
	obj_t_h = obj_w;
	sender_w = obj_t_w;
	sender_h = sr_h;
	receiver_w = obj_t_w;
	receiver_h = rr_h;
	term_w = obj_h + obj_h + ri_w;
	term_h = sr_h;
	effect_w = term_w;
	effect_h = RM_WEIGHT_6[0].size();
	effect_receiver_w = rr_w;
	effect_receiver_h = effect_h;
	aggregate_w = obj_t_w + effect_receiver_h;
	aggregate_h = obj_t_h;
	pred_w = OM_WEIGHT_4[0].size();
	pred_h = aggregate_h;
	out_w = term_h;
	out_h = RM_WEIGHT_6[0].size();

	float obj_arr[obj_w*obj_h];
	float obj_arr_t[obj_w*obj_h];
	float sender_arr[sender_w*sender_h];
	float sr_arr[sr_w*sr_h];
	float receiver_arr[receiver_w*receiver_h];
	float rr_arr[rr_w*rr_h];
	float interaction_term_arr[term_w*term_h];
	float ri_arr[ri_w*ri_h];
	float effect_arr[effect_w*effect_h];
	float effect_receiver_arr[effect_receiver_w*effect_receiver_h];
	float effect_receiver_arr_t[effect_receiver_w*effect_receiver_h];
	float agg_arr[aggregate_w*aggregate_h];
	float inf_arr[pred_h*pred_w];
	float predict_arr[pred_h*pred_w];
	float pred_arr[out_w*out_h];

	flatten2dvec2array(obj[i], obj_arr);
	flatten2dvec2array(sr[i], sr_arr);
	flatten2dvec2array(rr[i], rr_arr);
	flatten2dvec2array(ri[i], ri_arr);
	transpose(obj_arr, obj_arr_t, obj_w, obj_h);
	fastMatMul(obj_arr_t, sr_arr, sender_arr, obj_t_w, obj_t_h, sender_h);
	fastMatMul(obj_arr_t, rr_arr, receiver_arr, obj_t_w, obj_t_h, receiver_h);
	interaction_cat(term_w, term_h, sender_w, sender_h, receiver_w, receiver_h,
									ri_w, ri_h, sender_arr, receiver_arr, ri_arr, interaction_term_arr);
	relational_model(interaction_term_arr, effect_arr, term_w, term_h);
	fastMatMul(rr_arr, effect_arr, effect_receiver_arr, rr_w, rr_h, effect_h);
	transpose(effect_receiver_arr, effect_receiver_arr_t, effect_receiver_w, effect_receiver_h);
	aggregate_cat(obj_arr_t, effect_receiver_arr, agg_arr, obj_t_w, obj_t_h, effect_receiver_h, effect_receiver_w);
	object_model(agg_arr, inf_arr, aggregate_w, aggregate_h);
	transpose(inf_arr, predict_arr, pred_h, pred_w);
	fastMatMul(predict_arr, sr_arr, sender_arr, pred_w, pred_h, sender_h);
	fastMatMul(predict_arr, rr_arr, receiver_arr, pred_w, pred_h, receiver_h);
	interaction_cat(term_w, term_h, sender_w, sender_h, receiver_w, receiver_h,
									ri_w, ri_h, sender_arr, receiver_arr, ri_arr, interaction_term_arr);
	relational_model(interaction_term_arr, pred_arr, term_w, term_h);
	predict = array2_2dvec(pred_arr, out_w, out_h);
	int m = predict.size();
	predicted.push_back(predict);
	//for (int j = 0; j < m; j++) {
	//	cout << predict[j][0] << " \n";
	//}

	}
	return predicted;
}
