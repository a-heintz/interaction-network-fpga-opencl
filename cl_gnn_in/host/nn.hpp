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

vector<vector<float>> array2_2dvec(float* inp, int m, int n){
	vector<vector<float>> x(m, vector<float>(n));
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        x[i][j] = inp[global_idx(i, j, n)];
      }
    }
		return x;
}

void relational_model(cl_mem BUF_x_arr, cl_mem BUF_x_out, int x_w, int x_h, int RM_w0, int RM_w2, int RM_w4, int RM_w6,
											cl_mem BUF_RM_WEIGHT_0, cl_mem BUF_RM_WEIGHT_2, cl_mem BUF_RM_WEIGHT_4, cl_mem BUF_RM_WEIGHT_6,
											cl_mem BUF_RM_BIAS_0, cl_mem BUF_RM_BIAS_2, cl_mem BUF_RM_BIAS_4, cl_mem BUF_RM_BIAS_6,
											cl_mem BUF_RM_x1, cl_mem BUF_RM_x2, cl_mem BUF_RM_x4, cl_mem BUF_RM_x_arr_t)
{
	transpose(BUF_x_arr, BUF_RM_x_arr_t, x_w, x_h);
	cl_linear(BUF_RM_x_arr_t, BUF_RM_WEIGHT_0, BUF_RM_BIAS_0, BUF_RM_x1, x_h, x_w, RM_w0, "relu");
	cl_linear(BUF_RM_x1, BUF_RM_WEIGHT_2, BUF_RM_BIAS_2, BUF_RM_x2, x_h, RM_w0, RM_w2, "relu");
	cl_linear(BUF_RM_x2, BUF_RM_WEIGHT_4, BUF_RM_BIAS_4, BUF_RM_x4, x_h, RM_w2, RM_w4, "relu");
	cl_linear(BUF_RM_x4, BUF_RM_WEIGHT_6, BUF_RM_BIAS_6, BUF_x_out, x_h, RM_w4, RM_w6, "sigmoid");
}

void object_model(cl_mem BUF_x_arr, cl_mem BUF_x_out, int x_w, int x_h, int OM_w0, int OM_w2, int OM_w4,
									cl_mem BUF_OM_WEIGHT_0, cl_mem BUF_OM_WEIGHT_2, cl_mem BUF_OM_WEIGHT_4,
									cl_mem BUF_OM_BIAS_0, cl_mem BUF_OM_BIAS_2, cl_mem BUF_OM_BIAS_4,
									cl_mem BUF_OM_x_arr_t, cl_mem BUF_OM_x1, cl_mem BUF_OM_x2)
{
	transpose(BUF_x_arr, BUF_OM_x_arr_t, x_w, x_h);
	cl_linear(BUF_OM_x_arr_t, BUF_OM_WEIGHT_0, BUF_OM_BIAS_0, BUF_OM_x1, x_h, x_w, OM_w0, "relu");
	cl_linear(BUF_OM_x1, BUF_OM_WEIGHT_2, BUF_OM_BIAS_2, BUF_OM_x2, x_h, OM_w0, OM_w2, "relu");
	cl_linear(BUF_OM_x2, BUF_OM_WEIGHT_4, BUF_OM_BIAS_4, BUF_x_out, x_h, OM_w2, OM_w4, "None");
}

vector<vector<vector<float>>> ndr_forward(vector<vector<vector<float>>> obj,
																			vector<vector<vector<float>>> sr,
																			vector<vector<vector<float>>> rr,
																			vector<vector<vector<float>>> ri){
	vector<vector<float>> predict;
	vector<vector<vector<float>>> predicted;
	int N = data_len;
	int i = 0;
	//for(int i = 0; i < N; i++){

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
	effect_h = RM_WEIGHT_6_VEC[0].size();
	effect_receiver_w = rr_w;
	effect_receiver_h = effect_h;
	aggregate_w = obj_t_w + effect_receiver_h;
	aggregate_h = obj_t_h;
	pred_w = OM_WEIGHT_4_VEC[0].size();
	pred_h = aggregate_h;
	out_w = term_h;
	out_h = RM_WEIGHT_6_VEC[0].size();
	int RM_w0 = RM_WEIGHT_0_VEC[0].size();
	int RM_w2 = RM_WEIGHT_2_VEC[0].size();
	int RM_w4 = RM_WEIGHT_4_VEC[0].size();
	int RM_w6 = RM_WEIGHT_6_VEC[0].size();
	int OM_w0 = OM_WEIGHT_0_VEC[0].size();
	int OM_w2 = OM_WEIGHT_2_VEC[0].size();
	int OM_w4 = OM_WEIGHT_4_VEC[0].size();

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
	float RM_x_arr_t[term_w*term_h];
	float RM_x1[term_h*RM_w0];
	float RM_x2[term_h*RM_w2];
	float RM_x4[term_h*RM_w4];
	float OM_x_arr_t[aggregate_w*aggregate_h];
	float OM_x1[aggregate_h*OM_w0];
	float OM_x2[aggregate_h*OM_w2];

	flatten2dvec2array(obj[i], obj_arr);
	flatten2dvec2array(sr[i], sr_arr);
	flatten2dvec2array(rr[i], rr_arr);
	flatten2dvec2array(ri[i], ri_arr);

	cl_int status;
	cl_mem BUF_obj_arr = create_intermediate_buffer(obj_arr, obj_w*obj_h, status);
	cl_mem BUF_obj_arr_t = create_intermediate_buffer(obj_arr_t, obj_w*obj_h, status);
	cl_mem BUF_sr_arr = create_intermediate_buffer(sr_arr, pred_h*sender_h, status);
	cl_mem BUF_sender_arr = create_intermediate_buffer(sender_arr, pred_w*sender_h, status);
	cl_mem BUF_rr_arr = create_intermediate_buffer(rr_arr, pred_h*receiver_h, status);
	cl_mem BUF_ri_arr = create_intermediate_buffer(ri_arr, ri_w*ri_h, status);
	cl_mem BUF_receiver_arr = create_intermediate_buffer(receiver_arr, pred_w*receiver_h, status);
	cl_mem BUF_effect_receiver_arr = create_intermediate_buffer(effect_receiver_arr, rr_w*effect_h, status);
	cl_mem BUF_effect_receiver_arr_t = create_intermediate_buffer(effect_receiver_arr_t, effect_receiver_w*effect_receiver_h, status);
	cl_mem BUF_predict_arr = create_intermediate_buffer(predict_arr, pred_h*pred_w, status);
	cl_mem BUF_agg_arr = create_intermediate_buffer(agg_arr, aggregate_w*aggregate_h, status);
	cl_mem BUF_inf_arr = create_intermediate_buffer(inf_arr, pred_h*pred_w, status);
	cl_mem BUF_interaction_term_arr = create_intermediate_buffer(interaction_term_arr, term_w*term_h, status);
	cl_mem BUF_effect_arr = create_intermediate_buffer(effect_arr, rr_h*effect_h, status);
	cl_mem BUF_pred_arr = create_intermediate_buffer(pred_arr, out_w*out_h, status);
	cl_mem BUF_RM_x1 = create_intermediate_buffer(RM_x1, term_h*RM_w0, status);
	cl_mem BUF_RM_x2 = create_intermediate_buffer(RM_x2, term_h*RM_w2, status);
	cl_mem BUF_RM_x4 = create_intermediate_buffer(RM_x4, term_h*RM_w4, status);
	cl_mem BUF_RM_x_arr_t = create_intermediate_buffer(RM_x_arr_t, term_w*term_h, status);
	cl_mem BUF_OM_x1 = create_intermediate_buffer(OM_x1, aggregate_h*OM_w0, status);
	cl_mem BUF_OM_x2 = create_intermediate_buffer(OM_x2, aggregate_h*OM_w2, status);
	cl_mem BUF_OM_x_arr_t = create_intermediate_buffer(OM_x_arr_t, aggregate_w*aggregate_h, status);
	cl_mem BUF_RM_WEIGHT_0 = create_intermediate_buffer(RM_WEIGHT_0, RM_w0*RM_WEIGHT_0_VEC.size(), status);
	cl_mem BUF_RM_WEIGHT_2 = create_intermediate_buffer(RM_WEIGHT_2, RM_w2*RM_WEIGHT_2_VEC.size(), status);
	cl_mem BUF_RM_WEIGHT_4 = create_intermediate_buffer(RM_WEIGHT_4, RM_w4*RM_WEIGHT_4_VEC.size(), status);
	cl_mem BUF_RM_WEIGHT_6 = create_intermediate_buffer(RM_WEIGHT_6, RM_w6*RM_WEIGHT_6_VEC.size(), status);
	cl_mem BUF_OM_WEIGHT_0 = create_intermediate_buffer(OM_WEIGHT_0, OM_w0*OM_WEIGHT_0_VEC.size(), status);
	cl_mem BUF_OM_WEIGHT_2 = create_intermediate_buffer(OM_WEIGHT_2, OM_w2*OM_WEIGHT_2_VEC.size(), status);
	cl_mem BUF_OM_WEIGHT_4 = create_intermediate_buffer(OM_WEIGHT_4, OM_w4*OM_WEIGHT_4_VEC.size(), status);
	cl_mem BUF_RM_BIAS_0 = create_intermediate_buffer(RM_BIAS_0, RM_BIAS_0_VEC.size(), status);
	cl_mem BUF_RM_BIAS_2 = create_intermediate_buffer(RM_BIAS_2, RM_BIAS_2_VEC.size(), status);
	cl_mem BUF_RM_BIAS_4 = create_intermediate_buffer(RM_BIAS_4, RM_BIAS_4_VEC.size(), status);
	cl_mem BUF_RM_BIAS_6 = create_intermediate_buffer(RM_BIAS_6, RM_BIAS_6_VEC.size(), status);
	cl_mem BUF_OM_BIAS_0 = create_intermediate_buffer(OM_BIAS_0, OM_BIAS_0_VEC.size(), status);
	cl_mem BUF_OM_BIAS_2 = create_intermediate_buffer(OM_BIAS_2, OM_BIAS_2_VEC.size(), status);
	cl_mem BUF_OM_BIAS_4 = create_intermediate_buffer(OM_BIAS_4, OM_BIAS_4_VEC.size(), status);

	transpose(BUF_obj_arr, BUF_obj_arr_t, obj_w, obj_h);
	buf_fastMatMul(BUF_obj_arr_t, BUF_sr_arr, BUF_sender_arr, obj_t_w, obj_t_h, sender_h);
	buf_fastMatMul(BUF_obj_arr_t, BUF_rr_arr, BUF_receiver_arr, obj_t_w, obj_t_h, receiver_h);
	interaction_cat(term_w, term_h, sender_w, sender_h, receiver_w, receiver_h,
		ri_w, ri_h, BUF_sender_arr, BUF_receiver_arr, BUF_ri_arr, BUF_interaction_term_arr);
	relational_model(BUF_interaction_term_arr, BUF_effect_arr, term_w, term_h, RM_w0, RM_w2, RM_w4, RM_w6,
		BUF_RM_WEIGHT_0, BUF_RM_WEIGHT_2, BUF_RM_WEIGHT_4, BUF_RM_WEIGHT_6,
		BUF_RM_BIAS_0, BUF_RM_BIAS_2, BUF_RM_BIAS_4, BUF_RM_BIAS_6,
		BUF_RM_x1, BUF_RM_x2, BUF_RM_x4, BUF_RM_x_arr_t);
	buf_fastMatMul(BUF_rr_arr, BUF_effect_arr, BUF_effect_receiver_arr, rr_w, rr_h, effect_h);
	transpose(BUF_effect_receiver_arr, BUF_effect_receiver_arr_t, effect_receiver_w, effect_receiver_h);
	aggregate_cat(BUF_obj_arr_t, BUF_effect_receiver_arr_t, BUF_agg_arr, obj_t_w, obj_t_h, effect_receiver_h, effect_receiver_w);
	object_model(BUF_agg_arr, BUF_inf_arr, aggregate_w, aggregate_h, OM_w0, OM_w2, OM_w4,
		BUF_OM_WEIGHT_0, BUF_OM_WEIGHT_2, BUF_OM_WEIGHT_4, BUF_OM_BIAS_0, BUF_OM_BIAS_2, BUF_OM_BIAS_4,
		BUF_OM_x_arr_t, BUF_OM_x1, BUF_OM_x2);
	transpose(BUF_inf_arr, BUF_predict_arr, pred_h, pred_w);
	buf_fastMatMul(BUF_predict_arr, BUF_sr_arr, BUF_sender_arr, pred_w, pred_h, sender_h);
	buf_fastMatMul(BUF_predict_arr, BUF_rr_arr, BUF_receiver_arr, pred_w, pred_h, receiver_h);
	interaction_cat(term_w, term_h, sender_w, sender_h, receiver_w, receiver_h,
		ri_w, ri_h, BUF_sender_arr, BUF_receiver_arr, BUF_ri_arr, BUF_interaction_term_arr);
	relational_model(BUF_interaction_term_arr, BUF_pred_arr, term_w, term_h, RM_w0, RM_w2, RM_w4, RM_w6,
		BUF_RM_WEIGHT_0, BUF_RM_WEIGHT_2, BUF_RM_WEIGHT_4, BUF_RM_WEIGHT_6,
		BUF_RM_BIAS_0, BUF_RM_BIAS_2, BUF_RM_BIAS_4, BUF_RM_BIAS_6,
		BUF_RM_x1, BUF_RM_x2, BUF_RM_x4, BUF_RM_x_arr_t);
	read_out_buffer(BUF_pred_arr, pred_arr, out_w*out_h, status);
	predict = array2_2dvec(pred_arr, out_w, out_h);
	int m = predict.size();
	predicted.push_back(predict);
	for (int j = 0; j < m; j++) {
		cout << predict[j][0] << " \n";
	}

	//}
	return predicted;
}
