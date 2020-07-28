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
#include "nn_utils.hpp"
#include "cl_nn.hpp"
using namespace std;
using namespace std::chrono;

vector<vector<float>> linear(vector<vector<float>> x, vector<vector<float>> weight, vector<float> bias){
	x = matmul(x, weight);
	x = add_bias(x, bias);
	return x;
}

vector<vector<float>> relational_model(vector<vector<float>> x){
	x = transpose(x);
	x = relu( linear(x, RM_WEIGHT_0, RM_BIAS_0) );
	x = relu( linear(x, RM_WEIGHT_2, RM_BIAS_2) );
	x = relu(	linear(x, RM_WEIGHT_4, RM_BIAS_4) );
	x = sigmoid( linear(x, RM_WEIGHT_6, RM_BIAS_6) );

	return x;
}

vector<vector<float>> object_model(vector<vector<float>> x){
	x = transpose(x);
	x = relu( linear(x, OM_WEIGHT_0, OM_BIAS_0) );
	x = relu( linear(x, OM_WEIGHT_2, OM_BIAS_2) );
	x = linear(x, OM_WEIGHT_4, OM_BIAS_4);

	return x;
}

vector<vector<float>> forward(vector<vector<vector<float>>> obj,
															vector<vector<vector<float>>> sr,
															vector<vector<vector<float>>> rr,
															vector<vector<vector<float>>> ri){
	vector<vector<float>> predict;
	int N = data_len;
	int i = 0;
	//for(int i = 0; i < N; i++){
	int obj_h = obj[i][0].size();
	int sr_h = sr[i][0].size();
	int ri_w = ri[i].size();
	vector<vector<float>> obj_t = transpose(obj[i]);
	vector<vector<float>> sender = matmul(obj_t, sr[i]);
	vector<vector<float>> receiver = matmul(obj_t, rr[i]);
	int term_w = obj_h + obj_h + ri_w;
	int term_h = sr_h;
	vector<vector<float>> interaction_term = interaction_cat(term_w, term_h, sender, receiver, ri[i]);
	vector<vector<float>> effect = relational_model(interaction_term);
	vector<vector<float>> effect_receiver = matmul(rr[i], effect);
	effect_receiver = transpose( effect_receiver );
	vector<vector<float>> aggregate = aggregate_cat(obj_t, effect_receiver);
	vector<vector<float>> inf = object_model(aggregate);
	predict = transpose(inf);
	sender = matmul(predict, sr[i]);
	receiver = matmul(predict, rr[i]);
	int predict_shape_0 = 3;
	term_w = predict_shape_0 + predict_shape_0 + ri_w;
	term_h = sr_h;
	interaction_term = interaction_cat(term_w, term_h, sender, receiver, ri[i]);
	predict = relational_model(interaction_term);
	int m = predict.size();
	//cout << predict.size() << " " << predict[0].size();

	for (int j = 0; j < m; j++) {
		cout << predict[j][0] << " \n";
	}

	//}
	return predict;
}
