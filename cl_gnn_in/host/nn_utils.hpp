#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>
#include <iterator>
#include <math.h>
#include <vector>
#include <chrono>
#include "hdf5.h"

using namespace std;
using namespace std::chrono;


vector<vector<float>> matmul(const vector<vector<float>> a, const vector<vector<float>> b) {
	int m = a.size();
	int n = a[0].size();
	int p = b[0].size();
	vector<vector<float>> c(n, vector<float>(p));
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < p; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < n; ++k) {
				sum += a[i][k]*b[k][j];
			}
			c[i][j] = sum;
		}
	}
	return c;
}

vector<vector<float>> aggregate_cat(vector<vector<float>> obj_t, vector<vector<float>> effect_receiver){
	int term_w = obj_t.size() + effect_receiver.size();
	int term_h = obj_t[0].size();
	vector<float> obj_t_ = flatten(obj_t);
	vector<float> effect_receiver_ = flatten(effect_receiver);
	int m = term_h;
	vector<vector<float>> out(term_w, vector<float>(term_h));

	for(int x_idx = 0; x_idx < term_w; x_idx++){
		for(int y_idx = 0; y_idx < term_h; y_idx++){
			int x = x_idx * m + y_idx;
			if(x_idx < 3){
	        out[x_idx][y_idx] = obj_t_[x];
	    } else {
	        out[x_idx][y_idx] = effect_receiver_[x - (3 * m)];
	    }
		}
	}
	return out;
}

vector<vector<float>> interaction_cat(int term_w, int term_h, vector<vector<float>> sender, vector<vector<float>> receiver, vector<vector<float>> ri){
	vector<float> sender_ = flatten(sender);
	vector<float> receiver_ = flatten(receiver);
	vector<float> ri_ = flatten(ri);
	int m = term_h;
	vector<vector<float>> out(term_w, vector<float>(term_h));

	for(int x_idx = 0; x_idx < term_w; x_idx++){
		for(int y_idx = 0; y_idx < term_h; y_idx++){
			int x = x_idx * m + y_idx;
			if(x_idx < 3){
	        out[x_idx][y_idx] = sender_[x];
	    } else if(x_idx < 6){
	        out[x_idx][y_idx] = receiver_[x - (3 * m)];
	    } else {
	        out[x_idx][y_idx] = ri_[x - (6 * m)];
	    }
		}
	}
	return out;
}

vector<vector<float>> transpose(vector<vector<float> > &b)
{
    if (b.size() == 0)
        return b;
    vector<vector<float>> trans_vec(b[0].size(), vector<float>());
    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < b[i].size(); j++)
        {
            trans_vec[j].push_back(b[i][j]);
        }
    }
    return trans_vec;
}

vector<vector<float>> relu(vector<vector<float>> inp){
	int m = inp.size();
	int n = inp[0].size();
	vector<vector<float>> x(m, vector<float>(n));
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			float a = inp[i][j];
			x[i][j] = a > 0 ? a : 0;
		}
	}
	return x;
}

vector<vector<float>> sigmoid(vector<vector<float>> inp){
	int m = inp.size();
	int n = inp[0].size();
	vector<vector<float>> x(m, vector<float>(n));
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			x[i][j] = 1.0f / (1 + exp(-inp[i][j]));
		}
	}
	return x;
}

vector<vector<float>> add_bias(vector<vector<float>> inp, vector<float> bias){
	int m = inp.size();
	int n = inp[0].size();
	vector<vector<float>> x(m, vector<float>(n));
	for(int x_idx = 0; x_idx < m; x_idx++){
		for(int y_idx = 0; y_idx < n; y_idx++){
			x[x_idx][y_idx] = inp[x_idx][y_idx] + bias[y_idx];
		}
	}
	return x;
}
