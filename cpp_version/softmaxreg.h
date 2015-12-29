#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <math.h>
#include <time.h>
using namespace std;

struct feature
{
    vector<int> id_vec;                 //index
    vector<float> value_vec;            //value
};

class SoftmaxReg
{
private:
    vector<feature> samp_feat_vec;      //sample list
    vector<int> samp_class_vec;         //vec of samples' class
    vector< vector<float> > omega;      //weights matrix
    //vector<string> label_set;         //label set
    int feat_set_size;                  //dimension of feature
    int class_set_size;                 //the num of class

public:
    SoftmaxReg();
    ~SoftmaxReg();
    void save_model(string model_file);
    void load_model(string model_file);
    void load_training_file(string training_file);
    void init_omega();
    
    int train_SGD(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg);
    int train_batch(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg);
    vector<float> calc_score(feature &samp_feat);
    vector<float> score_to_prb(vector<float> &score);
    int score_to_class(vector<float> &score);
    
    float classify_testing_file(string testing_file, string output_file, int output_format);

private:
    void read_samp_file(string samp_file, vector<feature> &samp_feat_vec, vector<int> &samp_class_vec);

    void update_omega(int samp_class, feature &samp_feat, float learn_rate, float lambda);
    void calc_loss(double *loss, float *acc, float lambda);
    
    float calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec);   
    
    float sigmoid(float x);
    vector<string> string_split(string terms_str, string spliting_tag);
};