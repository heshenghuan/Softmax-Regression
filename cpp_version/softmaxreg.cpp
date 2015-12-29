/*
*A c++ Softmax Regression algorithm implementation
*author Heshenghuan (heshenghuan999@163.com)
*/
#include "SoftmaxReg.h"

SoftmaxReg::SoftmaxReg(){}
SoftmaxReg::~SoftmaxReg(){}

void SoftmaxReg::save_model(string model_file)
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
    // Write class_set_szie and feat_set_size
    // Also is the row and cloumn num of the weigths matrix
    fout << class_set_size << " " << feat_set_size << endl;

    // Write weights matrix.
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            fout << omega[k][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

void SoftmaxReg::load_model(string model_file)
{
    cout << "Loading model..." << endl;
    omega.clear();
    ifstream fin(model_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << model_file << endl;
        exit(0);
    }
    string line_str;
    // Read feat_set_size
    // This line is useless for c/c++ program, but useful for python program
    // So just read and ignore it. 
    getline(fin, line_str);
    
    // Read weights matrix
    while (getline(fin, line_str)) {
        vector<string> line_vec = string_split(line_str, " ");
        vector<float>  line_omega;
        for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
            float weight = (float)atof(it->c_str());
            line_omega.push_back(weight);
        }
        omega.push_back(line_omega);
    }
    fin.close();
    // the size of omega is feature dimension.
    feat_set_size = (int)omega.size();
    // the size of a element of omega is the num of label.
    class_set_size = (int)omega[0].size();
}

void SoftmaxReg::read_samp_file(string samp_file, vector<feature> &samp_feat_vec, vector<int> &samp_class_vec) {
    ifstream fin(samp_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
    string line_str;
    while (getline(fin, line_str)) {
        size_t class_pos = line_str.find_first_of("\t");
        int class_id = atoi(line_str.substr(0, class_pos).c_str());
        samp_class_vec.push_back(class_id);
        string terms_str = line_str.substr(class_pos+1);
        feature samp_feat;
        samp_feat.id_vec.push_back(0); // bias
        samp_feat.value_vec.push_back(1); // bias
        if (terms_str != "") {
            vector<string> fv_vec = string_split(terms_str, " ");
            for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) {
                size_t feat_pos = it->find_first_of(":");
                int feat_id = atoi(it->substr(0, feat_pos).c_str());
                float feat_value = (float)atof(it->substr(feat_pos+1).c_str());
                if (feat_value != 0) {
                    samp_feat.id_vec.push_back(feat_id);
                    samp_feat.value_vec.push_back(feat_value);              
                }
            }
        }
        samp_feat_vec.push_back(samp_feat);
    }
    fin.close();
}


void SoftmaxReg::load_training_file(string training_file)
{
    cout << "Loading training data..." << endl;
    read_samp_file(training_file, samp_feat_vec, samp_class_vec);
    feat_set_size = 0;
    class_set_size = 0;
    for (size_t i = 0; i < samp_class_vec.size(); i++) {
        if (samp_class_vec[i] > class_set_size) {
            class_set_size = samp_class_vec[i];
        }
        if (samp_feat_vec[i].id_vec.back() > feat_set_size) {
            feat_set_size = samp_feat_vec[i].id_vec.back();
        }   
    }
    class_set_size += 1;
    feat_set_size += 1;
}

void SoftmaxReg::init_omega()
{
    float init_value = 0.0;
    //float init_value = (float)1/class_set_size;
    for (int i = 0; i < feat_set_size; i++) {
        vector<float> temp_vec(class_set_size, init_value);
        omega.push_back(temp_vec);
    }
}

// Stochastic Gradient Descent (SGD) optimization
int SoftmaxReg::train_SGD(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg)
{
    int id = 0;
    double loss = 0.0;
    double loss_pre = 0.0;
    vector< vector<float> > omega_sum(omega);
    while (id <= max_loop*(int)samp_class_vec.size()) {
        // check loss value
        if (id%samp_class_vec.size() == 0) {
            int loop = id/(int)samp_class_vec.size();
            double loss = 0.0;
            float acc = 0.0;
            calc_loss(&loss, &acc, lambda);
            cout.setf(ios::left);
            cout << "Iter: " << setw(8) << loop << "Loss: " << setw(18) << loss << "Acc: " << setw(8) << acc << endl;
            if ((loss_pre - loss) < loss_thrd && loss_pre >= loss && id != 0) {
                cout << "Reaching the minimal loss value decrease!" << endl;
                break;
            }
            loss_pre = loss;
        }
        // update omega
        int r = (int)(rand()%samp_class_vec.size());   //randomly choose a sample
        //int r = (int)i%samp_class_vec.size();
        feature samp_feat = samp_feat_vec[r];
        int samp_class = samp_class_vec[r];
        update_omega(samp_class, samp_feat, learn_rate, lambda);
        if (avg == 1) {
            for (int i = 0; i < feat_set_size; i++) {
                for (int j = 0; j < class_set_size; j++) {
                    omega_sum[i][j] += omega[i][j];
                }
            }           
        }
        id++;
    }
    if (avg == 1) {
        for (int i = 0; i < feat_set_size; i++) {
            for (int j = 0; j < class_set_size; j++) {
                omega[i][j] = (float)omega_sum[i][j] / id;
            }
        }       
    }
    return 1;
}

int SoftmaxReg::train_batch(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg)
{
    int id = 0;
    int err_num = 0;
    size_t m = samp_class_vec.size();
    double regular = 0.0;
    double loss = 0.0;
    double loss_pre = 0.0;
    vector< vector<float> > omega_sum(omega);
    vector< vector<float> > delta;
    while (id <= max_loop) {
        loss = 0.0;
        err_num = 0;
        // delta is the accumulator for the gradient of all the sample
        delta.clear();
        for (int k = 0; k < feat_set_size; k++) {
            vector<float> vec;
            for (int j = 0; j < class_set_size; j++) {
                vec.push_back(-lambda * omega[k][j]);
                regular += omega[k][j] * omega[k][j];
            }
            delta.push_back(vec);
        }
        for (size_t i = 0; i < samp_class_vec.size(); i++) {
            int samp_class = samp_class_vec[i];
            feature samp_feat = samp_feat_vec[i];
            vector<float> score = calc_score(samp_feat);
            vector<float> prb = score_to_prb(score);
            int pred_class = score_to_class(score);
            if (pred_class != samp_class) err_num++;
            loss += log(prb[samp_class]);
            for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
                int feat_id = samp_feat.id_vec[k];
                float feat_value = samp_feat.value_vec[k];
                for(size_t j = 0; j < class_set_size; j++) {
                    // delta[feat_id][j] += learn_rate * ((samp_class == j?1:0 - prb[j]) * feat_value - lambda * omega[feat_id][j]);
                    delta[feat_id][j] += (samp_class == j?1:0 - prb[j]) * feat_value / m;
                }
            }

        }
        loss = -loss / m + lambda * regular / 2;
        float acc = 1 - (float)err_num / samp_class_vec.size();
        cout.setf(ios::left);
        cout << "Iter: " << setw(8) << id << "Loss: " << setw(18) << loss << "Acc: " << setw(8) << acc << endl;
        if ((loss_pre - loss) < loss_thrd && loss_pre >= loss && id != 0) {
            cout << "Reaching the minimal loss value decrease!" << endl;
            break;
        }
        loss_pre = loss;
        // update omega
        for (int i = 0; i < feat_set_size; i++) {
            for (int j = 0; j < class_set_size; j++) {
                omega[i][j] += learn_rate * delta[i][j];
            }
        }
        if (avg == 1) {
            for (int i = 0; i < feat_set_size; i++) {
                for (int j = 0; j < class_set_size; j++) {
                    omega_sum[i][j] += omega[i][j];
                }
            }
        }
        id++;
    }
    if (avg == 1) {
        for (int i = 0; i < feat_set_size; i++) {
            for (int j = 0; j < class_set_size; j++) {
                omega[i][j] = (float)omega_sum[i][j] / (id * samp_class_vec.size());
            }
        }       
    }
    return 1;
}

void SoftmaxReg::update_omega(int samp_class, feature &samp_feat, float learn_rate, float lambda)
{
    vector<float> score = calc_score(samp_feat);
    vector<float> prb = score_to_prb(score);
    for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
        int feat_id = samp_feat.id_vec[k];
        float feat_value = samp_feat.value_vec[k];
        for(size_t j = 0; j < class_set_size; j++) {
            omega[feat_id][j] += learn_rate * ((samp_class == j?1:0 - prb[j]) * feat_value);
        }
    }
    for (int k = 0; k < feat_set_size; k++)
        for (int j = 0; j < class_set_size; j++)
            omega[k][j] -= learn_rate * lambda * omega[k][j];
}

void SoftmaxReg::calc_loss(double *loss, float *acc, float lambda)
{
    double loss_value = 0.0;
    double regular = 0.0;
    int err_num = 0;
    for (size_t i = 0; i < samp_class_vec.size(); i++) {
        int samp_class = samp_class_vec[i];
        feature samp_feat = samp_feat_vec[i];
        vector<float> score = calc_score(samp_feat);
        vector<float> prb = score_to_prb(score);
        int pred_class = score_to_class(score);
        if (pred_class != samp_class) {
            err_num++;
        }
        loss_value += log(prb[samp_class]);
    }

    // calculate regularization
    for (int k = 0; k < feat_set_size; k++)
        for (int j = 0; j < class_set_size; j++)
            regular += omega[k][j] * omega[k][j];

    cout<< "regular:    "<<regular<<endl;
    cout<< "loss value: "<<loss_value<<endl;

    *acc = 1 - (float)err_num / samp_class_vec.size();
    *loss = -loss_value / samp_class_vec.size() + lambda * regular / 2;
}

vector<float> SoftmaxReg::calc_score(feature &samp_feat)
{
    vector<float> score(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++) {
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) {
            int feat_id = samp_feat.id_vec[k];
            float feat_value = samp_feat.value_vec[k];
            score[j] += omega[feat_id][j] * feat_value;
        }
    }
    return score;
}

vector<float> SoftmaxReg::score_to_prb(vector<float> &score)
{   
    vector<float> prb(class_set_size, 0);
    for (int i = 0; i < class_set_size; i++) {
        float delta_prb_sum = 0.0;
        for (int j = 0; j < class_set_size; j++) {
            delta_prb_sum += exp(score[j] - score[i]);
        }
        prb[i] = 1 / delta_prb_sum;
    }
    return prb;
}

int SoftmaxReg::score_to_class(vector<float> &score)
{
    int pred_class = 0; 
    float max_score = score[0];
    for (int j = 1; j < class_set_size; j++) {
        if (score[j] > max_score) {
            max_score = score[j];
            pred_class = j;
        }
    }
    return pred_class;
}

float SoftmaxReg::classify_testing_file(string testing_file, string output_file, int output_format)
{
    cout << "Classifying testing file..." << endl;
    vector<feature> test_feat_vec;
    vector<int> test_class_vec;
    vector<int> pred_class_vec;
    read_samp_file(testing_file, test_feat_vec, test_class_vec);
    ofstream fout(output_file.c_str());
    for (size_t i = 0; i < test_class_vec.size(); i++) {
        int samp_class = test_class_vec[i];
        feature samp_feat = test_feat_vec[i];
        vector<float> pred_score = calc_score(samp_feat);           
        int pred_class = score_to_class(pred_score);
        pred_class_vec.push_back(pred_class);
        fout << pred_class << "\t";
        if (output_format == 1) {
            for (int j = 0; j < class_set_size; j++) {
                fout << j << ":" << pred_score[j] << ' '; 
            }       
        }
        else if (output_format == 2) {
            vector<float> pred_prb = score_to_prb(pred_score);
            for (int j = 0; j < class_set_size; j++) {
                fout << j << ":" << pred_prb[j] << ' '; 
            }
        }
        fout << endl;       
    }
    fout.close();
    float acc = calc_acc(test_class_vec, pred_class_vec);
    return acc;
}

float SoftmaxReg::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
    size_t len = test_class_vec.size();
    if (len != pred_class_vec.size()) {
        cerr << "Error: two vectors should have the same lenght." << endl;
        exit(0);
    }
    int err_num = 0;
    for (size_t id = 0; id != len; id++) {
        if (test_class_vec[id] != pred_class_vec[id]) {
            err_num++;
        }
    }
    return 1 - ((float)err_num) / len;
}

float SoftmaxReg::sigmoid(float x) 
{
    double sgm = 1 / (1+exp(-(double)x));
    return (float)sgm;
}

vector<string> SoftmaxReg::string_split(string terms_str, string spliting_tag)
{
    vector<string> feat_vec;
    size_t term_beg_pos = 0;
    size_t term_end_pos = 0;
    while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) {
        if (term_end_pos > term_beg_pos) {
            string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
            feat_vec.push_back(term_str);
        }
        term_beg_pos = term_end_pos + 1;
    }
    if (term_beg_pos < terms_str.size()) {
        string end_str = terms_str.substr(term_beg_pos);
        feat_vec.push_back(end_str);
    }
    return feat_vec;
}