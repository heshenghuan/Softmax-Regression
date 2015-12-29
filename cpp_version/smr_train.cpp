#include <cstdlib>
#include <iostream>
#include <cstring>
#include "softmaxreg.h"

using namespace std;


void print_help() {
	cout << "\nOpenPR-smr training module, " <<"\n\n"
		<< "usage: smr_train [options] training_file model_file [pre_model_file]\n\n"
		<< "options: -h        -> help\n"
		<< "         -n int    -> maximal iteration loops (default 200)\n"
		<< "         -m double -> minimal loss value decrease (default 1e-03)\n"
		<< "         -t [0,1]  -> 0: training model by SGD optimization (default)\n"
		<< "                   -> 1: Gradient Descent optimization\n"
		<< "         -r double -> regularization parameter lambda of gaussian prior (default 0)\n"		
		<< "         -l float  -> learning rate (default 1.0)\n"
		<< "         -a        -> 0: final weight (default)\n"
		<< "                   -> 1: average weights of all iteration loops\n"
		<< "         -u [0,1]  -> 0: initial training model (default)\n"
		<< "                   -> 1: updating model (pre_model_file is needed)\n" 
		<< endl;
}

void read_parameters(int argc, char *argv[], char *training_file, char *model_file, 
						int *criter, int *max_loop, double *loss_thrd, float *learn_rate, float *lambda,
						int *avg, int *update, char *pre_model_file) {
	// set default options
	*criter = 0;
	*max_loop = 200;
	*loss_thrd = 1e-3;
	*learn_rate = 1.0;
	*lambda = 0.0;
	*avg = 0;
	*update = 0;
	int i;
	for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++) {
		switch ((argv[i])[1]) {
			case 'h':
				print_help();
				exit(0);
			case 't':
				*criter = atoi(argv[++i]);
				break;
			case 'n':
				*max_loop = atoi(argv[++i]);
				break;
			case 'm':
				*loss_thrd = atof(argv[++i]);
				break;
			case 'l':
				*learn_rate = (float)atof(argv[++i]);
				break;
			case 'r':
				*lambda = (float)atof(argv[++i]);
				break;
			case 'a':
				*avg = atoi(argv[++i]);
				break;
			case 'u':
				*update = atoi(argv[++i]);
				break;
			default:
				cout << "Unrecognized option: " << argv[i] << "!" << endl;
				print_help();
				exit(0);
		}
	}
	
	if ((i+1)>=argc) {
		cout << "Not enough parameters!" << endl;
		print_help();
		exit(0);
	}
	strcpy (training_file, argv[i]);
	strcpy (model_file, argv[i+1]);
	if (*update) {
		if ((i+2)>=argc) {
			cout << "Previous model file is needed in update mode!" << endl;
			print_help();
			exit(0);
		}
		strcpy (pre_model_file, argv[i+2]);
	}
}

int smr_train(int argc, char *argv[])
{
	char training_file[200];
	char model_file[200];
	int criter;
	int max_loop;
	double loss_thrd;
	float learn_rate;
	float lambda;
	int avg;
	int update;
	char pre_model_file[200];
	read_parameters(argc, argv, training_file, model_file, &criter, &max_loop, &loss_thrd, &learn_rate, &lambda, &avg, &update, pre_model_file);
    
    SoftmaxReg smr;
    smr.load_training_file(training_file);
    if (update) {
		smr.load_model(pre_model_file);
	}
	else {
		smr.init_omega();	
	}
	if (criter) {
		smr.train_batch(max_loop, loss_thrd, learn_rate, lambda, avg);		
	}
	else {
		smr.train_SGD(max_loop, loss_thrd, learn_rate, lambda, avg);
	}
	smr.save_model(model_file);
	return 1;
}

int main(int argc, char *argv[])
{
    return smr_train(argc, argv);
}
