// Implementation of the  Recursive Autoencoder Model by socher et al. EMNLP 2011
// Copyright (C) 2015 Marc Vincent
// 
// RAEcpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __COMMON_H_INCLUDED__
#define __COMMON_H_INCLUDED__

# define ARMA_32BIT_WORD
#ifdef USE_LIBLBFGS
#include <lbfgs.h>
#endif

#include <armadillo>
#include <Eigen/Dense>
#include <map>
#include <cstdlib>
#include <array>
#include <iostream>
#include <assert.h>
#include <utility>

// get rid of macro conflict
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include "dlib/optimization.h"

typedef dlib::matrix<double,0,1> column_vector; // from and for dlib
typedef std::function< double( const column_vector & ) > dlibdfunc; // from and for dlib
typedef std::function< const column_vector( const column_vector & ) > dlibfunc; // from and for dlib

const int thread_phrase = 0;
const int thread_bigram = 1;

const int z_is_tanh = 0;
const int z_is_tanh_norm = 1;
const int z_is_relu = 2;
const int z_is_relu_norm = 3;

const uint predict_term = 0;
const uint predict_max = 1;

// TODO erase \w references
const double keep_direct = 1; const double keep_indirect = 1;

enum run_types {
    sgd_phrase,
    sgd_minibatch,
    sgdmom_minibatch,
    adagrad_minibatch,
    adadelta_minibatch,
    gradient_batch,
    lbfgs_batch,
    dliblbfgs_batch
};

const uint dE_dL_i = 0;
const uint dE_deWh_i = 1;
const uint dE_deWr_i = 2;
const uint dE_deWc_i = 3;

const uint nan_int_value = std::numeric_limits<uint>::max();

using namespace std;
using namespace arma;
using namespace Eigen;

const std::map<std::string, uint> run_types_str = {
    { "sgd", sgd_phrase },
    { "sgd_minibatch", sgd_minibatch },
    { "sgdmom_minibatch", sgdmom_minibatch },
    { "adagrad_minibatch", adagrad_minibatch },
    { "adadelta_minibatch", adadelta_minibatch },
    { "gradient_batch", gradient_batch },
    { "lbfgs_batch", lbfgs_batch },
    { "dliblbfgs_batch", dliblbfgs_batch }
};

#ifdef USE_LIBLBFGS
const std::map<std::string, uint> lbfgs_ls_types = {
    { "armijo", LBFGS_LINESEARCH_BACKTRACKING_ARMIJO },
    { "morethuente", LBFGS_LINESEARCH_MORETHUENTE },
    { "wolfe", LBFGS_LINESEARCH_BACKTRACKING_WOLFE },
    { "strongwolfe", LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE }
};
#endif

const std::map<std::string, uint> bigger_perf_is_better = {
    { "auc", 1 },
    { "accuracy", 1 }
};

// some global variables
// TODO clean, encapsulate
extern uint basic_tree; // basic_tree components are pairs of words in order
extern uint clip_gradients; // 1 gives good result
extern double clip_constant; // per element when to
extern double transmission; // 1 gives good result, 0 with --to
extern uint clip_deltas; // 1 gives good result
extern double clip_deltas_factor; // 1 gives good result, 8 with no transmission
extern double const_param_norm; // if 0, no rescaling, 4.5 was good (kinda)
extern uint do_gradient_checks; //!< check if gradient is correct, disabled for speed
    // used in PARSE_TREE, main
extern uint replace_grad; // used in RAE, main
extern uint verbose;

// https://solarianprogrammer.com/2011/12/16/cpp-11-thread-tutorial/
struct dataset
{
    uvec labels;
    int n_phrases;
    uint v_size;
    vector <uvec> phrases;
    uvec train_idc;
    uvec val_idc;
    uvec test_idc;
};

struct Idx_Res
{
    int idx;
    double res;
};

struct perf_probs_labels_rep
{
    
    double perf = -1;
    string perf_type = "auc";
    uint sup_is_better = 1;
    mat probs;
    colvec labels;
    mat representations;

};

struct contrib
{
    
    rowvec deltas;
    rowvec deltas_hi;
    mat dE_dX;

};

struct phrases_labels {

    uvec* phrases;
    uvec labels;

};

/** \brief Structure containing informations to create a dataset from files

*/
struct project_path
{
    string labels_p = "";
    string phrases_p = "";
    string train_p = "";
    string val_p = "";
    string test_p = "";
    string save_models_dir = "";
    string save_predictions_dir = "";
    string save_representations_dir = "";
    int equalize = 0;
    int n_train = 0;
    int n_val = 0;
    int n_test = 0;
};

// optimization parameter

/** \brief Structure containing all the optimization parameters

*/
struct opt_params
{
    uint run_type = sgd_minibatch;
    double learning_rate = 0.005;
    double rho = 1; // for adadelta
    double epsilon = 0.00001; // for adadelta
    int thread_num = 8;
    int gradeval_batch_size = 64;    // target size of each batch
                                    // to be divided kby threads
    int predict_batch_size = 256;  // target size of each batch
                                    // to be divided kby threads
    int opt_batch_num = 40;    // in terms of minibatch
    int epoch_num = 100;    // in terms of minibatch
    int eval_after_epoch = 1;       // one epoch per training set
    uint equal_train_rt = 1;
#ifdef USE_LIBLBFGS
    lbfgs_parameter_t lbfgs_p;
#endif
    uint divide = 1;
    uint divide_L_type = 0; // do we divide dE_dL by tot n_phrases ( 0 )
                    // or n_phrases containing word ( 1 )
                    // or both ( 2 )
    double lr_decrease = 0;
    string perf_type = "auc";
    // everything for lbfgs
    uint wait_min = 1024;
    double wait_increase = 1.5;
    double improvement_threshold = 0.01;
    bool unsup_sup = false;
        // if true divide each optimization round in two parts:
        // unsupervised gradient/cost comp & supervised gradient/cost comp
    bool save_last_lbfgs = false;

};

// model parameters
struct model_params
{
    double alpha = 0.02;
    double lambda_reg_c = 0.;
    double lambda_reg_h = 0.;
    double lambda_reg_r = 0.;
    double lambda_reg_L = 0.;
    double lambda_reg_L_sup = 0.;
    int w_length = 50;
    uint z_type = z_is_tanh_norm;
    double nterm_w = 1; 

    uint predict_max = 0;
    uint fixed_tree = 0;
    bool single_words = false;
    bool norm_rec = false;
    bool weight_rec = true;
    bool divide_rec_contrib = false;
    bool original_L_scale = false; // random L init is multiplied bu 1e-3
    bool standard_L_scale = false; // random L init 
    bool unsup = true; // use unsupervised loss
    bool sup = true; // use supervised loss
    bool log_square_loss = false;
    bool keep_direct = true;
    bool keep_indirect = true;

};

#endif
