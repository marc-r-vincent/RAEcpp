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

#ifndef __PT_GROUP_H_INCLUDED__
#define __PT_GROUP_H_INCLUDED__

/** \brief a group of parse trees on which learning/testing is done

*/

struct grad_struct
{

    mat dE_dL;
    mat dE_deWr;
    mat dE_deWc;
    mat dE_deWh;

};

class PT_GROUP
{

public:

    PT_GROUP(){};

    PT_GROUP(
        vector <uvec>  phrases,
        uvec labels,
        AE * rae,
        uint fixed_tree,
        uint n_subgroups,
        opt_params op,
        std::thread * t_pool,
        PROGRESS_MGR * prog_mgr,
        std::string save_models
    );
    
    PT_GROUP(
        vector <uvec> phrases,
        uvec labels,
        uvec selection,
        AE * rae,
        uint fixed_tree,
        uint n_subgroups,
        opt_params op,
        std::thread * t_pool,
        PROGRESS_MGR * prog_mgr,
        std::string save_models_dir
    );

    ~PT_GROUP();

    int print_update_norms( int nrm );
    int learn();
    int predict();

    int put_back_in_model();

    int init_derivatives();
    int fill_derivatives();

    int init_measures();
    int init_pt_buffers();

    int copy_grad_to_rowvec( rowvec * theta_grad );
    int copy_rowvec_to_grad( rowvec theta_grad );
    int copy_grad_to_dlibvec( column_vector * theta_grad );
    int copy_dlibvec_to_grad( column_vector theta_grad );
    int populate(
        vector <uvec> phrases, uvec labels, uvec selection, AE * rae,
        uint fixed_tree, uint n_subgroups, opt_params op,
        std::thread * t_pool, PROGRESS_MGR *  prog_mgr,
        std::string save_models_dir
    );
    int accumulate_grad( PT_GROUP * ptg );
    int accumulate_phrase_grad( PARSE_TREE * pt );
    int learn_lbfgs();
    int learn_dliblbfgs();
    int learn_not_lbfgs();
    int divide_cost();
    int divide_grad();
    int divide_grad( uvec wordcounts, double n_phrases );
    int grad_multiply( double n );
    int addto_grad( double n );
    int grad_neg();
    int clip_grad();
    int update_adagrad();
    int update_adadelta();
    int update_momentum();
    int update_vanilla();
    int subdivide();
    int collect_grads( uint from, uint to );
    int collect_preds();
    int collect_costs( uint from, uint to );
    int collect_costs();
    int update_lr();

    double dlib_batch_cost( const column_vector& m );
    const column_vector dlib_gradeval( const column_vector& m );

    int cache_pars( uint n_dups );
    int get_cached_pars(  uint cache_slot );

    int n_phrases = -1;
    uint n_words = 0;
    int has_labels = 1;
    uint is_subdivided = 0;
    uint is_cached = 0;
    uint prediction_type = predict_term;
    uint fixed_tree;
    uint epoch_index = 0;
    int curr_lbfgs = 0;
    int last_lbfgs = -1;
    uint lbfgs_retval = 0;
    uint level = 0;

    std::string save_models_dir;
    vector <uvec> phrases;
    uvec labels;
    uvec indices;
    vector <PARSE_TREE> pts;
    AE * rae;
    PT_GROUP * ptg_predict;
    PROGRESS_MGR * prog_mgr;

    // arrays of permanent parameter matrices
    // vector < vector< mat > > *  params_buff;
    vector < vector< mat > > *  params_buff;
    // current group contribution to model
    mat *dE_deWc;
    mat *dE_deWh;
    mat *dE_deWr;
    mat *dE_dL;
    vector <mat *> params;

    mat probs;
    mat representations;
    uvec wordcounts;
    rowvec rec_errs;
    rowvec wnlls;
    double cost_unsup; // unsup cost, updated after a call to predict
    double cost_sup; // sup cost, updated after a call to predict
    double cost; // total cost, updated after a call to predict
    double prev_best_perf;
    rowvec costs; // subgroups costs
    rowvec * grad_accu; // gradient accumulator variable for adagrad and adadelta
    rowvec * param_accu; // gradient accumulator variable for adadelta
    
    opt_params op;

    vector <PT_GROUP>  subgroups;
    uint n_subgroups = 0 ;
    uint n_subgroups_at_sd = 0;

    struct perf_probs_labels_rep ppl;

    std::thread * t_pool;

#ifdef USE_LIBLBFGS
    lbfgsfloatval_t * theta;
    int copy_grad_to_1Darray( lbfgsfloatval_t * theta_grad );
    int copy_1Darray_to_grad( lbfgsfloatval_t * theta_grad );
    static lbfgsfloatval_t lbfgs_evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        );
    static int lbfgs_progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        );
#endif

    int gradeval();
    int gradeval_batch();
    int gradient_check();
    static void gradeval_batch_parts( PT_GROUP * );
    int predict_batch();
    static void predict_batch_parts( PT_GROUP * );
    struct perf_probs_labels_rep get_perf_p_l();
    int grad_load( grad_struct gs );
    int grad_add( grad_struct gs );
    int grad_save( grad_struct &gs );
    int grad_struct_fill( grad_struct &gs );

};

# endif
