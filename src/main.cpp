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

// #define ARMA_NO_DEBUG 
// no debug and potentially speedup...
# define ARMA_32BIT_WORD

#include <thread>
#include <sys/stat.h>
#include <common.h>
#include <defs.h>
#include <util.h>
#include <AE.h>
#include <PARSE_TREE.h>
#include <PROGRESS_MGR.h>
#include <PT_GROUP.h>
#include <main.h>

// glob is POSIX

/**
\mainpage

A C++ 11 implementation of the Recursive Auto Encoder model described in Socher et al. "Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions, Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning." EMNLP 2011.  

Minor modifications were added compared to the original publication such as the optionnal use of RELU activation function and use of stochastic gradient descent, adagrad or adadelta. Batch optimization using l-BFGS is also available using either libLBFGS, or dlib.

Comes with a CLI and options to run it on an arbitrary dataset.

The dataset is made of four files. One representing all the phrases of the corpus plus three sets files corresponding to: train, validation, test.

The corpus file has the following binary format:
< m : long int >{ 1 }< < n : long int >{ 1 }< w : long int >{ n }> >{ m }

Using as a notation convention:
* data name enclosed and data type are separated by a colon and within angle brackets
* quantifiers relating to data in the immediate left bracket are enclosed in curly brackets
* m is the number of phrases in the corpus n is the number of words in a given phrase w is the index of a word in a given phrase

Long int is int32 by default ( using a bigger storage would require that you change armadillo's compilation flags ).

Relies on Armadillo for linear algebra, and runs on CPU. Mulithreading is available but depends on the batch size used during optimzation. Each thread will deal with a set of phrases concurrently, therefore to occupy k threads you should have at least k phrases in your batch. To keep your CPU busy you will realistically need k\*l batches where k depends on the number of cores available and l depends on the size of the training set and whatever overall batch size is the best for the chosen optimzation procedure.

Automatically saves training progress, best models and learned representations/embeddings of words and phrases.

\par Dependencies 

- RAEcpp minimally requires *Armadillo*, *hdf5*, *dlib* and *eigen3* libraries. Dlib will be downloaded by cmake, if the other dependencies are present cmake should find them, if not you'll have to install them.
- For lBFGS type optimization either *dlib*, *libLBFGS* or both have to be present.
dlib relies on eigen3.
- To generate the complete documentation you need doxygen to be present

\par Installing

cd path-to-dir-containg-this-file
cmake CMakeLists.txt # check dependencies, making makefile
doxygen doxyset.dx # generate documentation using doxygen

\par Running

Scripts examples are provided to do training and testing on an example dataset ( Pang & lee's movie dataset ), for that you'll need wget :
chmod 755 ./utils/\* # so that our scripts become executable ( remove backslashes if you read the current file as plain text )
./utils/down_pang_lee.sh # download the dataset ( remove backslashes if you read the current file as plain text )
python ./Scripts/make_pg_ds.py # transform to run_rae format ( remove backslashes if you read the current file as plain text )
./example/scripts/run_adadelta.sh # train using adadelta and test ( remove backslashes if you read the current file as plain text )

Generally speaking:
- To learn a model, give path to data and training subset ( sufficient for l-BFGS methods )
and optionnaly a validation subset ( necessary for non l-BFGS methods )
- To evaluate a model, give path to data and validation subset with labels different from -1
- To do predictions on a corpus, give a path to data and test subset

\par License

Implementation of the  Recursive Autoencoder Model by socher et al. EMNLP 2011
Copyright (C) 2015 Marc Vincent

RAEcpp is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

void pt_group_learn( PT_GROUP* ptg ){ ptg->learn(); }

void pt_group_predict( PT_GROUP* ptg ){ ptg->predict(); }

struct dataset read_dataset(
    string labels_p,
    string phrases_p,
    string train_fold_p,
    string val_fold_p,
    string test_fold_p,
    int equalize,
    int n_train,
    int n_val,
    int n_test
){

    ivec labels_int;
    uvec phrases_ds;
    uvec train_idc;
    uvec val_idc;
    uvec test_idc;

    labels_int.load( labels_p, raw_binary );

    // all negs are converted to maximum uint value
    uvec labels = conv_to<uvec>::from( abs( labels_int ) );
    labels.elem( (uvec) find( labels_int < 0 ) ).fill( nan_int_value );

    phrases_ds.load( phrases_p, raw_binary );

    int n_phrases = phrases_ds[0];
    
    vector <uvec> phrases;

    cout << "# labels: " << labels.n_elem << endl;
    cout << "# phrases: " << n_phrases << endl;

    assert( labels.n_elem == n_phrases  );

    // read phrases rep
    int p_pos = 1;

    uint v_size = 0;

    for( int p_i = 0 ; p_i < n_phrases ; p_i++ ){

        int n_words = phrases_ds[ p_pos ];

        phrases.push_back( phrases_ds( span( p_pos + 1, p_pos + n_words ) ) );

        v_size = max( v_size, max( phrases[ p_i ] ) + 1 );

        p_pos = p_pos + n_words + 1;

    }

    int ssizes[3] = { n_train, n_val, n_test };
    std::string sets_p[3] = { train_fold_p, val_fold_p, test_fold_p };
    uvec * sets = new uvec [3];

    for( int set_i = 0; set_i < 3; set_i++ ){

        // the following are optional
        if( sets_p[ set_i ] != "" ){

            sets[ set_i ].load( sets_p[ set_i ], raw_binary );

            cout << "# set " << set_i << " length before equalize: " <<
                sets[ set_i ].n_elem << endl;

            // we do not cut the dataset nor change the order for the test set
            // but for the other two this is a valid option
            if( set_i != 2 ){
                uvec alt_order = order_labels_alternate( labels( sets[ set_i ] ) ,
                    equalize );
                sets[ set_i ] = sets[ set_i ]( alt_order );
            }

            cout << "# set " << set_i << " length after equalize: " <<
                sets[ set_i ].n_elem << endl; 

            if( ssizes[ set_i ] > 0 )
                sets[ set_i ] = sets[ set_i ]( span( 0, ssizes[ set_i ] - 1 ) );

        }

    }

    struct dataset ds;

    ds.labels = labels;
    ds.phrases = phrases;
    ds.n_phrases = n_phrases;
    ds.train_idc = sets[0];
    ds.val_idc = sets[1];
    ds.test_idc = sets[2];
    ds.v_size = v_size;

    delete[] sets;

    cout << "# Vocabulary size: " << ds.v_size << endl << endl;

    return ds;

}

struct dataset read_dataset(
    string labels_p,
    string phrases_p,
    string train_fold_p,
    string val_fold_p,
    string test_fold_p,
    int equalize
){

    return(
        read_dataset(
            labels_p,
            phrases_p,
            train_fold_p,
            val_fold_p,
            test_fold_p,
            equalize,
            0, 0, 0
        )
    );

}

void copy_dataset( struct dataset * ds, struct dataset * ds_cop ){

    /*if( ds_cop->labels.n_elem > 0 ){
        delete ds_cop->phrases;
    }*/

    ds_cop->labels = ds->labels;
    ds_cop->n_phrases = ds->n_phrases;
    ds_cop->v_size = ds->v_size;
    ds_cop->train_idc = ds->train_idc;
    ds_cop->val_idc = ds->val_idc;
    ds_cop->test_idc = ds->test_idc;

    for( uint i = 0; i < ds_cop->n_phrases; i++ )
        ds_cop->phrases.push_back( ds->phrases[ i ] );

}

int check_options_compat( model_params mp, opt_params op, uint n_out ){

    if( n_out > 2 and mp.log_square_loss  ){
        cerr << "you asked for a logistic loss but there are more than two " << 
            "labels in your dataset.\n" <<
            "Please remove the --log_square_loss option before re-launching.\n" <<
            "Exiting..." << endl; exit(0);
        
    }   // if we asked for a logistic loss check that there are indeed no more
        // than two labels

    if(
        op.unsup_sup and
        ( op.run_type != lbfgs_batch and op.run_type != dliblbfgs_batch )
    ){
        cerr << "you asked for a two pass gradient/cost evaluation which is not" <<
            "compatible with the optimization procedure you chose\n" <<
            "either remove the --unsup_sup option or choose an lbfgs optimization procedure\n" <<
            "Exiting..." << endl; exit(0);
    }

}

int run_dataset( dataset ds_in, model_params mp, opt_params op,
    project_path pp ){

    // check pathes are correct
    vector<string> to_test {
        pp.save_models_dir, pp.save_predictions_dir,
        pp.save_representations_dir
    };

    uint wrong_path = 0;
    for( string pth : to_test ){
        FILE * file;        
        if( FILE *file = fopen( pth.c_str(), "r" ) ) fclose( file );
        else{
            cerr << "error while opening: " << pth  << endl;
            wrong_path = 1;
        }
    }
    if( wrong_path ){ cout << "please check arguments." << endl; exit(0); }
    
    // no learning_rate if we use batch learning
    if(
        op.run_type == gradient_batch || op.run_type == lbfgs_batch ||
        op.run_type == dliblbfgs_batch 
    ){ op.learning_rate = 1; }

    uvec outcomes = unique( ds_in.labels );

    check_options_compat( mp, op, outcomes.n_elem );

    AE rae = AE(
            ds_in.v_size,
            mp,
            outcomes.n_elem
    );

    if( op.unsup_sup ){ rae.mp.unsup = true; rae.mp.sup = false; }

    uint prediction_type = mp.predict_max;

    // if save folder already contain models, ask user if the best
    // of them should be loaded
    uint keep_sel_size = 10;
    PROGRESS_MGR prog_mgr = PROGRESS_MGR(
        pp,
        keep_sel_size,
        &rae,
        op
    );

    if( verbose >= 2 )
        rae.regularizer_for_cost();

    prog_mgr.get_best_saved( true, false );

    // if we use lbfgs termination is managed by lbfgs library
    if( op.run_type == lbfgs_batch || op.run_type == dliblbfgs_batch)
        op.opt_batch_num = 1;

    cout << "# running with opts:" << endl;
    cout << "# mp.w_length: " << mp.w_length << endl;
    cout << "# ds_in.v_size: " << ds_in.v_size << endl;
    cout << "# op.learning_rate: " << op.learning_rate << endl;
    cout << "# op.rho: " << op.rho << endl;
    cout << "# op.unsup_sup: " << op.unsup_sup << endl;
    cout << "# mp.alpha: " << mp.alpha << endl;
    cout << "# mp.lambda_reg_c: " << mp.lambda_reg_c << endl;
    cout << "# mp.lambda_reg_h: " << mp.lambda_reg_h << endl;
    cout << "# mp.lambda_reg_r: " << mp.lambda_reg_r << endl;
    cout << "# mp.lambda_reg_L: " << mp.lambda_reg_L << endl;
    cout << "# mp.lambda_reg_L_sup: " << mp.lambda_reg_L_sup << endl;
    cout << "# mp.z_type: " << mp.z_type << endl;
    cout << "# mp.nterm_w: " << mp.nterm_w << endl;
    cout << "# mp.norm_rec: " << mp.norm_rec << endl;
    cout << "# mp.single_words: " << mp.single_words << endl;

    std::thread * t_pool = new std::thread[ op.thread_num ];

    // create train PT_GROUP if there is a train set and train on it
    // ( requires validation set for online methods )
    PT_GROUP ptg_train = PT_GROUP();
    // create predict PT_GROUP
    PT_GROUP ptg_predict = PT_GROUP();
    // create predict PT_GROUP
    PT_GROUP ptg_test = PT_GROUP();
 
    if( ds_in.train_idc.n_elem != 0 ){

        // for now none of the labels of training should be unknown ( < -1 )
        // TODO allow unknown ( ie. change labels from uvec to vec )
        assert( all( ds_in.labels( ds_in.train_idc ) != nan_int_value ) );

        ptg_train.populate(
            ds_in.phrases, ds_in.labels, ds_in.train_idc, &rae, mp.fixed_tree, 1,
            op, t_pool, &prog_mgr, pp.save_models_dir
        );

        ptg_train.cache_pars( 2 + op.thread_num );
        ptg_train.get_cached_pars( 0 );

        if( ds_in.val_idc.n_elem != 0 ){

            // for now none of the labels of training should be unknown
            // TODO allow unknown ( ie. change labels from uvec to vec )
            assert( all(
                ds_in.labels( ds_in.labels ) != nan_int_value
            ) );

            ptg_predict.populate(
                ds_in.phrases, ds_in.labels, ds_in.val_idc, &rae, mp.fixed_tree, 1, op,
                t_pool, nullptr, "" 
            );

            // share params buff with ptg_train
            ptg_predict.params_buff = ptg_train.params_buff;
            ptg_predict.get_cached_pars( 0 );

            // pass predict to learn optimizer
            ptg_train.ptg_predict = &ptg_predict;
        
        }

        if( op.run_type != lbfgs_batch && op.run_type != dliblbfgs_batch )
           assert( ds_in.val_idc.n_elem != 0 ); /* You NEED a validation set for the current optimization method*/ 
        ptg_train.learn();
        
    }

    // if there is a test set, apply the model
    if( ds_in.test_idc.n_elem != 0 ){

        cout << "# Predicting on test set." << endl;

        prog_mgr.get_best_saved( false, true );

        ptg_test.populate(
            ds_in.phrases, ds_in.labels, ds_in.test_idc, &rae, mp.fixed_tree, 1, op,
            t_pool, nullptr, "" 
        );

        // share params buff with ptg_train if it exists
        if( ds_in.train_idc.n_elem != 0 ){
            ptg_test.params_buff = ptg_train.params_buff;
            ptg_test.get_cached_pars( 0 );
        }else{
            ptg_test.cache_pars( 2 + op.thread_num );
            ptg_test.get_cached_pars( 0 );
        }

        // if there is only one value of labels,
        // TODO change that either to 
        // -1 labels are automatically considered as missing labels
        uvec uniqued_te_labels = unique( ptg_test.labels );
        if( uniqued_te_labels.n_elem < 1 && uniqued_te_labels[0] == nan_int_value )
            ptg_test.has_labels = 1;

        // do predictions
        ptg_test.predict();

        // save predictions, with labels if any, without otherwise
        cout << "# saving predictions." << endl;
        ofstream pfile (
            pp.save_predictions_dir + "/preds_test" + ".csv"
        );
        pfile << join_rows( ptg_test.ppl.probs, ptg_test.ppl.labels );
        pfile.close();

        if( ds_in.train_idc.n_elem == 0 ) delete ptg_test.params_buff;

    }

    if( ds_in.train_idc.n_elem != 0 ) delete ptg_train.params_buff;

    delete[] t_pool;

    return( 0 );

}


int test_triplet( model_params mp, opt_params op, int tt_length, uint n_out ){
    
    if( tt_length == 0 )
        cout << "no --tt_length option generating phrases of length 2" << endl;

    // TODO set options for the simulation
    double learning_rate = 0.01;
    uint v_size = 50000;
    // uint n_out = 2;
    uint p_num = 3;
    n_out = min( p_num, n_out );
    uint words_num = tt_length == 0 ?  2: tt_length; 
    v_size =  1000;
    uint min_plength = 3;
    uint max_p_length = 30;

    cout << "params:" << endl <<
        "mp.lambda_reg_L: " << mp.lambda_reg_L << endl <<
        "mp.lambda_reg_c: " << mp.lambda_reg_c << endl <<
        "mp.lambda_reg_h: " << mp.lambda_reg_h << endl <<
        "mp.lambda_reg_r: " << mp.lambda_reg_r << endl;

    AE rae = AE(
            v_size,
            mp,
            n_out
    );

    int thread_num = 1;
    std::thread * t_pool  = new std::thread[ thread_num ];

    // create a list of lists (3 entries, 2 to 5 words each)
    // pointer of pointers
    vector <uvec> phrases;

    uvec labels = zeros<uvec>( p_num );
    uint curr_lab = 0;

    for( uint i = 0 ; i < p_num ; i++ ){

        labels[ i ] = curr_lab;

        /* words_num = min_plength + ( uint ) 
                rand() % ( max_p_length - min_plength )
            ;*/

        phrases.push_back( randi<uvec>(
            words_num,
            distr_param(0, rae.L.n_rows - 1)
        ) );

        /*
        phrases[i] = zeros<uvec>( words_num );
        for( int k = 0 ; k < words_num ; k++ )
            phrases[i][k] =  i * words_num + k ;
        */

        cout << "# p" << i << ": " << phrases[ i ].t() << flush;

        if( curr_lab == n_out - 1 ) curr_lab = 0;
        else curr_lab += 1;

    }

    op.thread_num = thread_num;
    op.equal_train_rt = 0;
    op.epoch_num = 1;

    uint n_subgroups = 1;

    PT_GROUP ptg = PT_GROUP(
        phrases, labels, &rae, mp.fixed_tree, n_subgroups, op, t_pool, nullptr, ""
    );

    cout << "mean(rec_errs) p(y) costs" << endl;

    ptg.cache_pars( 2 + op.thread_num );
    ptg.get_cached_pars( 0 );

    if( op.run_type != lbfgs_batch && op.run_type != dliblbfgs_batch ){

        for( uint optr = 0; optr < 300; optr++ ){

            if( op.run_type != gradient_batch ) ptg.learn();
            else ptg.gradeval();
                // epoch num is set to 1 so should be ok
            if( op.run_type == gradient_batch ) ptg.put_back_in_model();
            ptg.predict();

            // rae.print_params_norms(2);
            cout << mean(ptg.rec_errs) << " " << mean(
                    probs_for_labels( ptg.probs, labels )
                ) << " " <<  mean( ptg.costs ) << endl;

            ptg.init_pt_buffers();
            ptg.init_derivatives();
            ptg.init_measures();
            
        }

    }else{

        ptg.subdivide(); // ##H NO,NO,NO you already do that ?
        uint rep_count = 0;

ptg.ptg_predict = &ptg;

#ifdef USE_LIBLBFGS
        while( ( ptg.lbfgs_retval != LBFGS_SUCCESS  || rep_count < 1 ) && rep_count < 6 ){
            ptg.learn();
            rep_count++;
        }
#endif

        ptg.rae->print_params_norms( 2 );
        // delete[] ptg.subgroups;
    }

    // minimize

    delete[] t_pool;

    return 0;
}

int test_onval( project_path pp, opt_params op, model_params mp ){

    struct dataset ds = read_dataset(
        pp.labels_p,
        pp.phrases_p,
        pp.train_p,
        pp.train_p,
        pp.train_p,
        1,
        100,
        100,
        100
    );

    run_dataset( ds, mp, op, pp );

    return( 0 );

}

int test_dumb(){

    int label = 0;
    int n_out = 3;
    int w_length = 2;
    double w_a = 1;
    double w_b = 2;

    AE rae = AE();
    rae.mp.w_length = 2;

    rowvec one = ones(1);
    rowvec i_h = randu<rowvec>( 2 * w_length);
    mat e_W_h = randu<mat>( 2 * w_length + 1, w_length); 
    mat e_W_c = randu<mat>( w_length + 1, n_out );
    mat e_W_r = randu<mat>( w_length + 1, 2 * w_length );

    rowvec z_h = rae.z_h_from_i_h( join_rows( i_h, one ), e_W_h );
    rowvec z_c = rae.z_c_from_i_c( join_rows( z_h, one ), e_W_c );
    rowvec z_r = rae.z_r_from_i_r( join_rows( z_h, one ), e_W_r );

    double p = z_c[ label ];
    double nll = - log( z_c[ label ] );
    double rec_err = rae.recc_err_from_i_h_z_r( i_h, z_r, w_a, w_b );

    cout << i_h << endl ;
    cout << z_h << endl ;
    cout << z_c << endl ;
    cout << z_r << endl ;
    cout << p << endl ;
    cout << nll << endl ;
    cout << rec_err << endl ;

    rand_product();

    return 0;
}

int test_auc(){

    uvec labels = randi<uvec>( 100, distr_param(0, 1) );
    vec probs = randu<vec>( 100 );

    cout << "auc:" << auc( probs, labels ) << endl;

    return 0;

}

int test_read( project_path pp ){

    struct dataset ds = read_dataset(
        pp.labels_p,
        pp.phrases_p,
        pp.train_p,
        "",
        "",
        1
    );

    cout << ds.labels( span(0, 10) ) << endl;

    for( int p_i = 0 ; p_i < 10 ; p_i++ ){
        cout << "phrase " << p_i  << ": " << ds.phrases[ p_i ].t() << endl;
    }

    cout << ds.train_idc << endl;
    cout << ds.labels( ds.train_idc( span(0, 10) ) ) << endl;

    for( int p_i = 0 ; p_i < 10 ; p_i++ ){
        cout << "train phrase " << p_i  << ": " <<
            ds.phrases[ ds.train_idc[ p_i ] ].t() << endl;
    }

    return 0;

}


int run( project_path pp, opt_params op, model_params mp ){

    struct dataset ds = read_dataset(
        pp.labels_p,
        pp.phrases_p,
        pp.train_p,
        pp.val_p,
        pp.test_p,
        pp.equalize,
        pp.n_train,
        pp.n_val,
        pp.n_test
    );

    if( pp.save_models_dir != "" ){
        
        struct stat info;

        if( stat( pp.save_models_dir.c_str(), &info ) != 0 ){
            std::string mkdir_cmd = std::string("mkdir ") +
                pp.save_models_dir;
            int succes = system( mkdir_cmd.c_str() );
        }
        else{ assert( info.st_mode & S_IFDIR ); }

    }

    run_dataset( ds, mp, op, pp );

    // delete[] ds.phrases;

    return( 0 );
}

int mem_test( uint do_reset ){

    mat a = zeros<mat>( 20000, 20000 );

    if( do_reset == 1 )
        a.reset();

    usleep( 5000000 );

    return( 0 );

}

int set_test_defaults( model_params * mp ){

    mp->lambda_reg_c = 0.00000;
    mp->lambda_reg_h = 0.00000;
    mp->lambda_reg_r = 0.00000;
    mp->lambda_reg_L = 0.00000;
    mp->alpha = 0.2;
    mp->w_length = 50;
    mp->fixed_tree = 0;

    return( 0 );
}

string help_text = \
"\n\
RAEcpp Copyright (C) 2015 Marc Vincent\n\
This program comes with ABSOLUTELY NO WARRANTY;\n\
This is free software, and you are welcome to redistribute it\n\
under certain conditions; read the LICENSE.txt file for details.\n\
\n\
\nUsage: run_rae [options]\n\
\n==============================================================================\
\nOptions:\n\
\n------------------------------------------------------------------------------\
\nPath and dataset:\n\
\n--phrases <path>\n\
    file format = contiguous int32 array corresponding to the data structure:\n\
    < m:long int >{ 1 }< < n:long int >{ 1 }< w:long int >{ n }> >{ m }\n\
    Where we adopt the following conventions:\n\
    * symbols within brackets correspond to elements of the array,\n\
    * quantifiers are enclosed in curly brackets.\n\
      ( and relate to data in the immediate left bracket. )\n\
    * m is the number of phrases in the corpus\n\
    * n is the number of words in a given phrase\n\
    * w is the index of a word in a given phrase\n\
\n--labels <path>\n\
    Path to a file containing labels, necessary for learning a model.\n\
    Labels should corresponds to the phrases given in the phrases.\n\
    Labels should be integers, -1 is considered missing\n\
    ( test labels can all be -1 ).\n\
\n--train <path>\n\
    path to train file containing the indices of phrases\n\
    ( from the phrases file ) that belong to the train sample.\n\
\n--val <path>\n\
    path to val file containing the indices of phrases\n\
    ( from the phrases file ) that belong to the\n\
    validation sample ( MANDATORY for non lBFGS training ).\n\
\n--test <path>\n\
    path to test file containing the indices of phrases\n\
    ( from the phrases file ) that belong to the test sample.\n\
\n--equalize_ds\n\
    equalize ( truncate ) the dataset prior to learning so that the number\n\
    of examples in each class is equal to the number of examples in the\n\
    class with the least examples.\n\
\n--no_equalize_rt\n\
\n--models_dir <path>\n\
    set the path of the directory where models are put/found.\n\
\n--predictions_dir <path>\n\
    set the path of the directory where predictions are put.\n\
\n--representations_dir <path>\n\
    set the path of the directory where phrases representations\n\
   ( ie. the representaion of the last node of the parse tree ) are put.\n\
\n--train_size <int>\n\
    truncate train set to this size\n\
\n--val_size <int>\n\
    truncate validation set to this size\n\
\n--test_size <int>\n\
    truncate test set to this size\n\
\n--tt\n\
    test objective minimization on a generated small random set of phrases\n\
\n------------------------------------------------------------------------------\
\nModel:\n\
\n--alpha <double>\n\
    set tradeoff between classification error and reconstruction error\n\
\n--lambda_reg_c <double>\n\
    set regularization for the matrix of classification layer weights\n\
\n--lambda_reg_h <double>\n\
    set regularization for the matrix of hidden layer weights\n\
\n--lambda_reg_r <double>\n\
    set regularization for the matrix of reconstruction layer weights\n\
\n--lambda_reg_L <double>\n\
    set regularization for the matrix of words embeddings\n\
\n--lambda_reg_L_sup <double>\n\
    set regularization for the matrix of words embeddings ( supervised )\n\
\n--log_square_loss\n\
    use logistic loss instead of softmax ( only for binary classification)\n\
\n--norm_rec\n\
    normalize reconstructions as in the original matlab implementation,\n\
    not in the article\n\
\n--single_words\n\
    do classification at the leaf nodes level and add this contribution to\n\
    loss and gradient evaluation. As in the original matlab implementation,\n\
    not in the article\n\
\n--nterm_w <double>\n\
    set weight of non terminal nodes ( default: 1 )\n\
\n--w_length <int>\n\
    set the length of vector representations ( default: 50 )\n\
\n--nl_is_tanh\n\
    set the non linearity to be tanh instead of normalized tanh\n\
\n--nl_is_relu\n\
    set the non linearity to be RELU instead of normalized tanh\n\
\n--nl_is_relu_norm\n\
    set the non linearity to be normalized RELU instead of normalized tanh\n\
\n--fixed_tree\n\
    set the binary parse tree to be fixed\n\
\n--predict_max\n\
\n------------------------------------------------------------------------------\
\nOptimization:\n\
\n--unsup_sup\n\
    !!! Only valid with lbfgs optimization !!!\n\
    do unsupervised learning and then supervised learning ( in batch mode )\n\
    as in the original matlab implementation\n\
\n--learning_rate <double>\n\
\n--rho <double>\n\
\n--max_epoch <int>\n\
\n--wait_min <int>\n\
    minimum number of minibatches to see before ending optimization.\n\
\n--wait_increase <double>\n\
    number by which to multiply the number of mini-batches necessary \n\
    to end the optimization when a new best performance is seen.\n\
\n--improvement_threshold <double>\n\
    a relative performance improvement of this much is considered significant\n\
    ( lies in [0,1] )\n\
\n--no_divide\n\
\n--divide_L_type\n\
\n--opt_type <string>\n\
\n--b_size <int>\n\
\n--opt_batch_num <int>\n\
\n--clip_gradients\n\
\n--clip_constant <double>\n\
\n--clip_deltas\n\
\n--clip_deltas_factor <double>\n\
\n( lBFGS only options ):\n\
\n--lbfgs_ls <string>\n\
    where string is one of: armijo, morethuente, wolfe, strongwolfe\n\
\n--lbfgs_delta <double>\n\
\n--lbfgs_delta <double>\n\
\n--lbfgs_maxiter <int>\n\
\n--lbfgs_maxls <int>\n\
\n--lbfgs_gtol <double>\n\
\n--lbfgs_xtol <double>\n\
\n--lbfgs_ftol <double>\n\
\n--lbfgs_wolfe <double>\n\
\n--lbfgs_m <int>\n\
\n--lbfgs_epsilon <double>\n\
\n------------------------------------------------------------------------------\
\nOther:\n\
\n--gradient_check\n\
\n--gradient_check_gp\n\
\n--gradient_check_all\n\
\n--replace_gradient\n\
\n--thread_num <int>\n\
\n--help\n\n";


int help(){

    cout << help_text;

    return 0;

}


int main(int argc, char** argv)
{
    
    replace_grad = 0;
    clip_gradients = 0;
    clip_constant = 1.5;
    transmission = 1;
    clip_deltas  = 0;
    clip_deltas_factor = 1;
    const_param_norm = 0;
    do_gradient_checks = 0;
    replace_grad = 0;
    verbose = 1; // only zero when you want to skip printing performances
            // e.g to use gradient check

    arma_rng::set_seed_random();

    // TEST actions available
    int do_test_dumb = 0;
    int do_test_auc = 0;
    int do_test_read = 0;
    int do_test_triplet = 0;
    int do_test_onval = 0;
    int do_print_help = 0;
    int tt_length = 0; // 2 words in test phrases
    uint tt_n_out = 3; // 3 labels for in test phrases

    // model parameters
    model_params mp;

    // optimization parameter
    opt_params op;
#ifdef USE_LIBLBFGS
    lbfgs_parameter_init( &( op.lbfgs_p ) );
#endif
    // dataset parameters
    project_path pp;

    // saving
    string models_dir;
    uint keep_sel_size = 10; // if keep size > 0 do not keep all, just the best

    for( int i = 1 ; (i<argc) && (int)*(*(argv + i)) == '-' ; i++ ){

        if( strcmp( argv[i], "--td" ) == 0 )
            do_test_dumb = 1;
            // do dumb tes// t
        else if( strcmp( argv[i], "--ta" ) == 0 )
            do_test_auc = 1;
            // test auc
        else if( strcmp( argv[i], "--tr" ) == 0 )
            do_test_read = 1;
            // test auc
        else if( strcmp( argv[i], "--tt" ) == 0 ){
            do_test_triplet = 1;
            set_test_defaults( &mp );
        }
        else if( strcmp( argv[i], "--to" ) == 0 )
            do_test_onval = 1;
        else if( strcmp( argv[i], "--tm" ) == 0 ){
            cout << "without reset:" << endl;
            mem_test( 0 );
            cout << "with reset:" << endl;
            mem_test( 1 );
            exit(0);
        }
        else if( strcmp( argv[i], "--phrases" ) == 0 ){
            ++i;
            pp.phrases_p = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--labels" ) == 0 ){
            ++i;
            pp.labels_p = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--train" ) == 0 ){
            ++i;
            pp.train_p = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--val" ) == 0 ){
            ++i;
            pp.val_p = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--test" ) == 0 ){
            ++i;
            pp.test_p = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--train_size" ) == 0 ){
            ++i;
            pp.n_train = atof( argv[ i ] ); }
        else if( strcmp( argv[i], "--val_size" ) == 0 ){
            ++i;
            pp.n_val = atof( argv[ i ] ); }
        else if( strcmp( argv[i], "--test_size" ) == 0 ){
            ++i;
            pp.n_test = atof( argv[ i ] ); }
        else if( strcmp( argv[i], "--models_dir" ) == 0 ){
            ++i;
            pp.save_models_dir = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--predictions_dir" ) == 0 ){
            ++i;
            pp.save_predictions_dir = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--representations_dir" ) == 0 ){
            ++i;
            pp.save_representations_dir = std::string( argv[ i ] ); }
        else if( strcmp( argv[i], "--equalize_ds" ) == 0 ){
            op.equal_train_rt = 1; }
        else if( strcmp( argv[i], "--no_equalize_rt" ) == 0 ){
            pp.equalize = 0; }
        else if( strcmp( argv[i], "--divide_rec_contrib" ) == 0 ){
            mp.divide_rec_contrib = true; }
        else if( strcmp( argv[i], "--full_model" ) == 0 ){
            mp.single_words = true;
            mp.norm_rec = true;
            }
        else if( strcmp( argv[i], "--single_words" ) == 0 ){
            mp.single_words = true;
            }
        else if( strcmp( argv[i], "--norm_rec" ) == 0 ){
            mp.norm_rec = true;
            }
        else if( strcmp( argv[i], "--no_weight_rec" ) == 0 ){
            mp.weight_rec = false;
            }
        else if( strcmp( argv[i], "--alpha" ) == 0 ){
            ++i;
            mp.alpha = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--lambda_reg_c" ) == 0 ){
            ++i;
            mp.lambda_reg_c = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--lambda_reg_h" ) == 0 ){
            ++i;
            mp.lambda_reg_h = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--lambda_reg_r" ) == 0 ){
            ++i;
            mp.lambda_reg_r = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--lambda_reg_L" ) == 0 ){
            ++i;
            mp.lambda_reg_L = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--lambda_reg_L_sup" ) == 0 ){
            ++i;
            mp.lambda_reg_L_sup = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--nterm_w" ) == 0 ){
            ++i;
            mp.nterm_w = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--w_length" ) == 0 ){
            ++i;
            mp.w_length = atoi( argv[i] );
            }
        else if( strcmp( argv[i], "--nl_is_tanh" ) == 0 )
            mp.z_type = z_is_tanh;
        else if( strcmp( argv[i], "--nl_is_relu" ) == 0 )
            mp.z_type = z_is_relu;
        else if( strcmp( argv[i], "--nl_is_relu_norm" ) == 0 )
            mp.z_type = z_is_relu_norm;
        else if( strcmp( argv[i], "--fixed_tree" ) == 0 ){
            mp.fixed_tree = 1;
            }
        else if( strcmp( argv[i], "--predict_max" ) == 0 )
            mp.predict_max = 1;
        else if( strcmp( argv[i], "--original_L_scale" ) == 0 )
            mp.original_L_scale = true;
        else if( strcmp( argv[i], "--log_square_loss" ) == 0 )
            mp.log_square_loss = true;
        else if( strcmp( argv[i], "--standard_L_scale" ) == 0 )
            mp.standard_L_scale = true;
        else if( strcmp( argv[i], "--no_direct_rec" ) == 0 )
            mp.keep_direct = false;
        else if( strcmp( argv[i], "--no_indirect_rec" ) == 0 )
            mp.keep_indirect = false;
        else if( strcmp( argv[i], "--perf_type" ) == 0 ){
            ++i;
            op.perf_type = string( argv[i] );
            if( bigger_perf_is_better.count( op.perf_type ) == 0 ){
                cerr << "performance measure: " << op.perf_type << 
                    " not known, please check the option" << endl;
                cerr << "known measures: ";
                for( auto kv: bigger_perf_is_better ){
                    cerr << kv.first << " ";
                }
                cerr << endl;
                exit( 1 );
            }
            }
        else if( strcmp( argv[i], "--learning_rate" ) == 0 ){
            ++i;
            op.learning_rate = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--rho" ) == 0 ){
            ++i;
            op.rho = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--wait_min" ) == 0 ){
            ++i;
            op.wait_min = atoi( argv[i] );
            }
        else if( strcmp( argv[i], "--wait_increase" ) == 0 ){
            ++i;
            op.wait_increase = atof( argv[i] );
            }
        else if( strcmp( argv[i], "--max_epoch" ) == 0 ){
            ++i;
            op.epoch_num = atof( argv[i] );
#ifdef USE_LIBLBFGS
            op.lbfgs_p.max_iterations = op.epoch_num;
#endif
            }
        else if( strcmp( argv[i], "--save_last_lbfgs" ) == 0 ){
            op.save_last_lbfgs = true;
            }
        else if( strcmp( argv[i], "--no_divide" ) == 0 ){
            op.divide = 0;
            }
        else if( strcmp( argv[i], "--divide_L_type" ) == 0 ){
            ++i;
            op.divide_L_type = atol( argv[i] );
            if( op.divide_L_type > 2 ){
                cout << "divide_L_type should be betw. 0 and 2" << endl;
                do_print_help = 1;
            }
            }
        else if( strcmp( argv[i], "--opt_type" ) == 0 ){
            ++i;
            if(
                run_types_str.find( std::string( argv[ i ] ) ) ==
                run_types_str.end() 
            ){
                cout << "no such run type:" << std::string( argv[ i ] ) << endl;
                do_print_help = 1;
            }else{
                op.run_type = run_types_str.at( std::string( argv[ i ] ) );
            }
            }
        else if( strcmp( argv[i], "--b_size" ) == 0 ){
            ++i;
            op.gradeval_batch_size = atol( argv[i] );
            }     
        else if( strcmp( argv[i], "--opt_batch_num" ) == 0 ){
            ++i;
            op.opt_batch_num = atol( argv[i] );
            }
        else if( strcmp( argv[i], "--thread_num" ) == 0 ){
            ++i;
            op.thread_num = atol( argv[i] );
            }
        else if( strcmp( argv[i], "--unsup_sup" ) == 0 ){
            op.unsup_sup = true;
            }
#ifdef USE_LIBLBFGS
        else if( strcmp( argv[i], "--lbfgs_ls" ) == 0 ){
            ++i;
            op.lbfgs_p.linesearch = lbfgs_ls_types.at( std::string( argv[ i ] ) );
            }
        else if( strcmp( argv[i], "--lbfgs_delta" ) == 0 ){
            ++i;
            op.lbfgs_p.delta = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_delta" ) == 0 ){
            ++i;
            op.lbfgs_p.delta = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_maxiter" ) == 0 ){
            op.lbfgs_p.max_iterations = atol( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_maxls" ) == 0 ){
            ++i;
            op.lbfgs_p.max_linesearch = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_gtol" ) == 0 ){
            ++i;
            op.lbfgs_p.gtol = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_xtol" ) == 0 ){
            ++i;
            op.lbfgs_p.xtol = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_ftol" ) == 0 ){
            ++i;
            op.lbfgs_p.ftol = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_wolfe" ) == 0 ){
            ++i;
            op.lbfgs_p.wolfe = atof( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_m" ) == 0 ){
            ++i;
            op.lbfgs_p.m = atol( argv[ i ] );
            }
        else if( strcmp( argv[i], "--lbfgs_epsilon" ) == 0 ){
            ++i;
            op.lbfgs_p.epsilon = atof( argv[ i ] );
            }
#endif
        else if( strcmp( argv[i], "--clip_gradients" ) == 0 )
            clip_gradients = 1;
        else if( strcmp( argv[i], "--clip_constant" ) == 0 ){
            ++i;
            clip_constant = atof( argv[i] );           
            }
        else if( strcmp( argv[i], "--clip_deltas" ) == 0 )
            clip_deltas = 1;
        else if( strcmp( argv[i], "--clip_deltas_factor" ) == 0 ){
            ++i;
            clip_deltas_factor = atof( argv[i] );           
            }
        else if( strcmp( argv[i], "--gradient_check" ) == 0 )
            do_gradient_checks = 1;
        else if( strcmp( argv[i], "--gradient_check_gp" ) == 0 )
            do_gradient_checks = 2;
        else if( strcmp( argv[i], "--gradient_check_all" ) == 0 )
            do_gradient_checks = 3;
        else if( strcmp( argv[i], "--replace_gradient" ) == 0 )
            replace_grad = 1;  
        else if( strcmp( argv[i], "--tt_length" ) == 0 ){
            ++i;
            tt_length = atoi( argv[i] );           
            }
        else if( strcmp( argv[i], "--tt_n_out" ) == 0 ){
            ++i;
            tt_n_out = atoi( argv[i] );           
            }
        else if( 
                strcmp( argv[i], "--help" ) == 0 ||
                strcmp( argv[i], "-h" ) == 0  ||
                strcmp( argv[i], "-?" ) == 0 
            ){
                do_print_help = 1;
                cout << " triggered here" << endl;
            }
        else if(
            strcmp( argv[i], "--replace_gradient" ) == 0 ||
            strcmp( argv[i], "-v" ) == 0
            )
            verbose += 1;  
        else{
            cout << "option not known: " << argv[i] << endl << endl;
            do_print_help = 1;
            }
    }

    if( argc < 2 )
        do_print_help = 1;

    if( do_print_help ){
        help();
        if( argc < 2 )
            cout << endl << "you should give some parameters." << endl;
        exit( 0 );
        }

    if( do_test_dumb )
        test_dumb();
    else if( do_test_auc )
        test_auc();
    else if( do_test_read )
        test_read( pp );
    else if( do_test_triplet )
        test_triplet( mp, op, tt_length, tt_n_out );
    else if( do_test_onval )
        test_onval( pp, op, mp );

    else
        run( pp, op, mp );

    return( 0 );
}
