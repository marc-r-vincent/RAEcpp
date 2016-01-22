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

#include <common.h> 
#include <AE.h>
#include <PARSE_TREE.h>
#include <PROGRESS_MGR.h>
#include <thread>
#include <util.h>
#include <PT_GROUP.h>

PT_GROUP::PT_GROUP(
    vector <uvec> phrases,
    uvec labels,
    AE * rae,
    uint fixed_tree,
    uint n_subgroups,
    opt_params op,
    std::thread * t_pool,
    PROGRESS_MGR * prog_mgr,
    std::string save_models_dir
)        
{

    uvec selection;
    populate(
        phrases,
        labels,
        selection,
        rae,
        fixed_tree,
        n_subgroups,
        op,
        t_pool,
        prog_mgr,
        save_models_dir
    );

};

PT_GROUP::PT_GROUP(
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
){
    populate(
        phrases,
        labels,
        selection,
        rae,
        fixed_tree,
        n_subgroups,
        op,
        t_pool,
        prog_mgr,
        save_models_dir
    );
};

PT_GROUP::~PT_GROUP()
{

    // if PT_GROUP was ever populated ( with n_phrases != - 1 )
    // delete allocated ressources
    /*
    if( n_phrases != -1  ){
        delete[] phrases;
        delete[] pts;
    }*/

    // del_subgroups();

};

int PT_GROUP::grad_load( grad_struct gs ){

    *dE_dL = gs.dE_dL;
    *dE_deWc = gs.dE_deWc;
    *dE_deWh = gs.dE_deWh;
    *dE_deWr = gs.dE_deWr;

    return 0;

}

int PT_GROUP::grad_add( grad_struct gs ){

    *dE_dL += gs.dE_dL;
    *dE_deWc += gs.dE_deWc;
    *dE_deWh += gs.dE_deWh;
    *dE_deWr += gs.dE_deWr;

    return 0;

}

int PT_GROUP::grad_save( grad_struct &gs ){

    gs.dE_dL = *dE_dL;
    gs.dE_deWc = *dE_deWc;
    gs.dE_deWh = *dE_deWh;
    gs.dE_deWr = *dE_deWr;

    return 0;

}

int PT_GROUP::grad_struct_fill( grad_struct &gs ){

    gs.dE_dL.fill( 0 );
    gs.dE_deWc.fill( 0 );
    gs.dE_deWh.fill( 0 );
    gs.dE_deWr.fill( 0 );

    return 0;

}

int PT_GROUP::print_update_norms( int nrm ){

    mat * param;

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    vector <std::string> params_names {
        "dE_deWr", "dE_deWc", "dE_deWh", "dE_dL" };

    for( uint pi = 0; pi < params.size(); ++pi ){

        param = params[ pi ];

        if( op.unsup_sup 
            and ( params_names[pi] == "dE_deWh" or
                params_names[pi] == "dE_deWr" ) 
        ){

            span left = span( 0, rae->mp.w_length - 1 );
            span right = span( rae->mp.w_length, 2 * rae->mp.w_length - 1 );

            vec param_lin_left =
                params_names[ pi ] == "dE_deWh" ?
                vectorise( (*param).rows( left ) ):
                vectorise( (*param).cols( left ) );

            vec param_lin_right =
                params_names[ pi ] == "dE_deWh" ?
                vectorise( (*param).rows( right ) ):
                vectorise( (*param).cols( right ) );


            cout << "# ||" << params_names[ pi ] << "_left|| " <<
                sqrt( dot( param_lin_left, param_lin_left ) ) << endl;

            cout << "# ||" << params_names[ pi ] << "_right|| " <<
                sqrt( dot( param_lin_right, param_lin_right ) ) << endl;

        }else{

            vec param_lin = vectorise( *param );

            cout << "# ||" << params_names[ pi ] << "|| " <<
                sqrt( dot( param_lin, param_lin ) ) << endl;
        }

    }

    return 0;

}

int PT_GROUP::populate(
    vector <uvec> phrases, uvec labels, uvec selection, AE * rae,
    uint fixed_tree, uint n_subgroups, opt_params op,
    std::thread * t_pool, PROGRESS_MGR * prog_mgr,
    std::string save_models_dir
){

    if( selection.n_elem == 0 ){
        selection = zeros<uvec>( labels.n_elem );
        for( uint i = 0; i < selection.n_elem; i++ )
            selection[ i ] = i;
    }
    
    this->n_phrases = selection.n_elem;
    this->rae = rae;
    this->labels = labels( selection );
    this->n_subgroups = n_subgroups;
    this->fixed_tree = fixed_tree;
    this->op = op;
    this->ptg_predict = nullptr;
    this->prog_mgr = prog_mgr;
    this->t_pool = t_pool;

    indices = zeros<uvec>( n_phrases );
    for( uint i = 0; i < n_phrases; i++ )
        indices[ i ] = i;

    wordcounts = zeros<uvec>( rae->v_size );

    // set best performance to the one given by progress manager
    if( prog_mgr != nullptr )
        this->prev_best_perf = prog_mgr->best_perfs( prog_mgr->best_perfs.n_rows - 1, 0 );

    for( int pt_i = 0; pt_i < n_phrases ; pt_i++ ){

        this->phrases.push_back( phrases[ selection[ pt_i ] ] );

        pts.push_back( PARSE_TREE(
            rae, this->phrases[ pt_i ], this->labels[ pt_i ], fixed_tree
        ) );

        for( int w_i = 0; w_i < pts[ pt_i ].n_words; w_i++ ){

            uint w_i_L = this->phrases[ pt_i ][ w_i ]; // index of the word in L
            wordcounts[ w_i_L ] += 1;

        }

    }

    n_words = sum( wordcounts );

    init_measures();

    return(0);

}

int PT_GROUP::cache_pars( uint n_dups ){
    // order: dE_dL, dE_deWh, dE_deWr, dE_deWc

    params_buff = new vector < vector< mat > >( 4, vector < mat >( n_dups ));
        
    uint n_rows;
    uint n_cols;
    for( uint i = 0; i < 4; i++ ){
        if( i == dE_dL_i ){
            n_rows = rae->v_size;
            n_cols = rae->mp.w_length;
        }else if( i == dE_deWh_i ){
            n_rows = 2  * rae->mp.w_length + 1;
            n_cols = rae->mp.w_length;
        }
        else if( i == dE_deWr_i ){
            n_rows = rae->mp.w_length + 1;
            n_cols = 2 * rae->mp.w_length;
        }
        else if( i == dE_deWc_i ){
            n_rows = rae->mp.w_length + 1;
            n_cols = rae->n_out;
        }

        for( uint j = 0; j < n_dups; j++ ){
            (*params_buff)[ i ][ j ] = zeros<mat>(n_rows, n_cols);
        }
    }

    is_cached = 1;

    return( 0 );

}

// parameters gradients cache array,
// one slot per concurrent active groups spawned
int PT_GROUP::get_cached_pars( uint cache_slot ){

    dE_dL = &( (*params_buff)[ dE_dL_i ][ cache_slot ] );
    dE_deWh = &( (*params_buff)[ dE_deWh_i ][ cache_slot ] );
    dE_deWr = &( (*params_buff)[ dE_deWr_i ][ cache_slot ] );
    dE_deWc = &( (*params_buff)[ dE_deWc_i ][ cache_slot ] );

    return( 0 );

}

int PT_GROUP::init_measures(){

    rec_errs = zeros<rowvec>(n_phrases);
    wnlls = zeros<rowvec>(n_phrases);
    costs = zeros<rowvec>(n_phrases);
    representations = zeros<mat>( n_phrases, rae->mp.w_length );
    probs = zeros<mat>( n_phrases, rae->n_out );

    return(0);
}

int PT_GROUP::init_derivatives(){
    // current group contribution to model
    *dE_dL = zeros<mat>( rae->v_size, rae->mp.w_length );
    *dE_deWh = zeros<mat>( 2  * rae->mp.w_length + 1, rae->mp.w_length );
    *dE_deWr = zeros<mat>( rae->mp.w_length + 1, 2 * rae->mp.w_length );
    *dE_deWc = zeros<mat>( rae->mp.w_length + 1, rae->n_out );

    return( 0 );

}

int PT_GROUP::fill_derivatives(){

    // current group contribution to model
    dE_dL->fill( 0 );
    dE_deWh->fill( 0 );
    dE_deWr->fill( 0 );
    dE_deWc->fill( 0 );

    return( 0 );

}

int PT_GROUP::init_pt_buffers(){

    for( int p_i = 0 ; p_i < n_phrases ; p_i++ )
         pts[ p_i ].init_buffers();

    return( 0 );

}

int PT_GROUP::accumulate_grad(
    PT_GROUP * ptg
){

    *dE_dL += *(ptg->dE_dL);
    *dE_deWc += *(ptg->dE_deWc);
    *dE_deWh += *(ptg->dE_deWh);
    *dE_deWr += *(ptg->dE_deWr);

    return( 0 );

}

int PT_GROUP::accumulate_phrase_grad(
    PARSE_TREE * pt
){

    if( pt->n_words > 1 || rae->mp.single_words ){

        dE_dL->rows( pt->input_array ) += pt->dE_dL_phrase;
        *dE_deWc += pt->dE_deWc;
        *dE_deWh += pt->dE_deWh;
        *dE_deWr += pt->dE_deWr;

    }

    return( 0 );

}

int PT_GROUP::divide_grad(){

    divide_grad( wordcounts, n_phrases );

    return 0;

}

int PT_GROUP::divide_grad( uvec wordcounts, double n_phrases ){

    if( op.divide_L_type == 1 || op.divide_L_type == 2 ){
        for( uword w_i = 0; w_i < wordcounts.n_elem; w_i++ ){

            if( wordcounts[ w_i ] != 0 ){

                for( uword L_col_i = 0; L_col_i < rae->L.n_cols; L_col_i++ ){
                    (*dE_dL)( w_i, L_col_i ) = (*dE_dL)( w_i, L_col_i ) /
                        wordcounts[ w_i ];
                        // divide contribution of dE_dL by the number of
                        // phrases that have the given word ???
                }
                
            }
        }
    }

    if( op.divide_L_type == 0 || op.divide_L_type == 2 ) *dE_dL =
        *dE_dL/ n_phrases;

    *dE_deWh = *dE_deWh / n_phrases;
    *dE_deWr = *dE_deWr / n_phrases;
    *dE_deWc = *dE_deWc / n_phrases;

    return 0;

}

int PT_GROUP::divide_cost(){
    
    if( op.unsup_sup ){

        if( verbose >= 2 ){
            cout << "n_words:" <<  n_words << endl;
            cout << "n_phrases: " << n_phrases << endl;
        }

        double unsup_cost = 0, sup_cost = 0;

        if( rae->mp.unsup ){
            unsup_cost = sum( rec_errs ) / ( n_words - n_phrases );
            // sup_cost = sum( wnlls ) / ( n_words - n_phrases );
        }else{
            // unsup_cost = sum( rec_errs ) / n_phrases;
            sup_cost = sum( wnlls ) / n_phrases;
        }

        if( verbose != 0 ){
            cout << "# unsup cost: " << unsup_cost << endl; 
            cout << "# sup cost: " << sup_cost << endl;
        }

        cost = unsup_cost + sup_cost;

    }
    else{
        if( op.divide_L_type == 1 || op.divide_L_type == 2 ){
           // TODO equivalent
        }
        cost = sum( costs ) / n_phrases; 
    }


    return 0;

}

int PT_GROUP::grad_multiply( double n ){

    *dE_dL = *dE_dL * n;
    *dE_deWh = *dE_deWh * n;
    *dE_deWr = *dE_deWr * n;
    *dE_deWc = *dE_deWc * n;

    return 0;

}

int PT_GROUP::addto_grad( double n ){

    *dE_dL = *dE_dL + n;
    *dE_deWh = *dE_deWh + n;
    *dE_deWr = *dE_deWr + n;
    *dE_deWc = *dE_deWc + n;

    return 0;

}

int PT_GROUP::grad_neg(){

    *dE_dL = -*dE_dL;
    *dE_deWh = -*dE_deWh;
    *dE_deWr = -*dE_deWr;
    *dE_deWc = -*dE_deWc;

    return 0;

}

int PT_GROUP::put_back_in_model(){

    uvec all = zeros<uvec>( rae->v_size );
    for( uint i = 0 ; i < rae->v_size ; i++ ) all[ i ] = i;

    rae->put_back_in_model( dE_dL, dE_deWh, dE_deWr, dE_deWc, all, 1 );

    return 0;

}


int PT_GROUP::copy_grad_to_rowvec( rowvec * theta_grad ){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*theta_grad)( theta_idx ) = (*par)( r_i, c_i );
                theta_idx++;
            }

    }

    return( 0 );

}

int PT_GROUP::copy_rowvec_to_grad( rowvec theta_grad ){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*par)( r_i, c_i ) = theta_grad( theta_idx );
                theta_idx++;
            }

    }

    return( 0 );

}

int PT_GROUP::copy_grad_to_dlibvec( column_vector * theta_grad ){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*theta_grad)( theta_idx ) = (*par)( r_i, c_i );
                theta_idx++;
            }

    }

    return( 0 );

}

int PT_GROUP::copy_dlibvec_to_grad( column_vector theta_grad ){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*par)( r_i, c_i ) = theta_grad( theta_idx );
                theta_idx++;
            }

    }

    return( 0 );

}


double PT_GROUP::dlib_batch_cost( const column_vector& m ){

    cout << "# dlib batch cost" << endl;

    this->rae->copy_from_dlibvec( m ); // copy colvec
    this->predict();

    return this->cost;
};

const column_vector PT_GROUP::dlib_gradeval( const column_vector& m ){

    cout << "# dlib gradev" << endl;

    this->rae->copy_from_dlibvec( m ); // copy colvec
    this->gradeval();
    column_vector grad = column_vector( this->rae->n_params );
    this->copy_grad_to_dlibvec( &grad );
    
    cout << "# grad norm: " << sqrt( dlib::sum( dlib::pow( grad, 2 ) ) ) << endl;

    if( this->ptg_predict != nullptr ){
        this->ptg_predict->predict();
        this->print_update_norms(2);
    }

    if( this->curr_lbfgs < this->n_subgroups - 1 )
        this->curr_lbfgs++;
    else
        this->curr_lbfgs=0;

    this->last_lbfgs = this->curr_lbfgs - 1;

    if( this->prog_mgr != nullptr )
        mat best_perfs = this->prog_mgr->save_progress(
            this->epoch_index, 0, this->ptg_predict->ppl );

    epoch_index++;
    return( grad );
}

// * instance will be a PT_GROUP object ( = caller, a bit convoluted but needed )
// * x will be a linearized version of theta = e_W_c, e_W_r, e_W_h

#ifdef USE_LIBLBFGS
// TODO template ?
int PT_GROUP::copy_grad_to_1Darray( lbfgsfloatval_t * theta_grad ){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                theta_grad[ theta_idx ] = (*par)( r_i, c_i );
                theta_idx++;
            }

    }

    return( 0 );

}

int PT_GROUP::copy_1Darray_to_grad( lbfgsfloatval_t * theta_grad ){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };
    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*par)( r_i, c_i ) = theta_grad[ theta_idx ];
                theta_idx++;
            }

    }

    return( 0 );

}

lbfgsfloatval_t PT_GROUP::lbfgs_evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{

    PT_GROUP * ptg = ( PT_GROUP * ) instance;

    int has_nn = 0;
    for( uint i = 0; i < ptg->rae->n_params; i++ )
        if( *( x + i ) == *( x + i ) ) has_nn++;

    if( has_nn == 0 )
        cout << "# all parameter values are null" << endl;

    int i;
    lbfgsfloatval_t fx = 0.0;

    double mean_cost =  0;
    uint non_null_phrases = 0;

    ptg->rae->copy_from_1Darray( x );

    cout << "# lr : " << ptg->op.learning_rate << endl;
    cout << "# testing step: " << step << endl;

    ptg->gradeval();
    fx = ptg->cost;

    // copy gradient and fill *g, get previous loss function back
    ptg->copy_grad_to_1Darray( g );

    cout << endl << "# done epoch " << ptg->epoch_index << "." << endl;
    cout << "# fx: " << fx << endl; 

    ptg->epoch_index++;
    ptg->last_lbfgs = ptg->curr_lbfgs;

    return fx;
}

int PT_GROUP::lbfgs_progress(
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
    )
{

    printf("# Iteration %d:\n", k);
    printf("# fx progress = %f \n", fx);
    printf("# xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");

    PT_GROUP * ptg = ( PT_GROUP * ) instance;
    if( ptg->ptg_predict != nullptr ){
        ptg->ptg_predict->predict();
        ptg->print_update_norms(2);
    }

    if( ptg->curr_lbfgs < ptg->n_subgroups - 1 )
        ptg->curr_lbfgs++;
    else
        ptg->curr_lbfgs=0;

    ptg->last_lbfgs = ptg->curr_lbfgs - 1;

    if( ptg->prog_mgr != nullptr ){
        if( ptg->ptg_predict != nullptr ){
            mat best_perfs = ptg->prog_mgr->save_progress(
                ptg->epoch_index, 0, ptg->ptg_predict->ppl );
        }
        else{
            perf_probs_labels_rep ppl;
            ppl.perf = k;
            ppl.perf_type = "epoch";
            mat best_perfs = ptg->prog_mgr->save_progress(
                ptg->epoch_index, 0, ppl );
        }

    }

    return 0;
}

int check_lbfgs_ret( int ret ){

    switch( ret )
    {
        case LBFGSERR_CANCELED:{ cout << "LBFGSERR_CANCELED" << endl; break; }
        case LBFGSERR_INCORRECT_TMINMAX:{ cout << "LBFGSERR_INCORRECT_TMINMAX" << endl; break; }
        case LBFGSERR_INCREASEGRADIENT:{ cout << "LBFGSERR_INCREASEGRADIENT" << endl; break; }
        case LBFGSERR_INVALIDPARAMETERS:{ cout << "LBFGSERR_INVALIDPARAMETERS" << endl; break; }
        case LBFGSERR_INVALID_DELTA:{ cout << "LBFGSERR_INVALID_DELTA" << endl; break; }
        case LBFGSERR_INVALID_EPSILON:{ cout << "LBFGSERR_INVALID_EPSILON" << endl; break; }
        case LBFGSERR_INVALID_FTOL:{ cout << "LBFGSERR_INVALID_FTOL" << endl; break; }
        case LBFGSERR_INVALID_GTOL:{ cout << "LBFGSERR_INVALID_GTOL" << endl; break; }
        case LBFGSERR_INVALID_LINESEARCH:{ cout << "LBFGSERR_INVALID_LINESEARCH" << endl; break; }
        case LBFGSERR_INVALID_MAXLINESEARCH:{ cout << "LBFGSERR_INVALID_MAXLINESEARCH" << endl; break; }
        case LBFGSERR_INVALID_MAXSTEP:{ cout << "LBFGSERR_INVALID_MAXSTEP" << endl; break; }
        case LBFGSERR_INVALID_MINSTEP:{ cout << "LBFGSERR_INVALID_MINSTEP" << endl; break; }
        case LBFGSERR_INVALID_N:{ cout << "LBFGSERR_INVALID_N" << endl; break; }
        case LBFGSERR_INVALID_N_SSE:{ cout << "LBFGSERR_INVALID_N_SSE" << endl; break; }
        case LBFGSERR_INVALID_ORTHANTWISE:{ cout << "LBFGSERR_INVALID_ORTHANTWISE" << endl; break; }
        case LBFGSERR_INVALID_ORTHANTWISE_END:{ cout << "LBFGSERR_INVALID_ORTHANTWISE_END" << endl; break; }
        case LBFGSERR_INVALID_ORTHANTWISE_START:{ cout << "LBFGSERR_INVALID_ORTHANTWISE_START" << endl; break; }
        case LBFGSERR_INVALID_TESTPERIOD:{ cout << "LBFGSERR_INVALID_TESTPERIOD" << endl; break; }
        case LBFGSERR_INVALID_WOLFE:{ cout << "LBFGSERR_INVALID_WOLFE" << endl; break; }
        case LBFGSERR_INVALID_XTOL:{ cout << "LBFGSERR_INVALID_XTOL" << endl; break; }
        case LBFGSERR_INVALID_X_SSE:{ cout << "LBFGSERR_INVALID_X_SSE" << endl; break; }
        case LBFGSERR_LOGICERROR:{ cout << "LBFGSERR_LOGICERROR" << endl; break; }
        case LBFGSERR_MAXIMUMITERATION:{ cout << "LBFGSERR_MAXIMUMITERATION" << endl; break; }
        case LBFGSERR_MAXIMUMLINESEARCH:{ cout << "LBFGSERR_MAXIMUMLINESEARCH" << endl; break; }
        case LBFGSERR_MAXIMUMSTEP:{ cout << "LBFGSERR_MAXIMUMSTEP" << endl; break; }
        case LBFGSERR_MINIMUMSTEP:{ cout << "LBFGSERR_MINIMUMSTEP" << endl; break; }
        case LBFGSERR_OUTOFINTERVAL:{ cout << "LBFGSERR_OUTOFINTERVAL" << endl; break; }
        case LBFGSERR_OUTOFMEMORY:{ cout << "LBFGSERR_OUTOFMEMORY" << endl; break; }
        case LBFGSERR_ROUNDING_ERROR:{ cout << "LBFGSERR_ROUNDING_ERROR" << endl; break; }
        case LBFGSERR_UNKNOWNERROR:{ cout << "LBFGSERR_UNKNOWNERROR" << endl; break; }
        case LBFGSERR_WIDTHTOOSMALL:{ cout << "LBFGSERR_WIDTHTOOSMALL" << endl; break; }
        case LBFGS_ALREADY_MINIMIZED:{ cout << "LBFGS_ALREADY_MINIMIZED" << endl; break; }
        case LBFGS_STOP:{ cout << "LBFGS_STOP" << endl; break; }
        case LBFGS_SUCCESS:{ cout << "LBFGS_SUCCESS" << endl; break; }
    }

    return 0;
}
#endif

int PT_GROUP::learn_lbfgs(){

#ifdef USE_LIBLBFGS
    theta = lbfgs_malloc( rae->n_params );

    lbfgsfloatval_t fx;

    rae->copy_to_1Darray( theta );

    cout << endl << "# starting:" << endl;
    lbfgs_retval = lbfgs(
        rae->n_params,
        theta,
        &fx,
        this->lbfgs_evaluate,
        this->lbfgs_progress,
        this,
        &( op.lbfgs_p )
    );

    cout << endl << "# at end:" << endl;
    cout << "# lbfgs return:" << lbfgs_retval << endl;
    check_lbfgs_ret( lbfgs_retval );

    if( ptg_predict != nullptr )
        ptg_predict->predict();

    cout << endl;

    if( prog_mgr != nullptr && op.save_last_lbfgs ){
        cout << "# saving last lbfgs model" << endl;
        if( ptg_predict != nullptr ){
            mat best_perfs = prog_mgr->save_progress(
                epoch_index, 0, ptg_predict->ppl );
        }
        else{
            perf_probs_labels_rep ppl;
            ppl.perf = epoch_index;
            ppl.perf_type = "epoch";
            mat best_perfs = prog_mgr->save_progress(
                epoch_index, 0, ppl );
        }

    }
#else

    cout << "libLBFGS is not used, please provide libLBFGS and re-install RAEcpp" <<
        endl;

#endif


    return( 0 );

}

int PT_GROUP::learn_dliblbfgs(){

    using namespace std::placeholders;

    // create cost function from object ( that will take m as only arg )
    function<double( const column_vector & )> cost_function =
        std::bind( &PT_GROUP::dlib_batch_cost, this, _1 );
    function<const column_vector( const column_vector & )> grad_function =
        std::bind( &PT_GROUP::dlib_gradeval, this, _1 );
    
    column_vector starting_point = column_vector( this->rae->n_params );
    this->rae->copy_to_dlibvec( &starting_point );

#ifdef USE_DLIB
    dlib::find_min(
        dlib::lbfgs_search_strategy(10),  // The 10 here is basically a measure of how much memory L-BFGS will use.
        dlib::objective_delta_stop_strategy(1e-7).be_verbose(),  // Adding be_verbose() causes a message to be 
                                                                // printed for each iteration of optimization.
        cost_function,
        grad_function,
        starting_point,
        -1.
        );
#else
    cout << "DLIB is not used, please provide DLIB and re-install RAEcpp" <<
        endl;
    return 0;

#endif

    return 0;

}

int PT_GROUP::gradient_check(){

    uint old_verbose = verbose; verbose = 0;

    double epsilon = 0.000001;

    rowvec rec_errs_or = rec_errs;
    rowvec wnlls_or = wnlls;
    double cost_or = cost;

    cout << "# fx gc: " << cost << endl;

    // get matrices of the same shape than params

    // for all value of all parameters
    
    vector <mat *> params {
        &( rae->e_W_c ), &( rae->e_W_h ), &( rae->e_W_r ), &( rae->L )
    };
    vector <std::string> params_names { "e_W_c", "e_W_h", "e_W_r", "L" };
    vector <mat *> bp_gradients = { dE_deWc, dE_deWh, dE_deWr, dE_dL };
    vector <mat> gradients( 4 );

    // compute gradients numerically for all W and L
    for( uint pi = 0; pi < 4; pi++ ){

        gradients[ pi ] = zeros( params[pi]->n_rows, params[pi]->n_cols );

        for( uint ri = 0; ri < params[pi]->n_rows; ri++ ){

            for( uint ci = 0; ci < gradients[pi].n_cols; ci++ ){
                
                // store current param value
                double old_val = ( * params[ pi ] )( ri, ci );
                // add epsilon to current param
                ( * params[ pi ] )( ri, ci ) += epsilon;
                // get cost
                if( level == 0 )  predict();
                else{
                    predict_batch();
                    divide_cost();
                    cost += rae->regularizer_for_cost();
                }
                double cost_diff = cost;
                // substract epsilon to param
                ( * params[ pi ] )( ri, ci ) = old_val - epsilon;
                if( level == 0 )  predict();
                else{
                    predict_batch();
                    divide_cost();
                    cost += rae->regularizer_for_cost();
                }
                cost_diff -= cost;
                // divide by epsilon diff
                cost_diff = cost_diff / ( 2 * epsilon );
                // store gradient eval
                gradients[ pi ]( ri, ci ) = cost_diff;
                // reset params
                ( * params[ pi ] )( ri, ci ) = old_val;

            }

        }

    };

    // compute differences
    double diff_norm;
    double tot_norm;

    for( uint pi = 0; pi < 4; pi++ ){

        tot_norm = norm( *params[ pi ], 2 );

        diff_norm = norm( gradients[ pi ] - *bp_gradients[ pi ], 2 );

        cout << "gradient check:" << endl << 
            "diff_norm: " << params_names[ pi ] << " " << diff_norm <<
            ", should be << w.r.t. tot_norm: " << tot_norm << endl << flush;

        if( replace_grad == 1 ) *(  bp_gradients[ pi ] ) = gradients[ pi ];

    }
    
    double t_norm_bp = 0;
    double t_norm_gd = 0;
    for( uint pi = 0; pi < 4; pi++ )
        for( uint ri = 0; ri < params[pi]->n_rows; ri++ )
            for( uint ci = 0; ci < gradients[pi].n_cols; ci++ ){
                t_norm_bp += abs( ( * bp_gradients[pi] )( ri, ci) );
                t_norm_gd += abs( ( gradients[pi] )( ri, ci ) );
            }
    cout << "# bp norm: " << t_norm_bp << endl;
    cout << "# gp norm: " << t_norm_gd << endl;

    cout << endl << flush;

    verbose = old_verbose;

    return(0);
    
}

int PT_GROUP::gradeval(){

    cout << endl << "# start epoch " << epoch_index << endl;

    cout << "# one '|' is a batch of size " << op.gradeval_batch_size <<
        " ..." << endl << endl;

    // set n_subgroups ?
    // subdivide into minibatches  PT_GROUP ??
    n_subgroups = ceil( (double) n_phrases / op.gradeval_batch_size );

    uvec curr_wc = zeros<uvec>( rae->v_size );
    double nphr = 0;

    // init derivatives to zero
    fill_derivatives();

    cout << "# divide main group to " << n_subgroups << " subgroups."
        << endl;

    subdivide();

    for( uint thiss_i = 0; thiss_i < n_subgroups; thiss_i ++ )
        subgroups[ thiss_i ].n_subgroups = op.thread_num;

    grad_struct grads_unsup, grads_sup;
    double reg_sup, reg_unsup, reg;
    double cost_sup = 0, cost_unsup = 0;

    if( op.unsup_sup ){ rae->mp.unsup = true; rae->mp.sup = false; }

    for( string pass: { "unsupervised", "supervised" } ){

        if( verbose >= 2 ) rae->print_params_norms(2);

        if( op.unsup_sup ) cout << "# doing pass: " << pass << endl;

        for( uint ptg_i = 0; ptg_i < n_subgroups; ptg_i ++ )
        {

            cout << "|" << std::flush;

            subgroups[ ptg_i ].gradeval_batch();

            accumulate_grad( &( subgroups[ ptg_i ] ) );
            subgroups[ ptg_i ].fill_derivatives();
            curr_wc += subgroups[ ptg_i ].wordcounts;
            nphr += subgroups[ ptg_i ].n_phrases;

        }
        
        cout << endl; // post | line return

        if( not op.unsup_sup ) break;
        else{

            if( verbose >= 2 ) print_update_norms(2);

            cout << endl;
            if( rae->mp.unsup ){
                grad_multiply( 1. / ( n_words - n_phrases ) );
            }
            else{
                grad_multiply( 1. / n_phrases );
            }

            rae->regularize_grad(
                    dE_dL, dE_deWc, dE_deWh, dE_deWr, zeros<uvec>(0) );

            if( rae->mp.unsup ){
                grad_save( grads_unsup );
                reg_unsup = rae->regularizer_for_cost();
            }else{
                grad_save( grads_sup );
                reg_sup = rae->regularizer_for_cost();
            }

            // ##H collect cost here
            collect_costs();
            // divide cost
            divide_cost();

            if( rae->mp.unsup ){
                cost += reg_unsup;
                cost_unsup = cost;
            }
            else{
                cost += reg_sup;
                cost_sup = cost;
            }

            if( rae->mp.unsup ) rae->mp.unsup = false; rae->mp.sup = true;

            fill_derivatives();
        }

    }

    if( not  op.unsup_sup ){ collect_costs(); divide_cost(); }
    else{ 
        cost = cost_sup + cost_unsup;
        grad_add( grads_unsup );
        grad_add( grads_sup );
    }

    if( clip_gradients == 1 ) clip_grad();

    if( not op.unsup_sup ){

        if( op.divide == 1 ) divide_grad(); 
        rae->regularize_grad(
                dE_dL, dE_deWc, dE_deWh, dE_deWr, zeros<uvec>(0) );

        reg = rae->regularizer_for_cost();

    }


    if( do_gradient_checks >= 2 ) gradient_check();
    
    return 0;

}

// pass over a minibatch eventually divided into sub-minibatches for threading
int PT_GROUP::gradeval_batch(){

    // subdivide into sub minibatches  PT_GROUP ??
    if( ! op.unsup_sup || ( op.unsup_sup and rae->mp.unsup ) )
        get_cached_pars( 1 );

    if( op.run_type == sgd_phrase ){ op.thread_num = 1; n_subgroups = 1; }
    else{ n_subgroups = op.thread_num; }

    subdivide();

    for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i++ ){

        subgroups[ ptgs_i ].get_cached_pars( ptgs_i + 2 );

        if( n_subgroups > 1 )
            t_pool[ ptgs_i ] = thread( PT_GROUP::gradeval_batch_parts,
                &( subgroups[ ptgs_i ]) );
        else
            PT_GROUP::gradeval_batch_parts( &( subgroups[ ptgs_i ] ) );

    }

    if( n_subgroups > 1 )
        for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i++ )
            t_pool[ ptgs_i ].join();

    collect_grads( (uint) 0, n_subgroups - 1 ); // you have
        // to do this because buffers for grad are re-used after
 
    if( op.run_type != lbfgs_batch && op.run_type != dliblbfgs_batch)
    { collect_costs(); } // ##H TODO ERASE ? 

    if(
        op.run_type == sgd_minibatch || op.run_type == adagrad_minibatch ||
        op.run_type == sgdmom_minibatch || op.run_type == adadelta_minibatch
    )
    {

        divide_grad();
        rae->regularize_grad( dE_dL, dE_deWc, dE_deWh, dE_deWr, zeros<uvec>(0) );

        if( do_gradient_checks >= 2 ) gradient_check();

        if( op.run_type == adagrad_minibatch ) update_adagrad();

        if( op.run_type == adadelta_minibatch ) update_adadelta();

        if( op.run_type == sgdmom_minibatch ) update_momentum();

        if( op.run_type == sgd_minibatch ) update_vanilla();

        put_back_in_model();

    }

    return 0;

}


// pass over a subminibatch within a single thread
void PT_GROUP::gradeval_batch_parts( PT_GROUP * ptg ){
#ifdef ProfilerStart
    ProfilerStart( "rae_gperf.log" );
#endif
    // gradeval_batch contains:
    // compute_gradients
    // put_back_in_model | accumulate
    // a lbfgs call should repetitively call these two
    ptg->fill_derivatives();

    for( int p_i = 0 ; p_i < ptg->n_phrases ; p_i++ ) {
    
        // TODO considering model phrases under
        // two elements shouldn't be considered
        // unless we explicitely state otherwise
        if( ptg->pts[ p_i ].n_words > 1 ) {

            ptg->pts[ p_i ].init_buffers();

            ptg->pts[ p_i ].pass_fwd(); // set tree and forward compute outputs

            ptg->pts[ p_i ].pass_bwd(); // compute derivatives of the parameters

            ptg->rec_errs[ p_i ] = sum( ptg->pts[ p_i ].rec_errs );

            ptg->wnlls[ p_i ] = sum( ptg->pts[ p_i ].wnlls );

            ptg->costs[ p_i ] = ptg->pts[ p_i ].cost ;

            if( do_gradient_checks == 1 || do_gradient_checks == 3 )
                ptg->pts[ p_i ].gradient_check();

            if( ptg->op.run_type == sgd_phrase ){

                if( clip_gradients == 1 ) ptg->pts[ p_i ].clip_grad();

                ptg->rae->regularize_grad(
                    &( ptg->pts[ p_i ].dE_dL_phrase ),
                    &( ptg->pts[ p_i ].dE_deWc ),
                    &( ptg->pts[ p_i ].dE_deWh ),
                    &( ptg->pts[ p_i ].dE_deWr ),
                    ptg->pts[ p_i ].input_array                       
                );

                mat up_dL_phr =
                    - ptg->op.learning_rate * ptg->pts[ p_i ].dE_dL_phrase;
                mat up_dWh = - ptg->op.learning_rate * ptg->pts[ p_i ].dE_deWh;
                mat up_dWr = - ptg->op.learning_rate * ptg->pts[ p_i ].dE_deWr;
                mat up_dWc = - ptg->op.learning_rate * ptg->pts[ p_i ].dE_deWc;

                ptg->rae->put_back_in_model(
                    &up_dL_phr,
                    &up_dWh,
                    &up_dWr,
                    &up_dWc,
                    ptg->pts[ p_i ].input_array,
                    1
                );

                ptg->update_lr();

            }
            else{ ptg->accumulate_phrase_grad( &( ptg->pts[ p_i ] ) ); }

        }
        ptg->pts[ p_i ].empty_buffers(); // set tree and forward compute outputs
    }

#ifdef ProfilerStop
    ProfilerStop();
#endif

}

int PT_GROUP::learn_not_lbfgs(){

    if( 
        op.run_type == adadelta_minibatch || op.run_type == adagrad_minibatch ||
        op.run_type == sgdmom_minibatch
    ){
        grad_accu = new rowvec( rae->n_params );
        grad_accu->fill( 0 );
        param_accu = new rowvec( rae->n_params );
        param_accu->fill( 0 );
    }

    // several rounds until criterion or max round is satisfied
    // if subgroups with only one thread then do only one round and then
    // finish
    uint end_training = 0;
    uint tot_mb = 0; // minibatches seen so far
    uint wait_value = op.wait_min;
    for( uint epoch = 0; epoch < op.epoch_num ; epoch++ )
    {

        cout << endl << "# epoch: " <<  epoch << endl << std::flush;

        uvec selection;

        // TODO bad, should be at every epoch and should reset is_subdivided
        // ( memory leak ? )
        if( epoch == 0 )
            if( op.equal_train_rt == 1 ){
                
                cout << "# balancing train set at runtime by random sampling..."
                    << endl;
                selection = equalize( labels );

            }

        cout << "# one '|' is a batch of size " << op.gradeval_batch_size <<
            " ..." << endl << endl;

        // set n_subgroups ?
        // subdivide into minibatches  PT_GROUP ??
        n_subgroups = ceil( (double) n_phrases / op.gradeval_batch_size );

        cout << "# divide main group to " << n_subgroups << " subgroups."
            << endl;

        subdivide();

        for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i ++ )
            subgroups[ ptgs_i ].n_subgroups = op.thread_num;

        uint last_mb = 0; // minibatches seen so far within epoch
        for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i ++ )
        {

            cout << "|" << std::flush;

            subgroups[ ptgs_i ].gradeval_batch();

            if( op.run_type == gradient_batch )
                accumulate_grad( &subgroups[ ptgs_i ] );

            if(
                ( 
                    ptgs_i == last_mb + op.opt_batch_num - 1 ||
                    ptgs_i == n_subgroups - 1
                ) &&
                ptg_predict != nullptr
            ){

                if( op.run_type == gradient_batch ){
                    cout << endl << "from: " << last_mb << " to: " <<
                        min( last_mb + op.opt_batch_num - 1, n_subgroups - 1 ) << endl;
                    // this->collect_grads(
                    //     last_mb, min( last_mb + op.opt_batch_num - 1, n_subgroups - 1 ) 
                    //);  ##H You can't quite do that as there is only one ( cached params ? )
                    //    ##H for all batch
                    this->print_update_norms( 2 );

                    if( clip_gradients == 1 ) this->clip_grad();

                    this->print_update_norms( 2 );

                    this->divide_grad();

                    if( do_gradient_checks >= 2 ) this->gradient_check();

                    this->put_back_in_model();
                }

                cout <<  endl << "# lr: " << op.learning_rate << endl;
                cout << "# predicting for validation set at epoch's batch: " << ptgs_i <<
                    " ,tot batch: " << tot_mb << endl;
                if( ptgs_i == n_subgroups - 1 ){
                    cout << endl << "# end of epoch " << epoch << "." << endl <<
                        std::flush;

                    cout << endl;
                    rae->print_params_norms(2);
                    cout << "# L: " << norm( rae->L, 2 ) << endl;
                }

                ptg_predict->predict();
                last_mb = ptgs_i + 1;
                tot_mb += ptgs_i + 1;

                if( prog_mgr != nullptr )
                    mat best_perfs = prog_mgr->save_progress(
                        epoch, ptgs_i, ptg_predict->ppl );

                // if there is an improvement wait a little more before ending
                // the optimization
                if( 
                    ptg_predict != nullptr  &&
                    perf_a_better_than_b(
                        op.perf_type,
                        ptg_predict->ppl.perf,
                        prev_best_perf * ( 1 + op.improvement_threshold )
                    )
                ){ 
                    wait_value = ( int )( max( (double) op.wait_min, tot_mb * op.wait_increase ) + 0.5 );
                    cout << "# validation performance increased" << endl;
                    cout << "# waiting longer before ending training" << 
                        " ( till batch " << wait_value << " )" << endl; 
                    cout << "# previous best performance: " << prev_best_perf << endl;
                    cout << "# best performance: " << ptg_predict->ppl.perf << endl;
                    prev_best_perf = ptg_predict->ppl.perf;
                }

                // if we are at training limit, end training
                if( wait_value != - 1 && tot_mb >= wait_value ){
                    cout << "# no performance increase in a while: ending training" << endl;
                    cout << "# called at tot_batch: " << tot_mb << " wait_value: " << wait_value << endl;
                    end_training = 1;
                    break;
                }

            }

        }

        cout << endl;
        if( end_training == 1 ) break;
    }
    /*
    for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i++ )
        delete[] subgroups[ ptgs_i ].subgroups;

    delete[] subgroups;
    */

    if( 
        op.run_type == adadelta_minibatch || op.run_type == adagrad_minibatch ||
        op.run_type == sgdmom_minibatch
    ){
        delete grad_accu;
        delete param_accu;
    }


    return 0;

}

int PT_GROUP::predict(){

    // several rounds until criterion or max round is satisfied
    // if subgroups with only one thread then do only one round and then
    // finish
cost = 0;
    // set n_subgroups ?
    // subdivide into minibatches  PT_GROUP ??
    n_subgroups = ceil( (double) n_phrases / op.predict_batch_size );
    subdivide();

    for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i ++ )
        subgroups[ ptgs_i ].n_subgroups = op.thread_num;

    if( op.unsup_sup ){ rae->mp.unsup = true; rae->mp.sup = false; }

    uvec curr_wc = zeros<uvec>( rae->v_size );
    double nphr = 0;
    double reg_sup, reg_unsup, reg;
    double cost_sup = 0, cost_unsup = 0;

    for( string pass: { "unsupervised", "supervised" } ){

        uint last_mb = 0;
        for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i ++ ){
            subgroups[ ptgs_i ].predict_batch();
            curr_wc += subgroups[ ptgs_i ].wordcounts;
            nphr += subgroups[ ptgs_i ].n_phrases;
        } 

        if( not op.unsup_sup ) break;
        else{

            if( rae->mp.unsup ){
                reg_unsup = rae->regularizer_for_cost();
            }
            else{
                reg_sup = rae->regularizer_for_cost();
            }

            // ##H collect cost here
            collect_costs();
            // divide cost
            divide_cost();

            if( rae->mp.unsup ){
                cost += reg_unsup;
                cost_unsup = cost;
            }
            else{
                cost += reg_sup;
                cost_sup = cost;
            }

            if( rae->mp.unsup ) rae->mp.unsup = false; rae->mp.sup = true;
        }
    }

    // cost and preds comp
    if( not op.unsup_sup ){ collect_costs(); divide_cost(); }
    else{ cost = cost_sup + cost_unsup; }
    collect_preds();

    // add total param norm
    if( not op.unsup_sup ) cost += rae->regularizer_for_cost();

    if( verbose != 0 ) cout << "# fx: " << cost << endl;

    // predict if there are labels
    if( has_labels == 1 ) get_perf_p_l();

    // delete[] subgroups;

    return 0;

}

int PT_GROUP::collect_costs(  ){

    collect_costs( 0, n_subgroups - 1 );

    return 0;

}

int PT_GROUP::collect_costs( uint from, uint to ){

    uint start = 0;

    for( uint gp_i = from; gp_i <= to; gp_i++ ){

        PT_GROUP * ptg = &( this->subgroups[ gp_i ] );

        // collect costs for subgroups of curr subgroup if any
        if( ptg->is_subdivided ) ptg->collect_costs();

        uint end = start + ptg->n_phrases - 1;

        rec_errs( span( start, end ) ) = ptg->rec_errs;
        wnlls( span( start, end ) ) = ptg->wnlls;

        costs( span( start, end ) ) = ptg->costs; // NOTE won't be used
            // if unsup_sup

        start = end + 1;

    }

    return 0;

}

int PT_GROUP::collect_preds(){

    uint start = 0;
    for( uint gp_i = 0; gp_i < n_subgroups; gp_i++ ){

        PT_GROUP * ptg = &( this->subgroups[ gp_i ] );

        // collect preds for subgroups of curr subgroup if any
        if( ptg->is_subdivided ) ptg->collect_preds();

        uint end = start + ptg->n_phrases - 1;

        representations.rows( start, end ) = ptg->representations;
        probs.rows( start, end ) = ptg->probs;

        start = end + 1;

    }

    return 0;

}

int PT_GROUP::clip_grad(){

    vector <mat *> params { dE_deWr, dE_deWc, dE_deWh, dE_dL };

    for_each(
        params.begin(), params.end(),
        []( mat * param){
            double p_norm = sum( sum( abs( *param ) ) );
            if( p_norm > clip_constant * ( *param ).n_elem ){
                *param = clip_constant * ( *param ).n_elem * (*param) /
                    p_norm; 
            }
        }
    );

    return 0;

}


/** \brief collect subgroups gradients to group ( ie. recursive addition of grads )
*/
int PT_GROUP::collect_grads( uint from, uint to ){

    fill_derivatives();

    for( uint gp_i = from; gp_i <= to; gp_i++ ){

        PT_GROUP * ptg = &( this->subgroups[ gp_i ] );

        // collect preds for subgroups of curr subgroup if any
        if( ptg->is_subdivided )
            ptg->collect_grads( 0, ptg->n_subgroups -  1 );

        accumulate_grad( ptg );

        ptg->fill_derivatives();

    }

    return 0;

}

/* \brief get a ( prediction ) ppl structure contaning performance, predictions, phrases representation
*/
struct perf_probs_labels_rep PT_GROUP::get_perf_p_l(){

    vec dlabels = conv_to< vec >::from( labels );
    double macro_p = 
        rae->mp.log_square_loss ?
        macro_perf(
            ( -dlabels + 1. ) + // 0 if 1 , 1 if 0 
            ( 2 * dlabels - 1. ) % probs, // -1 if 0, 1 if 1
            0 * labels,
            get_perf_function( op.perf_type )
        ):
        macro_perf(
            probs, labels, get_perf_function( op.perf_type )
        );

    double pred_mean = 
        rae->mp.log_square_loss ?
        mean( probs_for_labels(
            ( -dlabels + 1. ) + // 0 if 1 , 1 if 0 
            ( 2 * dlabels - 1. ) % probs, // -1 if 0, 1 if 1
            0 * labels ) ):
        mean( probs_for_labels( probs, labels ) );

    vec labels_d = zeros<vec>( labels.n_elem ); labels_d = labels_d + labels;

    ppl.perf_type = op.perf_type;
    ppl.perf = macro_p;
    ppl.probs = probs;
    ppl.labels = labels_d;
    ppl.representations = representations;

    if( verbose != 0 )
        cout << "# " <<  op.perf_type << ": " << macro_p << " mean prob: " <<
            pred_mean << " rec_errs: " << mean( rec_errs ) << endl;

    return( ppl );

}

int PT_GROUP::update_lr(){

    op.learning_rate = op.learning_rate / ( 1 + op.lr_decrease );

    return( 0 );

}

/** \brief transforms gradient vector to adagrad update
*/
int PT_GROUP::update_adagrad(){

    rowvec theta_grad = zeros<rowvec>( rae->n_params );
    copy_grad_to_rowvec( &theta_grad );
    *grad_accu = *grad_accu + pow( theta_grad, 2 );

    theta_grad =
        - op.learning_rate / ( op.epsilon + sqrt( *grad_accu ) ) % theta_grad;

    copy_rowvec_to_grad( theta_grad );
    
    return( 0 );

}

/** \brief transforms gradient vector to adadelta update
*/
int PT_GROUP::update_adadelta(){

    rowvec theta_grad = zeros<rowvec>( rae->n_params );
    copy_grad_to_rowvec( &theta_grad );

    *grad_accu = op.rho * *grad_accu + ( 1 - op.rho ) * pow( theta_grad, 2 );

    rowvec RMS_g_t = sqrt( *grad_accu + op.epsilon );
    rowvec RMS_x_tm1 = sqrt( *param_accu + op.epsilon );

    theta_grad = - ( RMS_x_tm1 / RMS_g_t ) % theta_grad;

    *param_accu = op.rho * *param_accu + ( 1 - op.rho ) * pow( theta_grad, 2 );

    copy_rowvec_to_grad( theta_grad );
    
    return( 0 );

}

/** \brief transforms gradient vector to sgd momentum update
*/
int PT_GROUP::update_momentum(){

    rowvec theta_grad = zeros<rowvec>( rae->n_params );
    copy_grad_to_rowvec( &theta_grad );

    *grad_accu = op.rho * *grad_accu - op.learning_rate * theta_grad;
        // accumulate grad
    theta_grad = *grad_accu; // transform grad

    copy_rowvec_to_grad( theta_grad );
    
    return( 0 );

}

/** \brief transforms gradient vector to sgd vanilla update
*/
int PT_GROUP::update_vanilla(){

    rowvec theta_grad = zeros<rowvec>( rae->n_params );

    copy_grad_to_rowvec( &theta_grad );

    theta_grad *= - op.learning_rate;

    copy_rowvec_to_grad( theta_grad );

    return( 0 );

}


// \brief predictions for a minibatch eventually divided into sub-minibatches for threading
int PT_GROUP::predict_batch(){

    // subdivide into sub minibatches PT_GROUP ??
    get_cached_pars( 1 );
    subdivide();

    std::thread * t_pool = new std::thread[ op.thread_num ];

    for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i++ ){

        subgroups[ ptgs_i ].get_cached_pars( ptgs_i + 2 );
        t_pool[ ptgs_i ] = thread(
            PT_GROUP::predict_batch_parts, &( subgroups[ ptgs_i ]) );

    }
    for( uint ptgs_i = 0; ptgs_i < n_subgroups; ptgs_i++ )
        t_pool[ ptgs_i ].join();

    if( is_subdivided ){
        collect_costs();
        collect_preds();
    }

    delete[] t_pool;
    return 0;

}

void PT_GROUP::predict_batch_parts( PT_GROUP * ptg ){

    rowvec empty = zeros<rowvec>( ptg->rae->mp.w_length );

    for( int p_i = 0 ; p_i < ptg->n_phrases ; p_i++ ){

        ptg->pts[ p_i ].init_buffers();

        if( ( ptg->phrases[ p_i ].n_elem > 1 ) || ptg->rae->mp.single_words ){

            ptg->pts[ p_i ].pass_fwd( );

            if( ptg->pts[ p_i ].n_words > 1 )
                ptg->rec_errs[ p_i ] = sum( ptg->pts[ p_i ].rec_errs );

            ptg->wnlls[ p_i ] = sum( ptg->pts[ p_i ].wnlls );
            ptg->costs[ p_i ] = ptg->pts[ p_i ].cost ;

            if( ptg->rae->mp.sup ){
                if( ptg->pts[ p_i ].n_words > 1 ){

                    if( ptg->prediction_type == predict_term )
                        ptg->probs.row( p_i ) = ptg->pts[ p_i ].z_c_buff.row(
                            ptg->pts[ p_i ].n_words - 2
                        );
                    else{
                        uword max_row;
                        uword max_col;
                        double max_prob = ptg->pts[ p_i ].z_c_buff.max(
                            max_row,
                            max_col
                        );
                        ptg->probs.row( p_i ) = ptg->pts[ p_i ].z_c_buff.row( max_row );
                    }
                    ptg->representations.row( p_i ) =
                        ptg->pts[ p_i ].representation;

                }
                else{
                    ptg->probs.row( p_i ) = ptg->pts[ p_i ].z_c_buff.row( 0 );

                    ptg->representations.row( p_i ) =
                        ptg->pts[ p_i ].L_phrase.row( 0 );
                }
            }


        }else{

                ptg->probs.row( p_i ) = 1. / ptg->rae->n_out
                    * ones<rowvec>( ptg->rae->n_out ) ;

                ptg->representations.row( p_i ) = empty;

        }

        ptg->pts[ p_i ].empty_buffers();
    }

    empty.reset();

}

/** \brief divide current group into subgroups
*/
int PT_GROUP::subdivide(){

    if( n_subgroups > n_phrases )
        n_subgroups = n_phrases;

    if( is_subdivided && n_subgroups_at_sd != n_subgroups ){
        cout << "not the same number of subgroups than saved: " <<
            n_subgroups << "!=" << n_subgroups_at_sd
            << std::flush;
        exit(0);

    }

    if( is_subdivided == 0 ){

        n_subgroups_at_sd = n_subgroups;

        uint target_size_max = ceil( ( double ) n_phrases / n_subgroups );
        uint target_size_min = floor( ( double ) n_phrases / n_subgroups );
        uint subgroup_n_elem = target_size_max;

        int already_placed = 0;

        for( int gt_i = 0; gt_i < n_subgroups ; gt_i++ ){

            uint groups_remaining = n_subgroups - gt_i;
            if(
                already_placed + target_size_min * groups_remaining == n_phrases
            )
                subgroup_n_elem = target_size_min;

            uvec selection = indices(
                span( already_placed, already_placed + subgroup_n_elem - 1 )
            );

            /*subgroups[ gt_i ].populate(
                phrases, labels, selection, rae, fixed_tree, 1, op, t_pool,
                nullptr, save_models_dir );*/
            subgroups.push_back( PT_GROUP(
                phrases, labels, selection, rae, fixed_tree, 1, op, t_pool,
                nullptr, save_models_dir ) );

            subgroups[ gt_i ].grad_accu = grad_accu;
            subgroups[ gt_i ].param_accu = param_accu;
            subgroups[ gt_i ].params_buff = params_buff;
            subgroups[ gt_i ].level = level + 1;
            already_placed += subgroup_n_elem;

        }

        is_subdivided = 1;
    }
    return 0;

}

/** \brief learn function used only by top PT_GROUP
 */
int PT_GROUP::learn(){

    // TODO TRicky Trap...
    this->op.predict_batch_size = this->op.gradeval_batch_size;

    if( op.run_type == lbfgs_batch ) learn_lbfgs();
    else if( op.run_type == dliblbfgs_batch ) learn_dliblbfgs();
    else learn_not_lbfgs();

    return 0;

}

