// Implementation of the Recursive AutoEncoder Model described by Socher et al. EMNLP 2011
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

int PARSE_TREE::set(
    AE * set_rae,
    uvec set_input_array,
    int set_y,
    uint fixed_tree
)
{

    rae = set_rae;
    input_array = set_input_array;
    y = set_y;

    n_words = input_array.n_elem;

    weights = ones<rowvec>( 2 * input_array.n_elem - 1 );
    
    tree_list = zeros<uvec>( 2 * input_array.n_elem - 1 );
        // tree_list contains the order of L_phrase elements
        // in the parse_tree
    
    wordcounts = zeros<uvec>( input_array.n_elem );

    for( int w_i = 0; w_i < input_array.n_elem; w_i++ )
        wordcounts[ w_i ] = wordcounts[ w_i ] + 1;

    this->fixed_tree = fixed_tree;

    reset_tree();
    
    return 0;

}

int PARSE_TREE::reset_tree(){

    for( int i = 0 ; i < tree_list.n_elem ; i++ )
        tree_list[ i ] = i;

    return( 0 );

}

int PARSE_TREE::init_buffers(){
    L_phrase = join_cols(
        rae->L.rows( input_array ),
        zeros<mat>( input_array.n_elem - 1, rae->L.n_cols )
    );

    uint classif_num = rae->mp.single_words ?
        2 * n_words - 1:
        n_words - 1;

    rec_errs = zeros<rowvec>( n_words - 1 );
    wnlls = zeros<rowvec>( classif_num );

    dE_dL_phrase = zeros<mat>( n_words, rae->mp.w_length );
    buff_deltas = zeros<mat>( input_array.n_elem - 1, rae->mp.w_length );
        // deltas for parents

    buff_deltas_hi_r = zeros<mat>(input_array.n_elem - 1, rae->mp.w_length);
        // deltas for parents
    buff_deltas_hi_c = zeros<mat>(input_array.n_elem - 1, rae->mp.w_length);
        // deltas for parents
    buff_deltas_r = zeros<mat>( 2 * input_array.n_elem - 1, rae->mp.w_length);
        // reconstruction deltas for children of parents

    // buffers for each hidden layer, each reconstruction, each classification
    z_r_buff = zeros<mat>(input_array.n_elem - 1, 2 * rae->mp.w_length);

    z_c_buff = zeros<mat>( classif_num, rae->n_out );

    dE_deWh = zeros<mat>( 2  * rae->mp.w_length + 1, rae->mp.w_length );
    dE_deWr = zeros<mat>( rae->mp.w_length + 1, 2 * rae->mp.w_length );
    dE_deWc = zeros<mat>( rae->mp.w_length + 1, rae->n_out );

    return( 0 );

}

int PARSE_TREE::empty_buffers(){

    vector <mat *> buffs {
        &L_phrase, &rec_errs, &wnlls, &dE_dL_phrase, &buff_deltas,
        &buff_deltas_hi_r, &buff_deltas_hi_c,
        &buff_deltas_r, &z_r_buff, &z_c_buff, &dE_deWc, &dE_deWr, &dE_deWh 
    };

    for_each(
        buffs.begin(), buffs.end(),
        []( mat * buff ){
            buff->reset();
        }
    );

    return( 0 );

}

PARSE_TREE::PARSE_TREE(){}

PARSE_TREE::PARSE_TREE(
    AE * set_rae,
    uvec set_input_array,
    int set_y,
    uint fixed_tree
)
{

    this->set(
        set_rae,
        set_input_array,
        set_y,
        fixed_tree
    );

}

PARSE_TREE::~PARSE_TREE(){}

void PARSE_TREE::fill_pair_wg_recerr(
    int buff_index, int left, int right, int out_i, double * rec_errs
)
{

    rowvec meaning_left, meaning_right;
    if( left < n_words )
        meaning_left = L_phrase.row( left );
    else
        meaning_left = rae->meaning_from_z_h( L_phrase.row( left ) );

    if( right < n_words )
        meaning_right = L_phrase.row( right );
    else
        meaning_right = rae->meaning_from_z_h( L_phrase.row( right ) );

    rowvec i_h = join_rows( meaning_left, meaning_right );

    /* cout << "l: " << left << " | r: " << right  << " | p: " << */
    /*     n_words + buff_index + out_i  << endl; // COM */

    double w_1 = weights[ left ];
    double w_2 = weights[ right ];

    double w_a = w_1 / (w_1 + w_2);
    double w_b = w_2 / (w_1 + w_2);

    L_phrase.row( n_words + buff_index + out_i ) =
        rae->z_h_from_i_h( join_rows( i_h, rae->one ), rae->e_W_h );

    /* cout << "L " << n_words + buff_index + out_i << ":" << */ 
    /*       L_phrase.row( n_words + buff_index + out_i ) << endl; // COM */

    rowvec meaning = rae->meaning_from_z_h(
        L_phrase.row( n_words + buff_index + out_i ) 
    );

    /* cout << "meaning: " << meaning; // COM */

    z_r_buff.row( buff_index + out_i ) = rae->z_r_from_i_r(
            join_rows( meaning, rae->one ),
            rae->e_W_r
        );

    double rec_err = rae->recc_err_from_i_h_z_r(
        i_h, z_r_buff.row( out_i + buff_index ), w_a, w_b
    );

    // check buff and L phrase
    /* cout <<  "comp L " << out_i + buff_index  + n_words << ": " << */
    /*     L_phrase.row( n_words + buff_index + out_i ); // COM */
    /* cout <<  "comp z_r " << out_i + buff_index << ": " << */
    /*     z_r_buff.row( out_i + buff_index ) << endl << flush; // COM */

    if( rec_err >= datum::inf ){
        cout << endl << 
            "optimization went south (A), infinite rec_err. Aborting" << endl;

        cout << endl;
        print_update_norms( 2 );
        print_L_phrase_norm( 2 );
        rae->print_params_norms( 2 );

        exit(1);
    }

    /* cout << "rec_err: " << rec_err << endl << flush; */
    // cout << &(rae->e_W_c) << endl << flush; // COM

    *( rec_errs + out_i ) = rec_err;

}

struct Idx_Res PARSE_TREE::get_best_rec(
    uvec left_candidates, uvec right_candidates,
    int buff_index, double * rec_errs_candidates, int prev_best
)
{
    // compute hidden representation, then reconstruction, select best tree
    // cache everything ( hidden and recons ) in the process
    double min_rec_err = datum::inf;
    int best_offer = -1;
    int start, end;

    // if we are not at the first buff_index you do not need to compute
    // more than 2 rec_errs_candidates ( all other candidates have already
    // been computed )
    if( buff_index == 0 || fixed_tree == 1 ){
        start = 0;
        end = left_candidates.n_elem - 1;
    }
    else{
        // recompute only the previously created pair candidate
        // which is found at prev_best - 1 pos 
        start = max( 0, prev_best - 1 );
        end = std::min( prev_best, (int) left_candidates.n_elem - 1 );
    }

    // go through all candidates
    for( int i = start ; i <= end ; i++ ){

        int left = left_candidates[ i ];
        int right = right_candidates[ i ];
        fill_pair_wg_recerr( buff_index, left, right, i, rec_errs_candidates );
        if( basic_tree == 1 || fixed_tree == 1 ) break;

    }

    /* cout << "left cand: " << left_candidates.t(); // COM */
    /* cout << "right cand: " << right_candidates.t(); // COM */
    
    if( basic_tree == 0 && fixed_tree == 0 ){

        for( int i = 0 ; i < left_candidates.n_elem ; i++ )
            if( min_rec_err > rec_errs_candidates[ i ] ){
                min_rec_err = rec_errs_candidates[ i ];
                best_offer = i;
            }

    }else{

        best_offer = 0;
        min_rec_err = rec_errs_candidates[ 0 ];

    }

    if( min_rec_err >= datum::inf ){
        cout << endl << 
            "optimization went south (B), infinite rec_err. Aborting" << endl;
        cout << "rec_errs:" << rec_errs << endl;
        cout << endl;
        print_update_norms( 2 );
        print_L_phrase_norm( 2 );
        rae->print_params_norms( 2 );
        exit(1);
    }

    struct Idx_Res idx_rec;
    idx_rec.idx = best_offer;
    idx_rec.res = min_rec_err;

    // sets hidden buff at the right index to be the one giving the minimal
    // reconstruction error
    if( best_offer != 0 ){

        L_phrase.row( n_words + buff_index ) =
            L_phrase.row( n_words + buff_index + best_offer );

        z_r_buff.row( buff_index ) =
            z_r_buff.row( buff_index + best_offer );

    }

    // move rec_errs_candidates left of the best rec_err one position to the right
    for( int prev_i = idx_rec.idx - 2 ; prev_i >= 0; prev_i-- )
        rec_errs_candidates[ prev_i + 1 ] = rec_errs_candidates[ prev_i ]; 

    return idx_rec;
}

int PARSE_TREE::get_best_tree( )
{


    int tmp_idx_a;
    int tmp_idx_b;

    int prev_best = -1;

    uvec tree_list_buff = zeros<uvec>( tree_list.n_elem );
    
    if( fixed_tree == 0 )
        reset_tree();

    /* cout << "tree_list start:" << endl << tree_list.t() << endl << flush; // COM */

    double * rec_errs_candidates = (double*) malloc(
        sizeof(double) * input_array.n_elem - 1
    );

    for( int bigram = 0 ; bigram < input_array.n_elem - 1 ; bigram++ ){


        if( fixed_tree == 1 ){
            // do not recompute tree and weights (use previous ones)
            // compute rec_err

            uvec left_candidates = zeros<uvec>( 1 );
            uvec right_candidates = zeros<uvec>( 1 );

            left_candidates[0] = tree_list[ bigram * 2 ];
            right_candidates[0] = tree_list[ bigram * 2 + 1 ];

            struct Idx_Res idx_rec = get_best_rec(
                left_candidates,
                right_candidates,
                bigram,
                rec_errs_candidates + bigram,
                prev_best
            );

            rec_errs[ bigram ] = idx_rec.res;

        }
        else{
            
            /* cout << endl << "Selecting bigram: " << bigram << endl; // COM */

            // bigram is the number of the pair being set
            /* cout << "tree_list start:" << endl << tree_list.t() << endl; // COM */
            /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

            int remaining = input_array.n_elem - 1 - bigram;
                // number of remaining parents to set
                // ie. number of remaining pairs to test / 2

            uvec left_candidates = zeros<uvec>( remaining );
            uvec right_candidates = zeros<uvec>( remaining );

            for( int i = 0; i < remaining ; i++ ){

                left_candidates[i] = tree_list[ i + bigram * 2 ];
                right_candidates[i] = tree_list[ i + bigram * 2 + 1 ];

            }

            /* cout << "left_candidates: " << left_candidates.t() << endl; // COM */
            /* cout << "right_candidates: " << right_candidates.t() << endl; // COM */

            struct Idx_Res idx_rec = get_best_rec(
                left_candidates,
                right_candidates,
                bigram,
                rec_errs_candidates + bigram,
                prev_best
            );

            rec_errs[ bigram ] = idx_rec.res;
            prev_best = idx_rec.idx;

            /* cout << "sel idx: " << idx_rec.idx << endl; // COM */
            
            int filled = 0; // NUMBER of already filled positions in tree_list_buff

            if( bigram > 0 ){
                span set_elements = span( 0, 2 * bigram - 1 );
                    // put selected pair after set_elements
                tree_list_buff( set_elements ) = tree_list( set_elements );
                filled = set_elements.b - set_elements.a + 1;
            }

            /* cout << "filled for fixed:" << filled << endl;  // COM */
            /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

            int selected_pos = 2 * bigram + idx_rec.idx;

            weights[ n_words + bigram  ] =
                weights[ tree_list[ selected_pos ] ] +
                weights[ tree_list[ selected_pos + 1 ] ]; // weight of parent
                // is the sum of the weights of its children

            // put selected after fixed
            /* cout << selected_pos << " - " << selected_pos + 1 << endl;// COM */
            span selected_span = span(
                selected_pos,
                selected_pos + 1
            );

            tree_list_buff( span( filled, filled + 1) ) = tree_list( selected_span );
            filled = filled + 2;

            /* cout << selected_span.a  << " - " << selected_span.b << endl;  // COM */
            /* cout << "filled for selected:" << filled << endl;  // COM */
            /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

            // if there was something before selected 
            // add it now
            if( idx_rec.idx > 0 ){
                span pre_selected = span(
                    2 * bigram,
                    2 * bigram + idx_rec.idx - 1
                ) ;
                tree_list_buff(
                    span(
                        filled,
                        filled + pre_selected.b - pre_selected.a
                    )
                ) = tree_list( pre_selected );
                filled = filled + pre_selected.b - pre_selected.a + 1;

                /* cout << "pre:" << endl << tree_list( pre_selected ) << endl; // COM */
                /* cout << pre_selected.a << " - " << pre_selected.b << endl; // COM */
            }

            /* cout << "filled for pre:" << filled << endl; // COM */
            /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

            // place the replacing parent here
            // the first parent remaining is at n_words + bigram
            tree_list_buff[ filled ] = tree_list[ n_words + bigram ];
            filled  = filled + 1;
     
            /* cout << n_words + bigram << " - " << n_words + bigram << endl; // COM */
            /* cout << "filled for parent:" << filled << endl; // COM */
            /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

            if( bigram < input_array.n_elem - 2 ){

                // if the last bigram do not continue

                if( idx_rec.idx < remaining - 1  ){ 
                    span post_selected = span(
                        selected_pos + 2,
                        n_words + bigram - 1
                    ) ;

                    tree_list_buff(
                         span(
                            filled,
                            filled + post_selected.b - post_selected.a
                        )                   
                    ) = tree_list( post_selected );
                    filled = filled + post_selected.b - post_selected.a + 1;

                    /* cout << post_selected.a << " - " << post_selected.b << endl; // COM */

                }

                /* cout << "filled for post:" << filled << endl; // COM */
                /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

                if( filled < tree_list.n_elem ){

                    span last_elems = span( filled, tree_list.n_elem - 1  );
                    tree_list_buff( last_elems ) = tree_list( last_elems );

                }

                tree_list = tree_list_buff;

                /* cout << "filled for post_parents:" << filled << endl; // COM */
                /* cout << "tree_list buff:" << endl << tree_list_buff.t() << endl; // COM */

            }

        }

    }
    
    /* cout << "tree_list end:" << endl << tree_list.t() << endl << flush; // COM */

    free( rec_errs_candidates );

    return 0;
}

int PARSE_TREE::gradient_check(){

    double epsilon = 0.0000001;

    AE * rae_or = rae;
    
    AE rae_cop = rae->copy( 0 );

    rae = & rae_cop;

    rae->v_size = L_phrase.n_rows;

    rowvec rec_errs_or = rec_errs;
    rowvec wnlls_or = wnlls;
    double cost_or = cost;

    cout << "# phrase fx gc: " << cost << endl;

    vector <mat *> params {
        &( rae->e_W_c ), &( rae->e_W_h ), &( rae->e_W_r ), & L_phrase
    };
    vector <std::string> params_names {
        "e_W_c", "e_W_h", "e_W_r", "L_phrase"
    };

    vector <mat *> bp_gradients = { &dE_deWc, &dE_deWh, &dE_deWr, &dE_dL_phrase };
    vector <mat> gradients( 4 );
    
    // compute gradients numerically for all W and L
    for( uint pi = 0; pi < 4; pi++ ){

        // size of dE_dL is not size of L
        uint n_rows;
        if( pi != 3 )
            n_rows = params[pi]->n_rows;
        else
            n_rows = n_words;

        gradients[ pi ] = zeros( n_rows, params[pi]->n_cols );

        for( uint ri = 0; ri < n_rows; ri++ ){

            for( uint ci = 0; ci < gradients[pi].n_cols; ci++ ){
                
                double old_val = ( * params[ pi ] )( ri, ci );
                // add epsilon
                ( * params[ pi ] )( ri, ci ) += epsilon;
                /* cout << endl << "PASS A" << endl; // COM */
                uint old_fixed_tree = this->fixed_tree;
                // first pass fwd
                pass_fwd();
                double cost_diff = cost;
                // substract epsilon
                ( * params[ pi ] )( ri, ci ) = old_val - epsilon;
                // second pass fwd
                pass_fwd();
                this->fixed_tree = old_fixed_tree;
                cost_diff -= cost;
                // divide by epsilon diff
                cost_diff = cost_diff / ( 2 * epsilon );
                gradients[ pi ]( ri, ci ) = cost_diff;
                // reset
                ( * params[ pi ] )( ri, ci ) = old_val;

            }

        }

        cout << endl; // COM
    };
    
    // reset tree
    reset_tree();

    // compute differences
    double diff_norm;
    double tot_norm;

    cout << "phrase length: " << n_words << endl;
    for( uint pi = 0; pi < 4; pi++ ){

/*
if( 
    params_names[ pi ] == "e_W_r" ||
    params_names[ pi ] == "L_phrase"
){
cout << params_names[ pi ] << endl;
cout << gradients[ pi ] << endl;
cout << *bp_gradients[ pi ] << endl;
}
*/

        tot_norm = norm( *params[ pi ], 2 );

        // cout << "ck grad: " << gradients[ pi ]; // COM
        // cout << "bp grad: " << ( *bp_gradients[ pi ] ) << endl; // COM

        diff_norm = norm( gradients[ pi ] - *bp_gradients[ pi ], 2 );

        cout << "gradient check:" << endl << 
            "diff_norm: " << params_names[ pi ] << " " << diff_norm <<
            ", should be << w.r.t. tot_norm: " << tot_norm << endl << flush;

        if( replace_grad == 1 )
            *(  bp_gradients[ pi ] ) = gradients[ pi ];

    }

    double t_norm_bp = 0;
    double t_norm_gd = 0;
    for( uint pi = 0; pi < 4; pi++ ){
        uint n_rows;
        if( pi != 3 )
            n_rows = params[pi]->n_rows;
        else
            n_rows = n_words;
        for( uint ri = 0; ri < n_rows; ri++ )
            for( uint ci = 0; ci < gradients[pi].n_cols; ci++ ){
                t_norm_bp += abs( ( * bp_gradients[pi] )( ri, ci) );
                t_norm_gd += abs( ( gradients[pi] )( ri, ci) );
            }
    }
    cout << "# ph bp norm: " << t_norm_bp << endl;
    cout << "# ph gp norm: " << t_norm_gd << endl;

    cout << endl << flush;

    rae = rae_or;
    
    return( 0 );

}

int PARSE_TREE::pass_fwd(){

    get_best_tree(); // TODO innefficient but mandatory right now

    double div_r_contrib = rae->mp.divide_rec_contrib ?
        input_array.n_elem - 1:
        1
        ;
    double alpha_unsup =
        ( not rae->mp.unsup ) and rae->mp.sup ? 1 - rae->mp.alpha : rae->mp.alpha;
    double alpha_sup =
        ( not rae->mp.sup ) and rae->mp.unsup ? rae->mp.alpha: 1 - rae->mp.alpha;

    cost = 0;
    // now compute classification for each hidden layer ( parent )
    uint parent_c_buff_i; // parent index in classification buffers
    for( int parent = 0 ; parent < n_words - 1 ; parent = parent + 1 ){

        if( rae->mp.sup ){

            parent_c_buff_i = rae->mp.single_words ?
                n_words + parent:
                parent;

            representation = rae->meaning_from_z_h(
                L_phrase.row( n_words + parent ) 
            );

            z_c_buff.row( parent_c_buff_i ) =
                rae->z_c_from_i_c( 
                    join_rows( representation, rae->one ),
                    rae->e_W_c
                );

            /* cout << "z_c_buff: " << parent << " " <<  z_c_buff.row( parent ) << endl; // COM */

            wnlls[ parent_c_buff_i ] = rae->loss_c(
                ( ( rowvec ) z_c_buff.row( parent_c_buff_i ) ), y, true );

        }

        // compute cost function which is the sum over each bigram
        // of alpha * E cl + ( 1 - alpha  ) * E re
        // not the same weight for terminals and non terminals

        if( rae->mp.unsup ){
            rec_errs[ parent ] = alpha_unsup * rec_errs[ parent ];
            cost += rec_errs[ parent ] / div_r_contrib;
        }
        if( rae->mp.sup ){
            wnlls[ parent_c_buff_i ] = alpha_sup * wnlls[ parent_c_buff_i ];
            cost += wnlls[ parent_c_buff_i ];
        }
            // NOTE this one won't be used if unsup_sup

    }

    // optionnaly do the same for words
    if( rae->mp.single_words && rae->mp.sup ){

        for( int word = 0 ; word < n_words ; word = word + 1 ){

            z_c_buff.row( word ) =
                rae->z_c_from_i_c( 
                    join_rows( L_phrase.row( word ), rae->one ),
                    rae->e_W_c
                );

            wnlls[ word ] = alpha_sup * rae->loss_c(
                z_c_buff.row( word ), y, false );

            cost += wnlls[ word ];
            // NOTE this one won't be used if unsup_sup

        }

    }

    /* cout << "cost: " << cost << endl; // COM */

    return 0;

}

int PARSE_TREE::pass_bwd(){

    uvec w_idcs_hidden = zeros<uvec>( rae->mp.w_length );

    uvec left_words = zeros<uvec>( ( tree_list.n_elem - 1 ) / 2 );
    uvec right_words = zeros<uvec>( ( tree_list.n_elem - 1 )  / 2 );
    uvec parents = zeros<uvec>( ( tree_list.n_elem - 1 )  / 2 );

    for( int i = 0 ; i < left_words.n_elem ; i++ ){
        left_words[ i ] = tree_list[ i * 2 ];
        right_words[ i ] = tree_list[ i * 2 + 1 ];
        parents[ i ] = n_words + i;
    }

    /* cout << left_words.t(); // COM */
    /* cout << right_words.t(); // COM */
    /* cout << parents.t() << endl; // COM */

    int left;
    int right;
    int parent;

    /* cout << "buff_deltas start:" << endl << buff_deltas; // COM */
    double div_r_contrib = rae->mp.divide_rec_contrib ?
        input_array.n_elem - 1:
        1
        ;
    double alpha_unsup =
        ( not rae->mp.unsup ) and rae->mp.sup ? 1  - rae->mp.alpha : rae->mp.alpha;
    double alpha_sup =
        ( not rae->mp.sup ) and rae->mp.unsup ? rae->mp.alpha: 1 - rae->mp.alpha;

    for(
        int triplet_i = input_array.n_elem - 1 - 1;
        triplet_i > -1 ; 
        triplet_i--
    ){

        left = tree_list[ 2 * triplet_i ];
        right = tree_list[ 2 * triplet_i + 1 ];
        parent = n_words + triplet_i;

        // check if parent has a grand parent ( ie. parent is not last )
        // and guess which side and number 
        // it is (search parent in left nodes, right nodes)
        uvec p_idx_as_lc = find( left_words == parent );
        uvec p_idx_as_rc = find( right_words == parent );

        bool parent_is_left = true;
        bool children_have_gp = true;

        if( p_idx_as_rc.n_elem > 0 ) parent_is_left = false;
        else if( p_idx_as_lc.n_elem == 0 ) children_have_gp = false;

        double w_1 = weights[ left ];
        double w_2 = weights[ right ];

        rowvec one = ones<rowvec>(1);

        rowvec meaning_left, meaning_right;
        if( left < n_words ) meaning_left = L_phrase.row( left );
        else meaning_left = rae->meaning_from_z_h( L_phrase.row( left ) );
        if( right < n_words ) meaning_right = L_phrase.row( right );
        else meaning_right = rae->meaning_from_z_h( L_phrase.row( right ) );

        rowvec hl_input = join_rows( meaning_left, meaning_right );

        mat d_z_hl_d_a_hl = rae->d_z_hl_d_a_hl(
            L_phrase.row( n_words + triplet_i )
        );

        rowvec meaning = rae->meaning_from_z_h(
            L_phrase.row( n_words + triplet_i )
        );

        rowvec e_z_h = join_rows( meaning, one );

        if( rae->mp.sup ){

            rowvec deltas_c = rae->mp.single_words ?
                rae->deltas_c_from_z_c( z_c_buff.row( parent ), y ):
                rae->deltas_c_from_z_c( z_c_buff.row( triplet_i ), y );

            deltas_c *= rae->mp.nterm_w * alpha_sup;

            rowvec deltas_per_W_cl = 
                deltas_c * rae->e_W_c.rows( span( 0, rae->mp.w_length - 1 ) ).t();

            buff_deltas_hi_c.row( triplet_i ) = deltas_per_W_cl * d_z_hl_d_a_hl;

            dE_deWc += e_z_h.t() * deltas_c;

            for( uint child: { left, right } ){

                if( child < n_words ){

                    // if model includes single words then add gradient from direct classification
                    // ( affects W_c and L
                    if( rae->mp.single_words ){

                        rowvec e_child = join_rows(
                            L_phrase.row( child ), rae->one );

                        rowvec deltas_c_child = alpha_sup *
                            rae->deltas_c_from_z_c( z_c_buff.row( child ), y );

                        dE_deWc += e_child.t() * deltas_c_child;

                        dE_dL_phrase.row( child ) += deltas_c_child *
                            rae->e_W_c.rows( 0, rae->mp.w_length - 1 ).t();

                    }

                }

            }

        }

        if( rae->mp.unsup ){
          
            rowvec rl_output = z_r_buff.row( triplet_i );

            contrib contrib_re = rae->contrib_unsupervised( 
                e_z_h, hl_input, rl_output, d_z_hl_d_a_hl, w_1, w_2, div_r_contrib );

            buff_deltas_hi_r.row( triplet_i ) = contrib_re.deltas_hi;
            rowvec deltas_r = contrib_re.deltas;
            dE_deWr += contrib_re.dE_dX;

            // transmit delta_r through children hidden layer non linearity
            for( uint child: { left, right } ){

                uint c_start, c_end;

                if( child == left ){ c_start = 0; c_end = rae->mp.w_length - 1; }
                else{
                    c_start = rae->mp.w_length; c_end = 2 * rae->mp.w_length - 1;
                }

                // store deltas for future use
                buff_deltas_r.row( child ) = deltas_r( span( c_start, c_end ) );

                // if children are words (not meanings) then transmit error to them
                // -deltas_r is eq to dE / dx

                if( child < n_words ){

                    // add direct contribution to reconstruction error:
                    dE_dL_phrase.row( child ) -= 
                        keep_direct * deltas_r( span( c_start, c_end ) );
                    /* cout << "dE_dL_phrase.row( child ): " << dE_dL_phrase.row( child ); // COM */
                }

            }

        }

        rowvec deltas_gp_hi_r, deltas_gp_hi_c;

        auto pair_r = std::make_pair(
            std::ref( deltas_gp_hi_r ), std::ref( buff_deltas_hi_r ) );
        auto pair_c = std::make_pair( 
            std::ref( deltas_gp_hi_c ), std::ref( buff_deltas_hi_c ) );

        rowvec deltas_gp;
        rowvec deltas_gp_per_W_hi;

        if( children_have_gp ){

            bool doing_rec = true;

            for( auto dvec_bmat: { pair_r, pair_c } ){
                if( 
                    doing_rec  && rae->mp.unsup ||
                    ( ! doing_rec ) && rae->mp.sup
                ){

                    if( parent_is_left ){ 
                        
                        deltas_gp = dvec_bmat.second.row( p_idx_as_lc[0] );

                        deltas_gp_per_W_hi = deltas_gp *
                            rae->e_W_h.rows( 0, rae->mp.w_length - 1 ).t();
                            
                    }else{

                        deltas_gp = dvec_bmat.second.row( p_idx_as_rc[0] );

                        deltas_gp_per_W_hi = 
                            deltas_gp *
                            rae->e_W_h.rows
                                ( rae->mp.w_length, 2 * rae->mp.w_length - 1 ).t();

                    }

                    dvec_bmat.first = deltas_gp_per_W_hi * d_z_hl_d_a_hl ;

                    // CONST NORM CLIP OF ANCESTOR'S DELTA
                    if( clip_deltas == 1 ){
                       double a_norm = sum( dvec_bmat.first );
                       if( a_norm > clip_constant * dvec_bmat.first.n_elem ){
                            dvec_bmat.first =
                                clip_constant * dvec_bmat.first.n_elem *
                                clip_deltas_factor * dvec_bmat.first / a_norm; 
                       }
                    }

                }

                doing_rec = false;

            }

        }

        // Take contrib of hl_input to deltas_r, transmit by adding deltas
        // to deltas buff if left or right is a parent

        // limit transmission of deltas ?
        if( children_have_gp ){
            if( rae->mp.unsup ){
                buff_deltas_hi_r.row( triplet_i ) +=
                    transmission * deltas_gp_hi_r;
                buff_deltas_hi_r.row( triplet_i ) +=
                    -buff_deltas_r.row( parent ) * d_z_hl_d_a_hl;
            }
            if( rae->mp.sup )
                buff_deltas_hi_c.row( triplet_i ) +=
                    transmission * deltas_gp_hi_c;
        }

        rowvec e_hl_input = join_rows( hl_input, one );

        dE_deWh += e_hl_input.t() * (
            ( rae->mp.unsup ? 1: 0 ) *
                buff_deltas_hi_r.row( triplet_i ) +
            ( rae->mp.sup ? 1: 0 ) *
                buff_deltas_hi_c.row( triplet_i )
        );

        for( uint child: { left, right } ){

            if( child < n_words ){

                uint c_start, c_end;

                if( child == left ){ c_start = 0; c_end = rae->mp.w_length - 1; }
                else{
                    c_start = rae->mp.w_length; c_end = 2 * rae->mp.w_length - 1;
                }
                dE_dL_phrase.row( child ) += ( 
                    ( rae->mp.unsup ? 1: 0 ) *
                        buff_deltas_hi_r.row( triplet_i ) +
                    ( rae->mp.sup ? 1: 0 ) *
                        buff_deltas_hi_c.row( triplet_i ) 
                    ) * rae->e_W_h.rows ( span( c_start, c_end ) ).t();

                /* cout << "dE_dL_phrase.row( child ): " << dE_dL_phrase.row( child ); // COM */

            }

        }

    }

    return 0;

}

int PARSE_TREE::clip_grad(){

    vector <mat *> params { &dE_deWr, &dE_deWc, &dE_deWh, &dE_dL_phrase };

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

int PARSE_TREE::print_update_norms( int nrm ){

    cout << "dE_deWc: " << norm( dE_deWc, nrm ) << endl ;
    cout << "dE_deWh: " << norm( dE_deWh, nrm ) << endl ;
    cout << "dE_deWr: " << norm( dE_deWr, nrm ) << endl ;
    cout << "dE_dL_phrase: "<< norm( dE_dL_phrase, nrm ) << endl << endl;
    cout << "buff_deltas: " << norm( buff_deltas, nrm ) << endl ;

    return 0;

}

int PARSE_TREE::print_L_phrase_norm( int nrm ){

    cout << "children: " << norm( L_phrase.rows(
            span( 0, n_words - 1 )
        ), nrm ) << endl ;
    cout << "parents: " << norm( L_phrase.rows(
            span( n_words, 2 * n_words - 2 )
        ), nrm ) << endl ;

    return 0;
}
