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

#ifndef __PARSE_TREE_H_INCLUDED__
#define __PARSE_TREE_H_INCLUDED__

/** \brief Class for a parse tree whose structure can be learned.

As defined in Socher 2011. The parse tree is a sequence of bigrams.\n
The class defines functions to compute forward and backward passes at the level
of tbe parse tree.\n
During the forward pass the structure of the parse tree is selected using
a greedy search, minimizing the reconstruction error of the sequence of bigrams.

*/
class PARSE_TREE
{

public:

    PARSE_TREE( AE *, uvec, int, uint ); // constructor
    PARSE_TREE( ); // constructor
    ~PARSE_TREE(); // destructor

    int set( AE * set_rae, uvec set_input_array, int set_y,
         uint
    );
    int pass_fwd( );
    void fill_pair_wg_recerr(
        int buff_index, int left, int right, int out_i, double * rec_errs
    );
    struct Idx_Res get_best_rec(
        uvec left_candidates,
        uvec right_candidates,
        int buff_index,
        double * rec_errs_candidates,
        int prev_best
    );

    int get_best_tree( );
    int pass_bwd();
    int print_update_norms( int nrm );
    int print_L_phrase_norm( int nrm );
    int gradient_check();
    int clip_grad();

    int init_buffers();
    int reset_tree();
    int empty_buffers();

    AE * rae;
    uvec input_array;
    int y;
    int n_words; //!< number of words in the phrase
    int thread_num;
    uint fixed_tree;

    mat L_phrase;
    mat z_r_buff;
    mat z_c_buff;
    rowvec representation;
    rowvec rec_errs;
    rowvec wnlls;
    double cost;

    mat buff_deltas; // deltas from parents (hidden layer above hidden layer)
    mat buff_deltas_r; // deltas from parents
    mat buff_deltas_hi_r; // deltas from parents
    mat buff_deltas_hi_c; // deltas from parents
    mat dE_deWh;
    mat dE_deWr;
    mat dE_deWc;
    mat dE_dL_phrase;

    rowvec weights;

    uvec tree_list;

    uvec wordcounts;

};

#endif
