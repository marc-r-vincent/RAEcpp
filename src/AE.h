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

#ifndef __AE_H_INCLUDED__
#define __AE_H_INCLUDED__

/** \brief Auto Encoder class.

*/
class AE
{
/**
 * Auto Encoder class as described in Socher et al. 2011., defines a neural network module.
 * The module contains three types of layers:
 * \* hidden layer ( h )
 * \* reconstruction layer ( r )
 * \* classification layer ( c )
 *
 * Specifies the mathematical functions necessary to compute the forward pass and the
 * backward pass in the module. Stores the parameters associated with each layers.
 *
 * with the following notation convention:\n
 * \* i: input of the layer
 * \* a: linear transformation of the layer
 * \* z: output of the AE\n
 * \f$ z_h = \frac{ tanh( a_h ) }{ ||tanh( a_h )|| } \f$
 */

public:

    AE(
    ); // constructor

    AE(
        uint v_size,
        model_params mp,
        int n_out
    ); // constructor

    AE copy( uint L_too ); // copy constructor

    uint v_size = 50000;
    model_params mp;
    
    int n_out = 3;
    int n_params;
    
    mat L;
    rowvec one;
    mat e_W_h;
    mat e_W_c;
    mat e_W_r;
    vector <mat *> params;
    vector <std::string> params_names;
    vector <double *> regul_coeffs;

    mat z_h_from_i_h( rowvec i_h, mat e_W_h );
    mat z_r_from_i_r( rowvec i_r, mat e_W_r );
    rowvec softmax_from_a_c( rowvec a_c );
    rowvec deltas_c_from_z_c( rowvec z_c, int y );

    mat d_z_hl_d_a_hl( rowvec z_h );
    mat d_z_div_normz_d_z( rowvec z );
    mat z_c_from_i_c( rowvec i_c, mat e_W_c );
    rowvec meaning_from_z_h( rowvec z_h );
    double recc_err_from_i_h_z_r(
        rowvec i_h, rowvec z_r, double w_a, double w_b
    );
    double loss_c( rowvec z_c, int y, bool is_term );
    rowvec sigmoid_from_a_c( rowvec a_c );
    // future
    mat z_m_from_i_m( rowvec i_m, mat e_W_m );
    double morph_err_from_m_z_m( rowvec m, rowvec z_m, double w_a,
        double w_b );
    int print_params_norms( int );
    int init_parameters();
    int load_model( std::string save_models, uint ind );
    int save_model( std::string save_models, uint ind );
    int remove_model( std::string save_models_dir, uint ind );
    int update_lr();
    int copy_from_vec( vec theta );
    int copy_to_vec( vec * theta );
    int copy_from_dlibvec( column_vector theta );
    int copy_to_dlibvec( column_vector * theta );
#ifdef USE_LIBLBFGS
    int copy_from_1Darray( const lbfgsfloatval_t * theta );
    int copy_to_1Darray( lbfgsfloatval_t * theta );
    int print_1Darray_norm( const lbfgsfloatval_t * theta );
#endif
    int put_back_in_model(
        mat * up_L,
        mat * up_eWh,   
        mat * up_eWr,    
        mat * up_eWc,
        uvec input_array,
        uint rescale_par
    );
    int regularize_grad(
        mat * dE_dL,
        mat * dE_deWc,
        mat * dE_deWh,   
        mat * dE_deWr,
        uvec input_array  
    );
    double regularizer_for_cost();

    contrib contrib_unsupervised(
        rowvec e_z_h, rowvec hl_input, rowvec rl_output, mat d_z_hl_d_a_hl,
        double w_1, double w_2, double div_r_contrib 
    );

};

#endif
