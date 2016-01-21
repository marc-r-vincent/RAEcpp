// Implementation of the Recursive Auto Encoder model originally described by Socher et al. EMNLP 2011
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

# include <common.h>
# include <util.h>
# include <hdf5.h>
# include <AE.h>

AE::AE()
{

    init_parameters();

    n_params =
        e_W_c.n_elem + e_W_h.n_elem + e_W_r.n_elem +
        L.n_elem;

    params = { &e_W_r, &e_W_c, &e_W_h, &L }; 
    params_names = { "e_W_r", "e_W_c", "e_W_h", "L" };
    regul_coeffs = {
        &(this->mp.lambda_reg_r), &(this->mp.lambda_reg_c), &(this->mp.lambda_reg_h),
        &(this->mp.lambda_reg_L)
    };


}

AE::AE(
    uint v_size,
    model_params mp,
    int n_out
)
{

    this->v_size = v_size;
    this->mp = mp;
    if( n_out == 2 && mp.log_square_loss ) this->n_out = 1;
    else this->n_out = n_out;  

    init_parameters();

    this->n_params =
        e_W_c.n_elem + e_W_h.n_elem + e_W_r.n_elem + L.n_elem;

    params = { &e_W_r, &e_W_c, &e_W_h, &L }; 
    params_names = { "e_W_r", "e_W_c", "e_W_h", "L" };
    regul_coeffs = {
        &(this->mp.lambda_reg_r), &(this->mp.lambda_reg_c), &(this->mp.lambda_reg_h),
        &(this->mp.lambda_reg_L)
    };
}

int AE::init_parameters(){

    double L_scale = 1;
    if( mp.original_L_scale ) L_scale = 1e-3;

    // generate L
    if( v_size != 0 ){
        L.set_size( v_size, mp.w_length );
        L = L_scale * randn<mat>( v_size, mp.w_length );
    }

    if( ( not mp.original_L_scale and not mp.standard_L_scale ) )
        for( int L_r = 0; L_r < L.n_rows; L_r++ ){ 
            L.row( L_r ) = L.row( L_r ) / norm( L.row( L_r ), 2 );
        }

    one = ones(1);
    e_W_h = get_rand_wmat( 2 * mp.w_length + 1, mp.w_length ); 
    e_W_c = get_rand_wmat( mp.w_length + 1, n_out );
    e_W_r = get_rand_wmat( mp.w_length + 1, 2 * mp.w_length );

    return( 0 );

}

contrib AE::contrib_unsupervised(
    rowvec e_z_h, rowvec hl_input, rowvec rl_output, mat d_z_hl_d_a_hl, 
    double w_1, double w_2, double div_r_contrib 
){

    double alpha_unsup =
        ( not mp.unsup ) and mp.sup ? 1 - mp.alpha: mp.alpha;

    double w_a;
    double w_b;
    
    if( mp.weight_rec ){
        w_a = w_1 / (w_1 + w_2); w_b = w_2 / (w_1 + w_2);
    }
    else{ w_a = 1; w_b = 1; }

    contrib contrib_re;

    rowvec diff = ( 1 - mp.keep_direct + mp.keep_direct * hl_input );

    if( mp.norm_rec ){
        diff( span( 0, mp.w_length - 1 ) ) -=
            mp.keep_indirect *     
            meaning_from_z_h( rl_output(
                span( 0, mp.w_length - 1 )
            ) );
        diff( span( mp.w_length, 2 * mp.w_length - 1 ) ) -=
            mp.keep_indirect *     
            meaning_from_z_h( rl_output (
                span( mp.w_length, 2 * mp.w_length - 1 )
            ) );
    }
    else diff -= rl_output;

    contrib_re.deltas = ( 1 / div_r_contrib ) * alpha_unsup * join_rows(
        (rowvec)( - w_a * diff( span( 0, mp.w_length - 1 ) ) ),
        (rowvec)( - w_b * diff( 
            span( mp.w_length, 2 * mp.w_length - 1 ) ) )
    );

    // if normalization of reconstruction do transform deltas_r
    // by multiplying by norm derivative 
    rowvec deltas_r_norm;
    if( mp.norm_rec ){

        rowvec z_r = rl_output;

        mat dnormzr_dzr_left =
            d_z_div_normz_d_z( z_r( span( 0, mp.w_length - 1 ) ) );

        mat dnormzr_dzr_right = 
            d_z_div_normz_d_z(
                z_r( span( mp.w_length, 2 * mp.w_length  - 1 ) ) );

        deltas_r_norm = join_rows(
            mp.keep_indirect *     
                contrib_re.deltas( span( 0, mp.w_length - 1 ) ) *
                dnormzr_dzr_left,
            mp.keep_indirect *     
                contrib_re.deltas( span( mp.w_length, 2 * mp.w_length - 1 ) ) * 
                dnormzr_dzr_right
        );

    }

    if( mp.norm_rec ) contrib_re.dE_dX = e_z_h.t() * deltas_r_norm;
    else contrib_re.dE_dX = e_z_h.t() * contrib_re.deltas;

    rowvec deltas_per_W_re;
    if( mp.norm_rec )
        deltas_per_W_re = deltas_r_norm *
        e_W_r.rows( span( 0, mp.w_length - 1 ) ).t();
    else
        deltas_per_W_re = contrib_re.deltas *
        e_W_r.rows( span( 0, mp.w_length - 1 ) ).t();

    // delta = dE / da
    contrib_re.deltas_hi = deltas_per_W_re * d_z_hl_d_a_hl;

    return( contrib_re );

}


int AE::put_back_in_model(
    mat * up_L,
    mat * up_eWh,   
    mat * up_eWr,    
    mat * up_eWc,
    uvec input_array,
    uint rescale_par
){

    if( const_param_norm != 0 && rescale_par == 1 )
    {
        cout << "ACHTUNG !!!" << endl;

        vector <mat *> params_to_scale { &e_W_c, &e_W_h, &e_W_r };

        for_each(
            params_to_scale.begin(), params_to_scale.end(),
            []( mat * param){
                double p_norm = norm( *param, 2 );
                if( p_norm >= const_param_norm  ){
                    *param = const_param_norm * ( *param ) / p_norm; 
                }
            }
        );

    }

    L.rows( input_array ) += ( *up_L ) ;
    e_W_r += ( *up_eWr );
    e_W_c += ( * up_eWc );
    e_W_h += ( * up_eWh );

    return 0;
}

/** \brief add the contribution of parameters regularization to gradient
 *
 *
 */
int AE::regularize_grad(
    mat * dE_dL,
    mat * dE_deWc,
    mat * dE_deWh,   
    mat * dE_deWr,
    uvec input_array
){

    double curr_lambda_L = mp.unsup ? mp.lambda_reg_L : mp.lambda_reg_L_sup;

    double alpha_scale = 1;
    if( not ( mp.unsup and mp.sup ) ) // if unsup and sup are done sequentially
        alpha_scale = mp.unsup ? mp.alpha : ( 1 - mp.alpha );

    if( input_array.n_elem != 0 && curr_lambda_L != 0. )
        (*dE_dL).rows( input_array ) = (*dE_dL).rows( input_array ) +
            alpha_scale * curr_lambda_L * L.rows( input_array );

    if( input_array.n_elem == 0 && curr_lambda_L != 0. )
        *dE_dL = *dE_dL + alpha_scale * curr_lambda_L * L;

    if( mp.lambda_reg_c != 0. and not mp.sup  )
        *dE_deWc = *dE_deWc + alpha_scale * mp.lambda_reg_c * e_W_c;

    if( mp.lambda_reg_h != 0. )
        *dE_deWh = *dE_deWh + alpha_scale * mp.lambda_reg_h * e_W_h;

    if( mp.lambda_reg_r != 0. )
        *dE_deWr = *dE_deWr + alpha_scale * mp.lambda_reg_r * e_W_r;

    return( 0 );
}

/** \brief add the contribution of parameters regularization to cost
 *
 *
 */
double AE::regularizer_for_cost(){

    regul_coeffs[ 3 ] = mp.unsup ? &( mp.lambda_reg_L ) : &(  mp.lambda_reg_L_sup );

    double alpha_scale = 1;
    if( not ( mp.unsup and mp.sup ) ) // in unsup and sup are done sequentially
        alpha_scale = mp.unsup ? mp.alpha : ( 1 - mp.alpha );

    // add total param norm
    vec x = zeros<vec>( n_params );
    copy_to_vec( &x );

    uint start = 0;
    mat * param;
    double reg = 0;
    double reg_contrib = 0;

    for( uint pi = 0; pi < params.size(); ++pi ){

        param = params[ pi ];

        if( ( not mp.sup ) or pi != 1 ){

            reg_contrib = 0.5 * alpha_scale * *( regul_coeffs[ pi ] ) *
                dot(
                    x( span( start, start + param->n_elem - 1 ) ),
                    x( span( start, start + param->n_elem - 1 ) )
                )
                ;

            if( verbose >= 2 ){
                cout << "# " << params_names[ pi ] << 
                    " reg contribution: " << reg_contrib << endl;
            }

            reg += reg_contrib;

        }

        start += param->n_elem;
    }
    if( verbose >= 2 ) cout << endl;

    return( reg );

}

int AE::copy_from_vec( const vec theta ){

    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*par)( r_i, c_i ) = theta( theta_idx );
                theta_idx++;
            }

    }

    return( 0 );

}

int AE::copy_to_vec( vec * theta ){

    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*theta)[ theta_idx ] = (*par)( r_i, c_i );
                theta_idx++;
            }

    }

    return( 0 );

}

int AE::copy_from_dlibvec( const column_vector theta ){

    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*par)( r_i, c_i ) = theta( theta_idx );
                theta_idx++;
            }

    }

    return( 0 );

}

int AE::copy_to_dlibvec( column_vector * theta ){

    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*theta)( theta_idx ) = (*par)( r_i, c_i );
                theta_idx++;
            }

    }

    return( 0 );

}

#ifdef USE_LIBLBFGS
int AE::copy_from_1Darray( const lbfgsfloatval_t * theta ){

    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                (*par)( r_i, c_i ) = theta[ theta_idx ];
                theta_idx++;
            }

    }

    return( 0 );

}


int AE::copy_to_1Darray( lbfgsfloatval_t * theta ){

    uint theta_idx = 0;
    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;

        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                theta[ theta_idx ] = (*par)( r_i, c_i );
                theta_idx++;
            }

    }

    return( 0 );

}

int AE::print_1Darray_norm( const lbfgsfloatval_t * theta ){

    double norm1D = 0;
    uint theta_idx = 0;

    for( auto it = params.begin(); it != params.end() ; ++it ){

        mat * par = *it;
        for( uint c_i = 0; c_i < par->n_cols; c_i++ )
            for( uint r_i = 0; r_i < par->n_rows; r_i++ ){
                norm1D += pow( theta[ theta_idx ], 2 );
                theta_idx++;
            }

    }

    norm1D = sqrt( norm1D );

    cout << "theta: " << norm1D << endl;

    return( 0 );

}
#endif

AE AE::copy( uint L_too )
{

    double cop_v_size = 0;
    if( L_too != 0 ){ cop_v_size = v_size; }

    AE rae = AE(
        cop_v_size,
        mp,
        n_out
    );

    // generate L
    if( L_too == 1 ){ rae.L = L; }

    rae.e_W_h = e_W_h;
    rae.e_W_c = e_W_c;
    rae.e_W_r = e_W_r;

    return( *this );
}

mat AE::z_m_from_i_m( rowvec i_m, mat e_W_m ){

    return( z_h_from_i_h( i_m, e_W_m) );

}

double AE::morph_err_from_m_z_m(
    rowvec m, rowvec z_m, double w_a, double w_b    
)
{

    return(
        pow( norm(
            m - 
            z_m( span( 0, mp.w_length - 1 ) )
            , 2 ), 2
        ) );
}

mat AE::z_h_from_i_h( rowvec i_h, mat e_W_h )
{

    mat z_h;

    rowvec a_h = i_h * e_W_h;

    if(
        mp.z_type == z_is_tanh ||
        mp.z_type == z_is_tanh_norm
    )
        z_h = tanh( a_h );
    else
        z_h = tanh( (rowvec) ( a_h % ( a_h > 0 ) ) );
    
    return( z_h );

}

rowvec AE::meaning_from_z_h( rowvec z_h )
{

    if(
        mp.z_type == z_is_tanh_norm ||
        mp.z_type == z_is_relu_norm
    )
        z_h = z_h / norm( z_h, 2 );

    return( z_h );

}

mat AE::z_r_from_i_r( rowvec i_r, mat e_W_r )
{

    return( i_r * e_W_r );

}

mat AE::d_z_div_normz_d_z( rowvec z ){

    mat I = eye<mat>( z.n_elem, z.n_elem );

    double norm_z = norm( z , 2 );

    // d( a * b ) / dx = b * da/dx + a * db/dx
    // d( f(g(x) ) / dx = df/dx( g(x) ) * dg(x)/dx
    mat dnormz_dz = I / norm_z - z.t() * z / pow( norm_z, 3 );

    return( dnormz_dz );

}

mat AE::d_z_hl_d_a_hl( rowvec z_h )
{

    mat sech_p2;

    if( 
        mp.z_type == z_is_relu ||
        mp.z_type == z_is_relu_norm        
    )
        sech_p2 = diagmat( (rowvec)( ones<rowvec>( z_h.n_elem ) % ( z_h > 0 ) ) ); // tested
    else
        sech_p2 = diagmat( (rowvec)( 1 - pow( z_h, 2 ) ) ); // tested
    
    if(
        mp.z_type == z_is_tanh_norm ||
        mp.z_type == z_is_relu_norm
    ){
        
        // TODO the z_h that you get is already the normalized z_h

        mat dnormz_dz = d_z_div_normz_d_z( z_h );

        mat d_z_hl_d_a_hl = dnormz_dz * sech_p2;

        return( d_z_hl_d_a_hl );

    }else{

        return( sech_p2 );

    }

}

rowvec AE::deltas_c_from_z_c( rowvec z_c, int y )
{

    // optionnally match original implementation
    rowvec deltas_c = zeros<rowvec>( n_out );

    int is_target;
    for( int i = 0 ; i < deltas_c.n_elem ; i ++ ){
        is_target = 0;
        if( i == y ) is_target = 1;
        deltas_c[ i ] =
            mp.log_square_loss ?
            - ( y - z_c[i] ) * z_c[i] * ( 1 - z_c[i] ):
            ( z_c[i] - is_target );
    }

    return( deltas_c );

}

rowvec AE::softmax_from_a_c( rowvec a_c )
{

    return( exp( a_c ) / sum( exp( a_c ) ) );

}

rowvec AE::sigmoid_from_a_c( rowvec a_c )
{

    return( 1 / ( 1 + exp( - a_c ) ) );

}

mat AE::z_c_from_i_c( rowvec i_c, mat e_W_c )
{
    
    rowvec a_c =  i_c * e_W_c;

    if( mp.log_square_loss ){
        return( sigmoid_from_a_c( a_c ) );
    }else{
        return( softmax_from_a_c( a_c ) );
    }

}

double AE::loss_c( rowvec z_c, int y, bool is_term ){

    // optionnally match original implementation
    rowvec y_minus_z_c;
    if( mp.log_square_loss ){ 
        y_minus_z_c = z_c;
        y_minus_z_c.transform( [ &y ]( double v ){ return ( y - v ); } );
    }

    double loss = 
        ( is_term ? mp.nterm_w: 1 ) * 
        ( mp.log_square_loss ?
            0.5 * dot( y_minus_z_c, y_minus_z_c ):
            ( - log( z_c[ y ] ) ) );



    return( loss );

}

double AE::recc_err_from_i_h_z_r(
    rowvec i_h, rowvec z_r, double w_a, double w_b    
)
{
    
    if( not mp.weight_rec ){ w_a = 1; w_b = 1; }

    rowvec z_r_used = z_r;
    if( mp.norm_rec ){
        z_r_used( span( 0, mp.w_length - 1 ) ) = 
            meaning_from_z_h( z_r( span( 0, mp.w_length - 1 ) ) );
        z_r_used( span( mp.w_length, 2 * mp.w_length - 1 ) ) = 
            meaning_from_z_h( z_r( span( mp.w_length, 2 * mp.w_length - 1 ) ) );
    }

    rowvec diff_a = 
        ( 
            1 - mp.keep_direct + 
            mp.keep_direct * i_h( span( 0, mp.w_length - 1 ) ) 
        ) - 
        mp.keep_indirect * z_r_used( span( 0, mp.w_length - 1 ) );

    rowvec diff_b =
        (
            1 - mp.keep_direct +
            mp.keep_direct * i_h( span( mp.w_length, 2 * mp.w_length - 1 ) ) 
        ) -
        mp.keep_indirect * z_r_used( span( mp.w_length, 2 * mp.w_length - 1 ) );

    return(
        (
            0.5 * w_a * dot( diff_a, diff_a )  +
            0.5 * w_b * dot( diff_b, diff_b )
        )
    );

}

int AE::print_params_norms( int nrm ){

    uint start = 0;
    mat * param;

    for( uint pi = 0; pi < params.size(); ++pi ){

        param = params[ pi ];

        if( not( mp.sup and mp.unsup ) 
            and ( params_names[pi] == "e_W_h" or params_names[pi] == "e_W_r" ) 
        ){

            span left = span( 0, mp.w_length - 1 );
            span right = span( mp.w_length, 2 * mp.w_length - 1 );

            vec param_lin_left =
                params_names[ pi ] == "e_W_h" ?
                vectorise( (*param).rows( left ) ):
                vectorise( (*param).cols( left ) );

            vec param_lin_right =
                params_names[ pi ] == "e_W_h" ?
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



    return( 0 );

}

int AE::load_model( std::string save_models_dir, uint ind ){

    std::string path;

    for( uint i = 0; i < params.size(); i++ ){

        path = save_models_dir + "/" + params_names[ i ] + std::string("_") +
            std::to_string( ind ) + ".h5";

        ( params[ i ] )->load( path );

    }

    return 0;

}

int AE::save_model( std::string save_models_dir, uint ind ){

    std::string path;

    for( uint i = 0; i < params.size(); i++ ){

        path = save_models_dir + "/" + params_names[ i ] + std::string("_") +
            std::to_string( ind ) + ".h5";

        ( params[ i ] )->save( path );

    }

    return 0;

}

int AE::remove_model( std::string save_models_dir, uint ind ){

    std::string path;

    for( uint i = 0; i < params.size(); i++ ){

        path = save_models_dir + "/" + params_names[ i ] + std::string("_") +
            std::to_string( ind );

        remove( path.c_str() );

    }

    cout << "# removed model " << ind << endl;

    return 0;

}
