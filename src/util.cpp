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

#include<common.h>
#include<util.h>

int rand_product()
{

    mat A = randu<mat>(4,5);
    mat B = randu<mat>(4,5);

    cout << A*B.t() << endl;

    return 0;
}

mat get_rand_wmat( int n_in, int n_out )
{

    // return randu<mat>( n_in, n_out ) *
    //   2  * sqrt( 6. /  (n_in + n_out) ) - sqrt( 6. /  (n_in + n_out) )
    //;

    double base = sqrt( 6. / ( n_in + n_out + 1 ) );
    return randu<mat>( n_in, n_out ) * 2  * base - base
    ;
}

double auc( vec preds, uvec labels )
{
    // one class auc, preds are for the positive class

    double auc;
    
    int n = preds.n_elem; 
    int n_pos = sum( labels );

    uvec order = sort_index( preds, "descend" );
    
    vec preds_ord = preds( order );
    uvec labels_ord = labels( order );

    uvec above = 
        cumsum( ones<uvec>( labels_ord.n_elem ) )
        - cumsum( labels_ord );
    
    auc = ( 1. - double(sum( above % labels_ord )) /
            ( double( n_pos ) * double( n - n_pos ) ) );

    return auc;
}

double accuracy( vec preds, uvec labels )
{
    // one class auc, preds are for the positive class

    double accuracy;
    
    double TP = sum( ( preds >= 0.5 ) % labels );
    double TN = sum( ( preds <  0.5 ) % ( labels == 0 )  );
    double FP = sum( ( preds >= 0.5 ) % ( labels == 0 ) );
    double FN = sum( ( preds <  0.5 ) % labels );

    accuracy = ( TP + TN ) / ( TP + TN + FP + FN );

    return accuracy;
}

function<double (vec, uvec)> get_perf_function( std::string desc ){

    if( desc == "auc" )
        return auc;
    else if( desc == "accuracy" )
        return accuracy;
    else{
        cerr << "performance measure: " << desc << " not known, please check" <<
            "options" << endl << endl;
        exit( 0 );
    }
}

bool perf_a_better_than_b( std::string desc, double a, double b ){
    
    uint res = false;
    
    if( desc == "auc" && a > b )
        res = true;
    else if( desc == "accuracy" && a > b )
        res = true;

    if( bigger_perf_is_better.count( desc ) == 0 ){
        cerr << "performance measure: " << desc << 
            " not known, please check the option" << endl;
        cerr << "known measures: ";
        for( auto kv: bigger_perf_is_better ){
            cerr << kv.first << " ";
        }
        cerr << endl;
        exit( 1 );
    }

    return( res );

}


vec probs_for_labels( mat probs, uvec labels )
{

    vec out_probs = zeros<vec>( labels.n_elem );

    for( int lab = 0; lab < labels.n_elem; lab++ ){

        out_probs( lab ) = probs( lab, labels[ lab ] );

    }

    return( out_probs );
}

double macro_perf(
    mat preds, uvec labels,
    function<double ( vec, uvec) > perffunc
)
{

    double macro_perf_val = 0;
    double curr_perf;

    for( uint lab = 0; lab < preds.n_cols ; lab++ ){

        vec preds_for_label = preds.col( lab );
        
        uvec labels_curr = labels;

        uvec zeros = find( labels_curr != lab );
        uvec ones = find( labels_curr == lab );

        labels_curr( zeros ).fill( 0 );
        labels_curr( ones ).fill( 1 );

        // preds_for_label( zeros ) = 1 - preds_for_label( zeros );

        curr_perf = perffunc( preds_for_label, labels_curr );

        macro_perf_val += curr_perf;

    }


    return macro_perf_val / preds.n_cols;

}

uvec equalize( uvec labels_in )
{

    uvec vals = unique( labels_in ); // vals.n_elem
    vector<uvec> indices( vals.n_elem );
    uvec sizes = zeros<uvec>( vals.n_elem );

    for( int i = 0; i < vals.n_elem ; i++ ){
        
        indices[ i ] = find( labels_in == vals[ i ] );
        sizes[ i ] = indices[ i ].n_elem;

        cout << "# label " << i << " #: " << sizes[ i ] << endl;

    }

    int min_size = min( sizes );
    uvec selection = zeros<uvec>( min_size * vals.n_elem );

    for( uint i = 0; i < vals.n_elem ; i++ )
            std::random_shuffle( indices[ i ].begin(), indices[ i ].end() );

    for( uint i = 0; i < vals.n_elem ; i++ )
        for( uint j = 0; j < min_size ; j++ )
            selection[ j * vals.n_elem + i ] = indices[ i ][ j ];

    return( selection );

}

uvec order_labels_alternate( uvec labels, int equalize ){
    /*
    get an order of labels that alternate between values.
    */
    
    uvec out;

    uvec vals = unique( labels );
            
    vector<uvec> indices( vals.n_elem );
            
    uvec sizes = zeros<uvec>( vals.n_elem );

    for( int i = 0; i < vals.n_elem ; i++ ){
        
        uvec indices_for_val = find( labels == vals[ i ] );
        indices[ i ] = indices_for_val; 
        sizes[ i ] = indices_for_val.n_elem;

        cout << "# label " << i << " #: " << sizes[ i ] << endl;

    }

    if( equalize == 1 ){

        int min_size = min( sizes );
        out = zeros<uvec>( vals.n_elem * min_size );

        for( int i = 0; i < min_size; i++ )
            for( int val = 0 ; val < vals.n_elem ; val++ )
                out[ i * vals.n_elem + val ] = indices[ val ][ i ];

    }else{

        int max_size = max( sizes );
        int idx = 0;

        out = zeros<uvec>( sum( sizes ) );

        for( int i = 0; i < max_size; i++ )
            for( int val = 0 ; val < vals.n_elem ; val++ ){
                if( i < sizes[ val ] ){
                    out[ idx ] = indices[ val ][ i ];
                    idx++;
                }
            }

    }

    return out;

}

