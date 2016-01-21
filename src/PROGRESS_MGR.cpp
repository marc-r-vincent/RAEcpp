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
#include <glob.h>
#include <regex>
#include <PROGRESS_MGR.h>

PROGRESS_MGR::PROGRESS_MGR(
   project_path pp,
   int keep_sel_size,
   AE * rae,
   struct opt_params op
){
    best_perfs = zeros<mat>( keep_sel_size, 5);
    best_perfs += -1;
    this->keep_sel_size = keep_sel_size;
    this->pp = pp;
    this->rae = rae;
    this->op = op;
}

// TODO add epoch, batch, lr info (first two rows)
// the rest should be handled by a sink manager for structs op and mp
// TODO use a single hdf5 file instead for model saving
mat PROGRESS_MGR::save_progress(
    uint epoch, uint batch, struct perf_probs_labels_rep ppl
){

    int lowest_perf_idx = (int) ceil( best_perfs( 0, 1 ) );
    // get last idx (-1 if none)
    uvec order = sort_index( best_perfs.col( 1 ), "descend" );
    int curr_idx = (int) ceil( best_perfs( order[0], 1 ) );
    // add 1 to make current idx
    curr_idx++;

    rowvec current = zeros<rowvec>( 5 );
    current[ 0 ] = ppl.perf;
    current[ 1 ] = curr_idx;
    current[ 2 ] = epoch;
    current[ 3 ] = batch;
    current[ 4 ] = op.learning_rate;

    best_perfs = best_perfs.rows( span( 1, best_perfs.n_rows - 1 ) );
    best_perfs = join_cols( best_perfs, current );

    // printf( "%6  ", perf_type );

    if(
        (
            (uvec) find( (uvec)( best_perfs.col( 0 ) < ppl.perf ) )
        ).n_elem > 0
    ){

        order = sort_index( best_perfs.col( 0 ), "ascend" );

        best_perfs = best_perfs.rows( order );
        
        cout << "# "<< best_perfs.n_rows << " best perfs so far:" << endl;

        vector< string > desc = {
            ppl.perf_type, "saves", "epoch", "batch", "l.rate", "curr." 
        };

        for( string dpart: desc ){
            if( dpart.size() > 8 ) dpart = dpart.substr( 0, 5 ) + ".";
            printf( "%8s  ", dpart.c_str() );
        }

        cout << endl;

        for( uint rowi = 0 ; rowi < best_perfs.n_rows; rowi++ ){
            
            int row_epoch;
            for( uint coli = 0; coli < best_perfs.n_cols; coli++ ){

                if( coli == 2 ) row_epoch = best_perfs( rowi, coli ); 

                printf( "%8.4g", best_perfs( rowi, coli ) );
                if( coli < best_perfs. n_cols - 1 ) cout << "  ";
                else{
                    if( row_epoch == epoch ) printf( "%8s",  "<---" );
                    cout << endl;
                }

            }

        }
        // cout << best_perfs;

        std::string path;

        path = pp.save_models_dir + "/best_models";
        best_perfs.save( path );

        // save preds only if we have preds
        if( ppl.probs.n_elem != 0 ){

            ofstream pfile (
                pp.save_predictions_dir + "/preds_val_" + std::to_string(curr_idx) + ".csv"
            );
            pfile << join_rows( ppl.probs, ppl.labels );
            pfile.close();

        }

        // save val representations anyways
        char oldfill = cout.fill();
        cout.fill( ' ' );

        int idx = 0;

        mat::const_row_iterator st = ppl.representations.begin_row(0);
        mat::const_row_iterator en = ppl.representations.end_row(0);
        st++;

        // TODO header useless right now
        std::string header = std::accumulate(
            st, 
            en,
            std::string( "C#f" ) + std::to_string( 0 ),
            [ &idx ]( const std::string & prev, double n ){
                idx += 1;
                return prev + " C#f" + std::to_string( idx );
            }
        );
        header += " D#class";

        path = pp.save_representations_dir + "/representations_val_" + std::to_string(curr_idx) +
            ".h5";

        (
            ( mat ) join_rows( ppl.representations, ppl.labels )
        ).save( path, hdf5_binary );

        cout.fill( oldfill );

        // save model
        cout << endl << "# epoch " << epoch << ", batch " << batch
            << ": (saved)" << endl;
        rae->save_model( pp.save_models_dir, curr_idx );

        // delete the lowest performing model in kept history
        if( lowest_perf_idx > -1 ){
            rae->remove_model( pp.save_models_dir, lowest_perf_idx );
        }
    }

    return( best_perfs );

}

mat PROGRESS_MGR::get_best_list(){

    std::string path = pp.save_models_dir + "/best_models";

    if( FILE *file = fopen( path.c_str(), "r" ) ){

        fclose(file);
        
        best_perfs.load( path );

    }

    return( best_perfs );

}

int PROGRESS_MGR::backup_best_list(){

    std::string path = pp.save_models_dir + "/best_models";

    uint ind = 0;
    while(
        FILE *file = fopen(
            (
                path + "_prev_" + std::to_string( ind )
            ).c_str(), "r"
        )
    ){
        fclose( file );
        ++ind;
    }

    rename(
        (const char*) path.c_str(),
        ( path + "_prev_" + std::to_string( ind ) ).c_str()
    );

    return( 0 );

}

int PROGRESS_MGR::get_best_saved( bool ask, bool keep_list ){  

    int max_model_idx = -1;

    // uint keep_sel_size = 10; // if keep size > 0 do not keep all, just the best
    best_perfs = get_best_list();

    if( keep_sel_size > 0 ){
    
        if( best_perfs( best_perfs.n_rows - 1, 1 ) > -1 ){

            max_model_idx = (int) best_perfs( best_perfs.n_rows - 1, 1 );

            cout << "# best model found, idx: " << max_model_idx <<
                " with perf: " << best_perfs( best_perfs.n_rows - 1, 0 )
                << endl;

        }

    }
    else{
        max_model_idx = find_last_save();
        cout << "# last model found, idx: " << max_model_idx << endl << std::flush;
    }

    uint start_round = 0;
    if( max_model_idx > -1 ){
        
        std::string answer;

        if( ask ){
            cout << "# found model idx: " << max_model_idx << ", start from here (y/n/c) ?" 
                << endl << std::flush;
            getline( std::cin, answer) ;
        }else{
            answer = "y";
        }

        if(
            answer == "y" || answer == "Y" || answer == "yes" || answer == ""
        ){

            if( not keep_list ){
                backup_best_list();
            }
            cout << "ok, starting from " <<  max_model_idx << endl <<
                std::flush;
            rae->load_model( pp.save_models_dir, max_model_idx );
            start_round = max_model_idx;

        }else if(
            answer == "c"        
        ){

            exit(0);

        }else{
    
            if( not keep_list ){
                backup_best_list();
                best_perfs.fill( 0 );
                best_perfs += -1;
            }

        }
    }

    return( 0 );

}

int PROGRESS_MGR::find_last_save(){
    
    int idx = -1;
    int c_idx;

    const  std::string patt = pp.save_models_dir + "/L_*";
    
    glob_t glob_result;

    // TODO adapt to non unix system ?
    glob( patt.c_str(), GLOB_TILDE, NULL, &glob_result );

    std::regex before_underscore(".*_");
    
    for( uint i=0; i < glob_result.gl_pathc; ++i ){

        std::string curr_path = std::string( glob_result.gl_pathv[i] );

        c_idx = (int)
            stoi( std::regex_replace( curr_path, before_underscore, std::string("") ) );

        if( c_idx > idx ) idx = c_idx;

    }

    globfree( &glob_result );

    return idx;

}

