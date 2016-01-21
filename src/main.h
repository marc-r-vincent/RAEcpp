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

void pt_group_learn( PT_GROUP* ptg );
void pt_group_predict( PT_GROUP* ptg );
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
);
struct dataset read_dataset(
    string labels_p,
    string phrases_p,
    string train_fold_p,
    string val_fold_p,
    string test_fold_p,
    int equalize
);
void copy_dataset( struct dataset * ds, struct dataset * ds_cop );
int check_options_compat( model_params mp, opt_params op, uint n_out );
int run_dataset( dataset ds_in, model_params mp, opt_params op,
    project_path pp );
int test_triplet( model_params mp, opt_params op );
int test_onval( project_path pp, opt_params op, model_params mp );
int test_dumb();
int test_auc();
int test_read( project_path pp );
int run( project_path pp, opt_params op, model_params mp );
int mem_test( uint do_reset );
int set_test_defaults( model_params * mp );
int help();
int main(int argc, char** argv);

