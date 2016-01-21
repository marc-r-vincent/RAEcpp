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

#ifndef __UTIL_H_INCLUDED__
#define __UTIL_H_INCLUDED__

int rand_product();
mat get_rand_wmat( int n_in, int n_out );
double auc( vec preds, uvec labels );
double accuracy( vec preds, uvec labels );
vec probs_for_labels( mat probs, uvec labels );
double macro_perf(
    mat preds, uvec labels,
    function<double ( vec, uvec) > perffunc
);
uvec equalize( uvec labels_in );
uvec order_labels_alternate( uvec labels, int equalize );
function<double (vec, uvec)> get_perf_function( std::string desc );
bool perf_a_better_than_b( std::string desc, double a, double b );
#endif
