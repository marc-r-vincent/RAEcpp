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
//
uint basic_tree = 0; // basic_tree components are pairs of words in order
uint clip_gradients = 0; // 1 gives good result
double clip_constant = 1.1; // per element when to
double transmission = 1; // 1 gives good result, 0 with --to
uint clip_deltas = 0; // 1 gives good result
double clip_deltas_factor = 1 ; // 1 gives good result, 8 with no transmission
double const_param_norm = 0; // if 0, no rescaling, 4.5 was good (kinda)
uint do_gradient_checks = 0; //!< check if gradient is correct, disabled for speed
    // used in PARSE_TREE, main
uint replace_grad = 0; // used in RAE, main
uint verbose = 1;
