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

#ifndef __PROGRESS_MGR_H_INCLUDED__
#define __PROGRESS_MGR_H_INCLUDED__

class PROGRESS_MGR
{
/**
 * responsible for storing model, keeping track of previous models
 *
 */

public:

    PROGRESS_MGR(
       project_path pp,
       int keep_sel_size,
       AE * rae,
       struct opt_params op
    ); // constructor

    mat best_perfs;
    int keep_sel_size;
    project_path pp;
    AE * rae;
    struct opt_params op;

    mat save_progress(
        uint epoch, uint batch, struct perf_probs_labels_rep ppl
    );

    mat get_best_list();
    int get_best_saved( bool ask, bool keep_list ); 
    int find_last_save();
    int backup_best_list();

};

#endif
