RAEcpp Documentation
====================
A C++ 11 implementation of the Recursive Auto Encoder model described in Socher et al. "Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions, Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning." EMNLP 2011.  

Some modifications were made compared to the original publication such as the optionnal use of stochastic gradient descent, adagrad or adadelta for optimization as well as an experimental use of RELU activation function. By default unsupervised and supervised learning is not divided into two separate passes. As in the original paper batch optimization using l-BFGS is still available using either libLBFGS, or dlib.

Comes with a CLI and options to run it on an arbitrary dataset.

The dataset is made of four files. One representing all the phrases of the corpus plus three sets files corresponding to: train, validation, test.

The corpus file has the following binary format:
< m : long int >{ 1 }< < n : long int >{ 1 }< w : long int >{ n }> >{ m }

Using as a notation convention:
* data name and data type are separated by a colon and within angle brackets
* quantifiers relating to data in the immediate left bracket are enclosed in curly brackets
* m is the number of phrases in the corpus n is the number of words in a given phrase w is the index of a word in a given phrase

Long int is int32 by default ( using a bigger storage would require that you change armadillo's compilation flags ).

Relies on Armadillo for linear algebra, and runs on CPU. Mulithreading is available but depends on the batch size used during optimzation. Each thread will deal with a set of phrases concurrently, therefore to occupy k threads you should have at least k phrases in your batch. To keep your CPU busy you will realistically need k\*l batches where k depends on the number of cores available and l depends on the size of the training set and whatever overall batch size is the best for the chosen optimzation procedure.

Automatically saves training progress, best models and learned representations/embeddings of words and phrases.

PERFORMANCE
-----------

On an 8 core i7-4710HQ / 2.50 GHz processor the adagrad example file runs for an average of 2-3 min depending on the random initialization of parameters ( ie. the optimization might take from 3 to 5 epochs to complete ). Using lbfgs with default settings is slower with a satisfactory model found around 80-100 epochs.

DEPENDENCIES
------------

- First of all you need a GNU compiler supporting c++11 or later ( ie. version 4.7 or later )
- RAEcpp minimally requires *Armadillo*, *hdf5*, *dlib* and *eigen3* libraries. Dlib will be downloaded by cmake, if the other dependencies are present cmake should find them, if not you'll have to install them.
- For lBFGS type optimization either *dlib*, *libLBFGS* or both have to be present.
dlib relies on eigen3.
- To generate the complete documentation you need doxygen to be present
- To use the examples you have to have python with numpy installed

To install the dependencies on different systems you can use:
```
sudo apt-get install libarmadillo-dev libhdf5-7 libhdf5-dev libeigen3-dev wget # UBUNTU / apt
sudo port install armadillo hdf5 eigen3 eigen3-devel wget # MACOSX / macports
```

INSTALLING
----------

```
cd path-to-dir-containg-this-file
cmake CMakeLists.txt # check dependencies, making makefile
make # run make
doxygen doxyset.dx # generate documentation using doxygen
```

Should you have any trouble building libLBFGS you can edit the CMakeLists.txt file and set USE\_LIBLBFGS to OFF. If cmake warns that eigen3 is not found at the tested places -and you are sure you have installed it- you can edit the EIGEN\_DIRS list and add the path to your eigen3 installation.

RUNNING
-------

Scripts examples are provided to do training and testing on an example dataset ( Pang & Lee's movie dataset ).
You will need wget and to run the following commands:

```
chmod 755 ./utils/* 
./utils/down_pang_lee.sh # download the dataset 
python ./example/scripts/make_pl_ds.py # transform to RAEcpp format 
./example/scripts/run_adadelta.sh # train using adadelta and test
```

Generally speaking:
- To learn a model, give path to data and training subset ( sufficient for l-BFGS methods )
and optionnaly a validation subset ( necessary for non l-BFGS methods )
- To evaluate a model, give path to data and validation subset with labels different from -1
- To do predictions on a corpus, give a path to data and test subset

LICENSE
-------

Implementation of the  Recursive Auto Encoder model originally described by socher et al. in EMNLP 2011
Copyright (C) 2015 Marc Vincent.
Website: http://www.dsi.unifi.it/~vincent

RAEcpp is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http:www.gnu.org/licenses/>.
