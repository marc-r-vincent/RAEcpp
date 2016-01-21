# Implementation of the  Recursive Autoencoder Model by socher et al. EMNLP 2011
# Copyright (C) 2015 Marc Vincent
# 
# RAEcpp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http:#www.gnu.org/licenses/>.


import re
import numpy as np
import codecs

ex_pth = "./example/data/rt-polaritydata/"

f = codecs.open( ex_pth + "rt-polarity.pos", "r", encoding='cp1252' )
lines = f.readlines()
n_pos = len( lines )
f.close()

f = codecs.open( ex_pth + "rt-polarity.neg", "r", encoding='cp1252'  )
lines += f.readlines()
n_neg = len( lines ) - n_pos
f.close()

labels = np.append( np.repeat( 1, n_pos ), np.repeat( 0, n_neg ) )
labels = labels.astype( 'int32' )

# get rid of brackets
# ( could join ellipsis points too and separate possessives and negation )
lines = [
    re.sub( "\[|\]| \n", "", line)
    for line in lines
]

# get dictionnary
words = np.unique( re.split( ' ', ' '.join( lines ) ) ) 
dictionnary = {
    word : index 
    for index, word in enumerate( words )
}

# transform original text data to an array of int array where ints are indices
# w.r.t the dictionnary
int_lines = [
    [
        dictionnary[ word ]
        for word in re.split( ' ', line )
    ]
    for line in lines
]

# gets the final data as an array of ints
data = np.array( [ len( int_lines ) ] + [
    elem
    for line in int_lines
        for elem in [ len( line ) ] + line
], dtype = 'int32' )

# save as array of int in binary file
data.tofile( './example/data/panglee.dat' )

# save dictionnary
dfile = codecs.open( "./example/data/rae_panglee.dat", "w", "utf-8" )

for w in words:
    dfile.write( w + "\n" )

dfile.close()

# save labels
labels.tofile( './example/data/panglee.tgt' )

# divide in train, validation, test ( stratified, not rand )
def split( prop, labels ):
    """ split function 

    prop : test proportion, float in ]0,1[
    labels : numpy array of int

    returns array of array, first array = train indices, second test
    """
    train_val = [[], []]
    for lab in np.unique( labels ):
        lab_idx = np.where( labels == lab )[0]
        to_take = int( round( len( lab_idx ) * prop ) )
        train_val[1] += lab_idx[ range( 0, to_take ) ].tolist()
        train_val[0] += lab_idx[
                range( to_take, len( lab_idx ) ) ]. tolist()
    return( train_val )

trainval, test = split( 0.1, labels )
lab_trainval = labels[ trainval ]

train, val = split( 0.1, lab_trainval )
train = np.array( trainval )[ train ].tolist()
val = np.array( trainval )[ val ].tolist()

# save train, val, test split
np.array( train, dtype='int32' ).tofile( './example/data/panglee_train.idx' )
np.array( val, dtype='int32' ).tofile( './example/data/panglee_val.idx' )
np.array( test, dtype='int32' ).tofile( './example/data/panglee_test.idx' )
