#!/bin/bash
any_missing=0
for prog in "wget" "unzip"
do
    type $prog >/dev/null 2>&1 || { echo "missing: ${prog}"; any_missing=1; }
done 

if [[ any_missing -eq 1 ]]
then
    exit 1
fi

echo "Downloading..."
#wget -N -P example/data/ http://nlp.stanford.edu/~socherr/codeDataMoviesEMNLP.zip
#unzip -d example/data/movies example/data/codeDataMoviesEMNLP.zip
wget -N -P example/data/ http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
tar -zxvf example/data/rt-polaritydata.tar.gz -C example/data
