#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat clean_train_pos.txt clean_train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > sample_vocab.txt
