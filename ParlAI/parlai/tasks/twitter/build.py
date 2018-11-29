#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Download and build the data if it does not exist.

try:
    from emoji.unicode_codes import UNICODE_EMOJI
    import unidecode
except ImportError:
    raise ImportError('Please `pip install emoji unidecode` for the twitter task.')

from collections import Counter, defaultdict

import parlai.core.build_data as build_data
import numpy as np
import pickle
import nltk
import math
import os


def replace_emoji(x):
    if x in UNICODE_EMOJI.keys():
        return ' ' + UNICODE_EMOJI[x].replace(':', '@') + ' '
    else:
        return x


def split_punctuation(x):
    return (
        x
        .replace('.', ' . ')
        .replace('. . .', '...')
        .replace(',', ' , ')
        .replace(';', ' ; ')
        .replace(':', ' : ')
        .replace('!', ' ! ')
        .replace('?', ' ? ')
        .replace('"', ' " ')
        .replace('(', ' ( ')
        .replace(')', ' ) ')
    )


def create_fb_format(data, dpath):
    fw1 = open(os.path.join(dpath, 'train.txt'), 'w')
    fw2 = open(os.path.join(dpath, 'valid.txt'), 'w')
    fw3 = open(os.path.join(dpath, 'test.txt'), 'w')
    fw4 = open(os.path.join(dpath, 'pmi.pkl'), 'wb')

    bigrams = []
    unigrams = []

    for i in range(0, len(data) - 1, 2):
        fout = fw1
        if (i % 500) == 0:
            fout = fw2
        elif (i % 500) == 2:
            fout = fw3
        use = True
        x = data[i].rstrip(' ').lstrip(' ').replace('\t', ' ')
        y = data[i + 1].rstrip(' ').lstrip(' ').replace('\t', ' ')
        x = x.replace('|', ' __PIPE__ ')
        y = y.replace('|', ' __PIPE__ ')
        x = ''.join(list(map(replace_emoji, x)))
        y = ''.join(list(map(replace_emoji, y)))
        x = split_punctuation(unidecode.unidecode(x))
        y = split_punctuation(unidecode.unidecode(y))

        x = x.split()
        y = y.split()

        x_bigram = list(nltk.bigrams(x))
        y_bigram = list(nltk.bigrams(y))

        bigrams.append(x_bigram)
        bigrams.append(y_bigram)

        unigrams.append(x)
        unigrams.append(y)

        x = ' '.join(x)
        y = ' '.join(y)

        if len(x) < 1 or len(y) < 1:
            use = False
        if use:
            s = 'text:' + x + '\tlabels:' + y + '\tepisode_done:True'
            fout.write('{} \n'.format(s))

    bigram_counter = Counter([tuple(sorted(bigram)) for bigram_sublist in bigrams for bigram in bigram_sublist])
    unigram_counter = Counter([unigram for unigram_sublist in unigrams for unigram in unigram_sublist])

    total_bi = sum(bigram_counter.values(), 0.0)
    for key in bigram_counter:
        bigram_counter[key] /= total_bi

    total_uni = sum(unigram_counter.values(), 0.0)
    for key in unigram_counter:
        unigram_counter[key] /= total_uni

    pmi = defaultdict(float)
    for bigram, bigram_prob in bigram_counter.items():
        pmi[bigram] = math.log(bigram_prob / (unigram_counter[bigram[0]]*unigram_counter[bigram[1]]))

    pickle.dump(pmi, fw4)

    fw1.close()
    fw2.close()
    fw3.close()
    fw4.close()


def build(opt):
    version = 'v1.1'
    dpath = os.path.join(opt['datapath'], 'Twitter')

    if not build_data.built(dpath, version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        fname1 = "twitter_en_big.txt.gz.partaa"
        fname2 = "twitter_en_big.txt.gz.partab"
        url = 'https://github.com/Marsan-Ma/chat_corpus/raw/master/'
        build_data.download(url + fname1, dpath, fname1)
        build_data.download(url + fname2, dpath, fname2)

        file1 = os.path.join(dpath, fname1)
        file2 = os.path.join(dpath, fname2)
        file3 = "twitter_en_big.txt.gz"
        outzipfile = os.path.join(dpath, file3)
        build_data.cat(file1, file2, outzipfile)

        import gzip
        with gzip.open(outzipfile, 'r') as f:
            file_content = bytes.decode(f.read())
        data = file_content.split('\n')[2:]
        create_fb_format(data, dpath)
        os.remove(outzipfile)

        # Mark the data as built.
        build_data.mark_done(dpath, version)
