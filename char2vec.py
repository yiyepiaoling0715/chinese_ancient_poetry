#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import itertools
from char_dict import CharDict
from gensim import models
from numpy.random import uniform
from paths import char2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import CHAR_VEC_DIM
import numpy as np
import os
import multiprocessing

def _gen_char2vec():
    print("Generating char2vec model ...")
    char_dict=CharDict()
    cpu_count=multiprocessing.cpu_count()
    poems=Poems()
    poems_str=[list(line) for line in list(itertools.chain.from_iterable(poems))]
    # for item in poems_str:
    #     print(item)
    # model=models.Word2Vecrd2Vec(sentences=poems, size=CHAR_VEC_DIM, alpha=0.025, window=5, min_count=5)
    model=models.Word2Vec(sentences=poems_str, size=CHAR_VEC_DIM, alpha=0.025, window=2, min_count=2,
                 workers=cpu_count, min_alpha=0.0001,sg=0, hs=1, negative=5,
                 cbow_mean=1, hashfxn=hash, iter=30, null_word=0,trim_rule=None, sorted_vocab=1)
    embedding=uniform(-1.0,1.0,size=[len(char_dict),CHAR_VEC_DIM])
    # print(len(model.wv))
    # for word in model.vocabulary.:
    #     print(word)
    counter_yes,counter_no=0,0
    for index,word in char_dict:
        if word in model.wv:
            embedding[index]=model.wv[word]
            counter_yes+=1
        else:
            counter_no+=1
            print('{}不在wv中'.format(word))
    print('有wv的字{}个没有wv的字{}个'.format(counter_yes,counter_no))
    np.save(char2vec_path,embedding)
class Char2Vec(Singleton):

    def __init__(self):
        if not check_uptodate(char2vec_path):
            _gen_char2vec()
        self.embedding = np.load(char2vec_path)
        self.char_dict = CharDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        # print('ch,self.char_dict.char2id(ch)',ch,self.char_dict.char2id(ch))
        return self.embedding[self.char_dict.char2id(ch)]
    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, CHAR_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    char2vec = Char2Vec()

