#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import collections
import json
import os

from paths import raw_dir, wordrank_path, check_uptodate
from poems import Poems
from segment import Segmenter
from singleton import Singleton

_stopwords_path = os.path.join(raw_dir, 'stopwords.txt')

_damp = 0.85


def _get_stopwords():
    stopwords = set()
    with open(_stopwords_path, 'r') as fr:
        for line in fr:
            stopwords.add(line.strip('\r\n '))
    return stopwords


# TODO: try other keyword-extraction algorithms. This doesn't work well.

class RankedWords(Singleton):

    def __init__(self):
        if not check_uptodate(wordrank_path):
            self._do_text_rank()
        self.word2rank = dict()
        self.rank2word = dict()
        with open(wordrank_path, 'r') as fr:
            word2score = json.load(fr)
            for rank, word in enumerate(word2score):
                self.word2rank[word[0]] = rank
                self.rank2word[rank] = word[0]
        if 1 in self.word2rank or 28 in self.word2rank or '28' in self.word2rank:
            print('1 在 scored内部')
        print('self.word2rank', len(self.word2rank))
        print('self.rank2word', len(self.rank2word))

    def _do_text_rank(self):
        """scores，给所有词设置 双score    每个句子进行词语之间的组合， 迭代词语分数   给分数排序"""
        print("Do text ranking ...")
        segment = Segmenter()
        scores = dict()
        adjlists = self._get_adjlists()
        for word in adjlists:
            scores[word] = [1.0, 1.0]

        for word, adjust in adjlists.items():
            sums = sum([w for _, w in adjust.items()])
            for word, weight in adjust.items():
                adjust[word] = weight / sums
        _damp = 0.85
        while True:
            for word, adjust in adjlists.items():
                scores[word][1] = (1 - _damp) + _damp * sum(
                    [scores[word][0] * adjlists[other][word] for other in adjust])
            eps = 0.0
            for word in scores:
                eps = max(eps, scores[word][0] - scores[word][1])
                scores[word][0] = scores[word][1]
            print('eps=>', eps)
            if eps < 0.05:
                break

        def tmp_key(x):
            word, score = x
            return 0 if word in segment.sxhy_dict else -1, -score

        word_and_scores = sorted([(word, score[0]) for word, score in scores.items()], key=tmp_key)
        with open(wordrank_path, 'w') as fw:
            json.dump(word_and_scores, fw)
        return scores

    def _get_adjlists(self):
        poems = Poems()
        segmenter = Segmenter()
        adjlists = collections.defaultdict(dict)
        for poem_set in poems:
            for poem in poem_set:
                words = segmenter.segment(poem)
                for i in range(len(words) - 1):
                    for j in range(i + 1, len(words)):
                        if words[j] not in adjlists[words[i]]:
                            adjlists[words[i]][words[j]] = 1.0
                        else:
                            adjlists[words[i]][words[j]] += 1.0
                        if words[i] not in adjlists[words[j]]:
                            adjlists[words[j]][words[i]] = 1.0
                        else:
                            adjlists[words[j]][words[i]] += 1.0
        return adjlists

    def __getitem__(self, index):
        return self.rank2word[index]

    def __len__(self):
        return len(self.word2rank)

    def __iter__(self):
        return map(lambda x: x[0], self.word2rank.items())

    def __contains__(self, word):
        return word in self.word2rank

    def get_rank(self, word):
        return self.word2rank[word]


# For testing purpose.
if __name__ == '__main__':
    ranked_words = RankedWords()
    for i in range(100):
        print(ranked_words[i])
    print('111')
