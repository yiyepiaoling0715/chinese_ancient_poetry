import os
from random import shuffle

from char_dict import CharDict
from paths import raw_dir, poems_path, check_uptodate
from singleton import Singleton
from utils import split_sentences

_corpus_list = ['qts_tab.txt']
"""
    从文件读取作者，诗句，题目， 只保留诗句，根据符号切分诗句，然后写进 poem.txt
"""


def _gen_poems():
    print("Parsing poems ...")
    chardict = CharDict()
    corpus = list()
    for corpus_name in _corpus_list:
        corpuspath = os.path.join(raw_dir, corpus_name)
        with open(corpuspath, 'r') as fr:
            for index, line in enumerate(fr):
                if index == 0:
                    continue
                all_in_char = True
                sentences = split_sentences(line.split()[3])
                for sentence in sentences:
                    for char in sentence:
                        if chardict[char] < 0:
                            all_in_char = False
                            # raise ValueError('char\t{}\t不在char_dict里边？？'.format(char))
                if all_in_char:
                    corpus.append(sentences)
    corpus_sorted = sorted(corpus, key=lambda x: (-len(x[0]), -len(x)))
    with open(poems_path, 'w') as fw:
        for sentences in corpus_sorted:
            fw.write(' '.join(sentences) + '\n')
    print("Finished parsing %s." % corpus)


class Poems(Singleton):

    def __init__(self):
        self.poems = list()
        for corpus_name in _corpus_list:
            corpuspath = os.path.join(raw_dir, corpus_name)
            if not check_uptodate(corpuspath):
                _gen_poems()
        if not check_uptodate(poems_path):
            _gen_poems()
        for corpus_name in _corpus_list:
            corpuspath = os.path.join(raw_dir, corpus_name)
            with open(corpuspath, 'r') as fr:
                for line in fr:
                    # print(line)
                    sentences = split_sentences(line.strip('\r\n ').split()[-1])
                    # print(sentences)
                    self.poems.append(sentences)
            print('self.poems==>', len(self.poems))

    def __getitem__(self, index):
        if index < 0 or index >= len(self.poems):
            return None
        return self.poems[index]

    def __len__(self):
        return len(self.poems)

    def __iter__(self):
        return iter(self.poems)

    def shuffle(self):
        shuffle(self.poems)


# For testing purpose.
if __name__ == '__main__':
    poems = Poems()
    for i in range(10):
        print(' '.join(poems[i]))
