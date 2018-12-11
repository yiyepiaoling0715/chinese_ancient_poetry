import os

import jieba

from paths import raw_dir, sxhy_path, check_uptodate

_rawsxhy_path = os.path.join(raw_dir, 'shixuehanying.txt')

"""
分词，加载已有词典列表，然后对句子进行分词

"""


def _gen_sxhy_dict():
    print("Parsing shixuehanying dictionary ...")
    words = set()
    with open(_rawsxhy_path, 'r') as fin:
        for line in fin:
            if line.startswith('<'):
                continue
            for phrase in line.split(' ')[1:]:
                idx = 0
                while True:
                    if idx + 4 < len(phrase):
                        words.add(phrase[idx:idx + 2])
                        idx += 2
                    else:
                        break
                if idx < len(phrase):
                    for word in jieba.lcut(phrase[idx:]):
                        words.add(word)
    words_sorted = sorted(words, key=lambda x: -len(x))
    with open(sxhy_path, 'w') as fw:
        for word in words_sorted:
            fw.write(word + '\n')


class Segmenter(object):
    def __init__(self):
        if not check_uptodate(sxhy_path):
            _gen_sxhy_dict()
        with open(sxhy_path, 'r') as fr:
            self.sxhy_dict = set(fr.read().split())
        for word in self.sxhy_dict:
            jieba.add_word(word)
        # for word in self.sxhy_dict:
        #     print('sxhy_dict==>',word)

    def segment(self, sentence):  # 为何要这种逻辑，2个字2个字的循环， 而且最后分词时候没有任何diy_words??
        idx = 0
        words = []
        while idx + 4 < len(sentence):
            if sentence[idx:idx + 2] in self.sxhy_dict:
                words.append(sentence[idx:idx + 2])
                idx += 2
            else:
                break
        if idx < len(sentence):
            if sentence[idx:] in self.sxhy_dict:
                words.append(sentence[idx:])
            else:
                for word in jieba.lcut(sentence[idx:]):
                    words.append(word)
        return words


if __name__ == '__main__':
    segmenter = Segmenter()
    sentences = ['山在湖心如黛簇']
    for sentence in sentences:
        out = segmenter.segment(sentence)
        print(out)
    r = jieba.lcut('山在湖心如黛簇')
    print(r)
