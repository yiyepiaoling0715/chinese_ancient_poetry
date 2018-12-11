import numpy as np

from char2vec import Char2Vec
from char_dict import CharDict
from char_dict import end_of_sentence, start_of_sentence
from paths import gen_data_path, plan_data_path
from poems import Poems
from rank_words import RankedWords
from segment import Segmenter
from utils import _BATCH_SIZE, _NUM_UNITS


class FuncUtils(object):
    def __init__(self):
        self.char2vec = Char2Vec()
        self.char_dict = CharDict()

    def fill_np_matrix(self, texts):
        maxlen = max(map(len, texts))
        matrix = np.zeros(shape=[_BATCH_SIZE, maxlen, _NUM_UNITS])
        for i in range(_BATCH_SIZE):
            for j in range(maxlen):
                matrix[i, j, :] = self.char2vec.get_vect(end_of_sentence())
        for index, text in enumerate(texts):
            matrix[index, :len(text)] = self.char2vec.get_vects(text)
        lens_seq = [len(texts[index]) if index < len(texts) else 0 for index in range(_BATCH_SIZE)]
        return matrix, np.array(lens_seq)

    def fill_targets(self, sentences):
        # todo 已验证
        targets = []
        for sentence in sentences:
            targets.extend(map(self.char_dict.char2id, sentence))
        return np.array(targets)


def gen_train_data():
    """获取每一句的keywords，拼起来写入文件"""
    print("Generating training data ...")
    segmenter = Segmenter()
    poems = Poems()
    ranked_words = RankedWords()

    gen_data = list()
    plan_data = list()

    valid = True
    counter_line = 0
    print('len(poems)==>', len(poems))
    for poem in poems:
        # print(len(poem))
        if len(poem) != 4:
            # print(poem)
            valid = False
            continue
        context = start_of_sentence()
        keywords = list()
        for sentence in poem:
            counter_line += 1
            keyword = ''
            if len(sentence) != 7:
                valid = False
                break
            filterwords = list(filter(lambda x: x in ranked_words, segmenter.segment(sentence)))
            if filterwords:
                keyword = filterwords[0]
            for word in filterwords:
                # print('word==>',word)
                if ranked_words.get_rank(word) < ranked_words.get_rank(keyword):
                    keyword = word
            if keyword:
                gen_line = sentence + end_of_sentence() + \
                           '\t' + keyword + '\t' + context + '\n'
                keywords.append(keyword)
                gen_data.append(gen_line)
                context += sentence + end_of_sentence()
        plan_data.append(' '.join(keywords))
    with open(plan_data_path, 'w') as fw:
        for data_iter in gen_data:
            fw.write(data_iter + '\n')
    with open(gen_data_path, 'w') as fw:
        for data_iter in gen_data:
            fw.write(data_iter)

    print('counter_line==>', counter_line)
    del segmenter, poems, ranked_words


def batch_train_data(batch_size):
    """ Training data generator for the poem generator."""
    gen_train_data()  # Shuffle data order and cool down CPU.
    keywords = []
    contexts = []
    sentences = []
    counter = 0
    with open(gen_data_path, 'r') as fin:
        for line in fin:
            toks = line.strip().split('\t')
            # print(toks)
            sentences.append(toks[0])
            keywords.append(toks[1])
            contexts.append(toks[2])
            if len(keywords) % batch_size == 0:
                # print('keywords==>',keywords)
                # print('contexts==>',contexts)
                # print('sentences==>',sentences)
                yield keywords, contexts, sentences
                keywords.clear()
                contexts.clear()
                sentences.clear()
                # print('batch_counter',counter)
            counter += 1


if __name__ == '__main__':
    for item in batch_train_data(2):
        print('item==>', item)
