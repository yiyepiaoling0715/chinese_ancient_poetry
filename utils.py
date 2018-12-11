
def is_cn_char(ch):
    """ Test if a char is a Chinese character. """
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def is_cn_sentence(sentence):
    """ Test if a sentence is made of Chinese characters. """
    for ch in sentence:
        if not is_cn_char(ch):
            return False
    return True

def split_sentences(text):
    """ Split a piece of text into a list of sentences. """
    sentences=[]
    i=0
    for j in range(len(text)+1):
        if j==len(text) or text[j] in [u'，', u'。', u'！', u'？', u'、', u'\n']:
            sentence=text[i:j]
            i=j+1
            if sentence:
                sentences.append(sentence)
    return tuple(sentences)


NUM_OF_SENTENCES = 4
CHAR_VEC_DIM = 512
_BATCH_SIZE = 64
_NUM_UNITS = 512
_logwriter_dir='./tensorboard'

if __name__=='__main__':
    out=split_sentences('你是我系内的一首歌？不论结局会如何。好想问你')
    print(out)
