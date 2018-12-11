from paths import raw_dir, char_dict_path, check_uptodate
from singleton import Singleton
from utils import is_cn_char
import os

MAX_DICT_SIZE = 6000

_corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
                'yuan.all', 'ming.all', 'qing.all']
"""
获取指定文件列表所有的chars，建立 char2id,id2char
"""

def start_of_sentence():
    return '^'


def end_of_sentence():
    return '$'


def _gen_char_dict():  # 读取corpus 文件列表，获取字，字频，统计，写入文档。 仅仅 写入所有字
    print("Generating dictionary from corpus ...")
    char2freq_dict=dict()
    for file in _corpus_list:
        filepath=os.path.join(raw_dir,file)
        with open(filepath,'r') as fr:
            chrs=filter(is_cn_char,fr.read())
            for char in chrs:
                if char not in char2freq_dict:
                    char2freq_dict[char]=1
                else:
                    char2freq_dict[char]+=1
    char2freq_sorted=sorted(char2freq_dict.items(),key=lambda x:-x[1])
    with open(char_dict_path,'w') as fw:
        for idx in range(min(MAX_DICT_SIZE-2,len(char2freq_dict))):
            fw.write(char2freq_sorted[idx][0])
class CharDict(Singleton):
    def __init__(self):
        print('CharDict 初始化一次')
        self._char2id=dict()
        self._id2char=dict()
        if not check_uptodate(char_dict_path):
            _gen_char_dict()
        with open(char_dict_path,'r') as fr:
            chrs=list(filter(is_cn_char,fr.read()))
            self._char2id[start_of_sentence()]=0
            self._id2char[0]=start_of_sentence()
            for idx in range(len(chrs)):  # 给start_sentence,end_sentence 留位置
                self._char2id[chrs[idx]]=idx+1
                self._id2char[idx+1]=chrs[idx]
            self._char2id[end_of_sentence()]=len(chrs)+1
            self._id2char[len(chrs)+1]=end_of_sentence()
            print('len(self._char2id)==>',len(self._char2id))
            print('len(self._id2char)==>',len(self._id2char))
    def char2id(self, ch):
        if ch in self._char2id:
            return self._char2id[ch]
        else:
            return -1
    def id2char(self, idx):
        return self._id2char[idx]

    def __len__(self):
        return len(self._id2char)

    def __iter__(self):
        return iter(self._id2char.items())
    def __getitem__(self, item):
        return self.char2id(item)
    def __contains__(self, ch):
        return ch in self._char2id

if __name__=='__main__':
    chardict_class=CharDict()
    for item in chardict_class:
        print(item)