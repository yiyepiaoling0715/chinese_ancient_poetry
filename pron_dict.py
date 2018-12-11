import collections
import os

from paths import raw_dir
from singleton import Singleton

_pinyin_path = os.path.join(raw_dir, 'pinyin.txt')

"""获取 汉字和 读音，音调的对应关系，  可以比较 两个汉字的音调是否相反， 读音是否相同"""


def _get_vowel(pinyin):
    i = len(pinyin) - 1
    while i >= 0 and pinyin[i] in ['A', 'E', 'I', 'O', 'U', 'V']:
        i -= 1
    return pinyin[i + 1:]


def _get_rhyme(pinyin):
    vowel = _get_vowel(pinyin)
    if vowel in ['A', 'IA', 'UA']:
        return 1
    elif vowel in ['O', 'E', 'UO']:
        return 2
    elif vowel in ['IE', 'VE']:
        return 3
    elif vowel in ['AI', 'UAI']:
        return 4
    elif vowel in ['EI', 'UI']:
        return 5
    elif vowel in ['AO', 'IAO']:
        return 6
    elif vowel in ['OU', 'IU']:
        return 7
    elif vowel in ['AN', 'IAN', 'UAN', 'VAN']:
        return 8
    elif vowel in ['EN', 'IN', 'UN', 'VN']:
        return 9
    elif vowel in ['ANG', 'IANG', 'UANG']:
        return 10
    elif vowel in ['ENG', 'ING']:
        return 11
    elif vowel in ['ONG', 'IONG']:
        return 12
    elif (vowel == 'I' and not pinyin[0] in ['Z', 'C', 'S', 'R']) or vowel == 'V':
        return 13
    elif vowel == 'I':
        return 14
    elif vowel == 'U':
        return 15
    return 0


class PronDict(Singleton):

    def __init__(self):
        self._pron_dict = collections.defaultdict(list)
        with open(_pinyin_path, 'r') as fr:
            for line in fr:
                parts = line.split()
                ch = chr(int(parts[0], 16))
                for tone in parts[1:]:
                    self._pron_dict[ch].append((tone[:-1], int(tone[-1])))
            # for k,v in self._pron_dict.items():
            #     print(k,v)

    def co_rhyme(self, a, b):
        """ Return True if two pinyins may have the same rhyme. """
        pron_a = self._pron_dict[a]
        pron_b = self._pron_dict[b]
        rhyme_a = map(lambda x: _get_rhyme(x[0]), pron_a)
        rhyme_b = map(lambda x: _get_rhyme(x[0]), pron_b)
        if set(rhyme_a).intersection(rhyme_b):
            return True
        else:
            return False

    def counter_tone(self, a, b):
        """ Return True if two pinyins may have opposite tones. """
        lambda_tone = lambda x: x[1] == 1 or x[1] == 2
        # tmp_a=self._pron_dict[a]
        # tmp_b=self._pron_dict[b]
        tone_a = list(map(lambda_tone, self._pron_dict[a]))
        tone_b = list(map(lambda_tone, self._pron_dict[b]))
        for tone_a_iter in tone_a:
            if (not tone_a_iter) in tone_b:
                return True
        else:
            return False

    def __iter__(self):
        return iter(self._pron_dict)

    def __getitem__(self, ch):
        return self._pron_dict[ch]


if __name__ == '__main__':
    pron_dict = PronDict()
    assert pron_dict.co_rhyme('生', '情')
    assert not pron_dict.co_rhyme('蛤', '人')
    r = pron_dict['平']
    assert pron_dict.counter_tone('平', '仄')
    assert not pron_dict.counter_tone('起', '弃')
    cnt = 0
    for ch in pron_dict:
        print(ch + ": " + str(pron_dict[ch]))
        cnt += 1
        if cnt > 20:
            break
