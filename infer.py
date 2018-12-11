#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from char_dict import end_of_sentence, start_of_sentence
from data_utils import FuncUtils
from generate import Generator
from paths import save_dir
from pron_dict import PronDict
from utils import NUM_OF_SENTENCES
from utils import _BATCH_SIZE

_model_path = os.path.join(save_dir, 'model')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _gen_prob_list(funcutils,probs, context, pron_dict):
    prob_list = probs.tolist()[0]
    prob_list[0] = 0
    prob_list[-1] = 0
    idx = len(context)
    used_chars = set(ch for ch in context)
    for i in range(1, len(prob_list) - 1):
        ch = funcutils.char_dict.id2char(i)
        # Penalize used characters.
        if ch in used_chars:
            prob_list[i] *= 0.6
        # Penalize rhyming violations.
        if (idx == 15 or idx == 31) and \
                not pron_dict.co_rhyme(ch, context[7]):
            prob_list[i] *= 0.2
        # Penalize tonal violations.
        if idx > 2 and 2 == idx % 8 and \
                not pron_dict.counter_tone(context[2], ch):
            prob_list[i] *= 0.4
        if (4 == idx % 8 or 6 == idx % 8) and \
                not pron_dict.counter_tone(context[idx - 2], ch):
            prob_list[i] *= 0.4
    return prob_list


def generate(funcutils,generator,keywords):

    assert NUM_OF_SENTENCES == len(keywords)
    pron_dict = PronDict()
    saver = tf.train.Saver()
    context = start_of_sentence()
    with tf.Session() as session:
        flag_trained = generator.initialize_session(session, saver)
        if not flag_trained:
            print("Please train the model first! (./train.py -g)")
            sys.exit(1)
        for keyword in keywords:
            ##为何 [keyword*_batch_size?  keyword_data 都是一样的
            keyword_data, keyword_length = funcutils.fill_np_matrix(
                [keyword] * _BATCH_SIZE)
            context_data, context_length = funcutils.fill_np_matrix(
                [context] * _BATCH_SIZE)
            char = start_of_sentence()
            for _ in range(7):
                # print('char==>', char)
                decoder_input, decoder_input_length = \
                    funcutils.fill_np_matrix([char])
                encoder_feed_dict = {
                    generator.keywords: keyword_data,
                    generator.length_keywords: keyword_length,
                    generator.context: context_data,
                    generator.context_length: context_length,
                    generator.sequence_decoder: decoder_input,
                    generator.length_decoder: decoder_input_length
                }
                if char == start_of_sentence():
                    pass
                else:
                    encoder_feed_dict[generator.initial_decode_state] = state
                # do = session.run([self.decoder_outputs], feed_dict=encoder_feed_dict)
                # print(do)
                # print(do[0].shape)
                probs, state = session.run(
                    [generator.logits, generator.decoder_final_state],
                    feed_dict=encoder_feed_dict)
                prob_list = _gen_prob_list(funcutils,probs, context, pron_dict)
                id_probmax = np.argmax(prob_list, axis=0)
                char = funcutils.char_dict.id2char(id_probmax)
                # prob_sums = np.cumsum(prob_list)
                # rand_val = prob_sums[-1] * random()
                # for i, prob_sum in enumerate(prob_sums):
                #     if rand_val < prob_sum:
                #         char = self.char_dict.int2char(i)
                #         break
                context += char
            context += end_of_sentence()
    return context[1:].split(end_of_sentence())


def generate_control():
    funcutils = FuncUtils()
    generator = Generator()

    control = input('请输入【once,while】 once:只执行一次测试， while：循环执行测试')

    if control == 'once':
        keywords_str = input('输入keywords，4个，通过空格区分')
        keywords = keywords_str.split(' ')
        generate(funcutils,generator,keywords)
    elif control == 'while':
        while True:
            keywords_str = input('输入keywords，4个，通过空格区分')
            keywords = keywords_str.split(' ')
            poems_generate=generate(funcutils,generator,keywords)
            print('老夫为你作诗一首，听好了\n')
            for poem in poems_generate:
                print(poem)
    else:
        print('输入不对，【once,while】  只允许这几种')
        sys.exit('退出')


# For testing purpose.
if __name__ == '__main__':
    generate_control()
