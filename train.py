#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os

import tensorflow as tf

from data_utils import FuncUtils
from data_utils import batch_train_data
from generate import Generator
from paths import save_dir
from utils import _BATCH_SIZE,_logwriter_dir

_model_path = os.path.join(save_dir, 'model')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# train_writer.add_summary(train_summary,step)#调用train_writer的add_summary方法将训练过程以及训练步数保存




def train(n_epochs=6):
    funcutils = FuncUtils()
    generator = Generator()

    tf.summary.scalar('accuracy', generator.loss)  # 生成准确率标量图
    tf.summary.scalar('lr', generator.lr)  # 生成准确率标量图
    merge_summary = tf.summary.merge_all()

    batch_no = 0
    saver = tf.train.Saver()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    loss_best = 1000
    with tf.Session(config=tf_config) as sess:

        train_writer = tf.summary.FileWriter(_logwriter_dir,sess.graph)#定义一个写入summary的目标文件，dir为写入文件地址

        generator.initialize_session(sess, saver)
        for epoch_iter in range(n_epochs):
            # print('epoch_iter\t{}'.format(epoch_iter))
            for keywords, contexts, sentences in batch_train_data(_BATCH_SIZE):
                # print('-4-'*5)
                if batch_no % 32 == 0:
                    # print('len(poems_set)==>',len(poems_set))
                    print_onehot, logits, lr, loss,summary_value = _train_a_batch(funcutils,generator,sess, epoch_iter, keywords, contexts, sentences,merge_summary)
                    train_writer.add_summary(summary_value,batch_no)
                    print('epoch\t{}\tloss\t{}lr\t{}'.format(epoch_iter, loss, lr))
                    # print('print_onehot==>',print_onehot)
                    # print('logits==>',logits)
                    if loss < loss_best:
                        saver.save(sess, _model_path)
                        loss_best = loss

                batch_no += 1





def _train_a_batch(funcutils,generator,session, epoch, keywords, contexts, sentences,merge_summary):
    keywords_vec, keywords_len = funcutils.fill_np_matrix(keywords)
    contexts_vec, contexts_len = funcutils.fill_np_matrix(contexts)
    # sentences_change=[start_of_sentence() + sentence[:-1] for sentence in sentences]
    sentences_change = [sentence[:-1] for sentence in sentences]
    sentences_vec, sentences_len = funcutils.fill_np_matrix(sentences_change)
    targets = funcutils.fill_targets(sentences_change)

    feed_batch = {
        generator.keywords: keywords_vec,
        generator.length_keywords: keywords_len,
        generator.context: contexts_vec,
        generator.context_length: contexts_len,
        generator.sequence_decoder: sentences_vec,
        generator.length_decoder: sentences_len,
        generator.targets: targets
    }

    print_onehot, logits, lr, loss, _,summary_value = session.run(
        [generator.print_onehot, generator.logits, generator.lr, generator.loss, generator.train_op,merge_summary],
        feed_dict=feed_batch)
    return print_onehot, logits, lr, loss,summary_value
    # print('epoch\t{}\tloss\t{}lr\t{}'.format(epoch,loss,lr))


# For testing purpose.
if __name__ == '__main__':
    train(1000)
