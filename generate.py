import os

import tensorflow as tf
from tensorflow.contrib import seq2seq

from char2vec import Char2Vec
from char_dict import CharDict
from paths import save_dir
from singleton import Singleton
from utils import CHAR_VEC_DIM
from utils import _BATCH_SIZE, _NUM_UNITS

_model_path = os.path.join(save_dir, 'model')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(Singleton):

    def __init__(self):
        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self.l2_loss = tf.constant(0.0, dtype=tf.float32)
        self._build_graph()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self.trained = False

    def _build_keyword_encoder(self):
        """ Encode keyword into a vector."""
        self.keywords = tf.placeholder(dtype=tf.float32, shape=[_BATCH_SIZE, None, CHAR_VEC_DIM], name='keywords')
        self.length_keywords = tf.placeholder(dtype=tf.int32, shape=[_BATCH_SIZE], name='length_keywords')

        cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2)
        cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.keywords,
                                                          sequence_length=self.length_keywords,
                                                          dtype=tf.float32, time_major=False, scope='keywords_bi')
        states_concat = tf.concat(states, axis=1)
        # print('states_concat.state=>',states_concat.shape)
        self.states_keywords = states_concat
        return states_concat

    def _build_context_encoder(self):
        """ Encode context into a list of vectors. """
        self.context = tf.placeholder(dtype=tf.float32, shape=[_BATCH_SIZE, None, CHAR_VEC_DIM], name='context')
        self.context_length = tf.placeholder(dtype=tf.int32, shape=[_BATCH_SIZE], name='context_length')
        # self.context_length=self.context_length
        cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2)
        cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.context,
                                                          sequence_length=self.context_length,
                                                          dtype=tf.float32, time_major=False, scope='context_bi')
        outputs_concat = tf.concat(outputs, axis=2)
        self.encoder_outputs = outputs_concat
        return outputs_concat

    def _build_decoder(self):
        """ Decode keyword and context into a sequence of vectors. """
        self.sequence_decoder = tf.placeholder(dtype=tf.float32, shape=[_BATCH_SIZE, None, CHAR_VEC_DIM],
                                               name='context')
        self.length_decoder = tf.placeholder(dtype=tf.int32, shape=[_BATCH_SIZE], name='length_keywords')
        attention = seq2seq.BahdanauAttention(_NUM_UNITS, memory=self.encoder_outputs,
                                              memory_sequence_length=self.context_length, name="BahdanauAttention")
        cell_attention = tf.contrib.rnn.GRUCell(_NUM_UNITS)
        attention_wrapper = seq2seq.AttentionWrapper(cell_attention, attention)

        self.initial_decode_state = attention_wrapper.zero_state(_BATCH_SIZE, dtype=tf.float32).clone(
            cell_state=self.states_keywords)

        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(attention_wrapper, self.sequence_decoder,
                                                                           sequence_length=self.length_decoder,
                                                                           initial_state=self.initial_decode_state,
                                                                           dtype=tf.float32, time_major=False)
        # self.length_decoder=self.length_decoder

    def _build_projector(self):
        """ Project decoder_outputs into character space. """
        initial_value = tf.truncated_normal(shape=[_NUM_UNITS, len(self.char_dict)], mean=0, stddev=0.1,
                                            dtype=tf.float32)
        softmax_w = tf.Variable(initial_value=initial_value, trainable=True, name='softmax_w', dtype=tf.float32)
        initial_value_b = tf.truncated_normal(shape=[len(self.char_dict)], mean=0, stddev=0.1, dtype=tf.float32)
        softmax_b = tf.Variable(initial_value=initial_value_b, trainable=True, name='softmax_b', dtype=tf.float32)
        self.l2_loss += tf.nn.l2_loss(softmax_w)
        self.l2_loss += tf.nn.l2_loss(softmax_b)
        x = self._reshape_decoder_outputs()
        # softmax_result=tf.nn.xw_plus_b(x, softmax_w, softmax_b)
        softmax_result = tf.nn.bias_add(tf.matmul(x, softmax_w), softmax_b)
        self.logits = softmax_result
        self.print_logits = tf.nn.softmax(softmax_result)
        return softmax_result

    def _reshape_decoder_outputs(self):
        """ Reshape decoder_outputs into shape [?, _NUM_UNITS]. """
        i = tf.constant(0, dtype=tf.int32)
        # v=tf.Variable(initial_value=tf.constant(0,dtype=tf.float32,shape=[1,_NUM_UNITS]),trainable=True,dtype=tf.float32)
        v = tf.zeros(shape=[0, _NUM_UNITS], dtype=tf.float32)

        def accumulate(idx, vars):
            iterow = tf.slice(self.decoder_outputs, [idx, 0, 0], [1, self.length_decoder[idx], _NUM_UNITS])
            iterow_squeeze = tf.squeeze(iterow, axis=0)
            vars_new = tf.concat([vars, iterow_squeeze], 0)
            return tf.add(idx, 1), vars_new

        i_last, v_last = tf.while_loop(lambda i, v: i < _BATCH_SIZE, accumulate, [i, v],
                                       shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, _NUM_UNITS])])
        return v_last

    def _build_optimizer(self):
        """ Define cross-entropy loss and minimize it. """
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None])
        targets_onehot = tf.one_hot(self.targets, len(self.char_dict))
        self.print_onehot = targets_onehot
        loss = tf.losses.softmax_cross_entropy(onehot_labels=targets_onehot, logits=self.logits)
        # loss=tf.nn.softmax_cross_entropy_with_logits(labels=targets_onehot,logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        lr = tf.clip_by_value(tf.multiply(1.6e-5, tf.pow(2.1, self.loss)), 0.0002, 0.02, name=None)
        self.lr = lr

        variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, variables)
        grad_vars, _ = tf.clip_by_global_norm(gradients, 5.0)
        adamoptimizer = tf.train.AdamOptimizer(lr)
        self.train_op = adamoptimizer.apply_gradients(list(zip(grad_vars, variables)))

        # self.train_op=tf.train.AdamOptimizer(lr).minimize(self.loss+self.l2_loss)

    def _build_graph(self):
        self._build_keyword_encoder()
        self._build_context_encoder()
        self._build_decoder()
        self._build_projector()
        self._build_optimizer()

    def initialize_session(self, session, saver):
        print(_model_path)
        # checkpointstate=tf.train.get_checkpoint_state(_model_path)
        checkpointstate = tf.train.get_checkpoint_state(os.path.dirname(_model_path))
        if checkpointstate:
            print('restore')
            modelpath = checkpointstate.model_checkpoint_path
            saver.restore(session, modelpath)
            self.trained = True
        else:
            print('initializer once')
            group_variable = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            session.run(group_variable)
        return self.trained


# For testing purpose.
if __name__ == '__main__':
    generator = Generator()
    flag = 'train'
    if flag == 'train':
        generator.train(1000)
    else:
        keywords = ['四时', '变', '雪', '新']
        poem = generator.generate(keywords)
        for sentence in poem:
            print(sentence)
