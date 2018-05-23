#-*- coding:utf-8 -*-

import os
import logging
import time
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn as learn

from scipy.stats import entropy

from deeptext.models.base import Base
import deeptext.utils.serialization
import deeptext.models.constants as constants

from utils import read_data_fund as read_data

reload(sys)  # reload 才能调用 setdefaultencoding 方法
sys.setdefaultencoding('utf-8')  # 设置 'utf-8'

class BiCrfSequenceLabeling(Base):
    def __init__(self, params):
        super(BiCrfSequenceLabeling, self).__init__(params)
        self.sess = tf.Session()
        model_dir = self.params[constants.PARAM_KEY_MODEL_DIR]
        err_dir = self.params[constants.PARAM_KEY_MODEL_ERR_DIR]

        token_vocab_path = os.path.join(model_dir, constants.FILENAME_TOKEN_VOCAB)
        if os.path.isfile(token_vocab_path):
            logging.info("loading token vocabulary ...")
            self.token_vocab = deeptext.utils.serialization.restore(token_vocab_path)
            logging.info("token vocabulary size = %d", len(self.token_vocab.vocabulary_))
        else:
            self.token_vocab = None

        label_vocab_path = os.path.join(model_dir, constants.FILENAME_LABEL_VOCAB)
        if os.path.isfile(label_vocab_path):
            logging.info("loading label vocabulary ...")
            self.label_vocab = deeptext.utils.serialization.restore(label_vocab_path)
            logging.info("label vocabulary size = %d", len(self.label_vocab.vocabulary_))
            logging.info(self.label_vocab.vocabulary_)
        else:
            self.label_vocab = None

    def __del__(self):
        self.sess.close()

    def preprocess(self, training_data_path):

        def tokenizer(iterator):
            for value in iterator:
                yield value

        tokens, labels = read_data(training_data_path)
        logging.info(tokens[0])
        logging.info(labels[0])
        model_dir = self.params[constants.PARAM_KEY_MODEL_DIR]

        if self.token_vocab is None:
            logging.info("generating token vocabulary ...")
            self.token_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
            self.token_vocab.fit(tokens)
            logging.info("token vocabulary size = %d", len(self.token_vocab.vocabulary_))

            token_vocab_path = os.path.join(model_dir, constants.FILENAME_TOKEN_VOCAB)
            deeptext.utils.serialization.save(self.token_vocab, token_vocab_path)

        self.token_ids = self.preprocess_token_transform(tokens)
        self.params[constants.PARAM_KEY_TOKEN_VOCAB_SIZE] = len(self.token_vocab.vocabulary_)

        if self.label_vocab is None:
            logging.info("generating label vocabulary ...")
            self.label_vocab = learn.preprocessing.VocabularyProcessor(
                max_document_length=self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN],
                tokenizer_fn=tokenizer)
            self.label_vocab.fit(labels)
            logging.info("label vocabulary size = %d", len(self.label_vocab.vocabulary_))

            label_vocab_path = os.path.join(model_dir, constants.FILENAME_LABEL_VOCAB)
            deeptext.utils.serialization.save(self.label_vocab, label_vocab_path)

        self.label_ids = self.preprocess_label_transform(labels)
        self.params[constants.PARAM_KEY_LABEL_VOCAB_SIZE] = len(self.label_vocab.vocabulary_)

        self.tensor_tokens = tf.placeholder_with_default(self.token_ids, name=constants.TENSOR_NAME_TOKENS, shape=[None,
                                                                                                                   self.params[
                                                                                                                       constants.PARAM_KEY_MAX_DOCUMENT_LEN]])
        self.tensor_labels = tf.placeholder_with_default(self.label_ids, name=constants.TENSOR_NAME_LABELS, shape=[None,
                                                                                                                   self.params[
                                                                                                                       constants.PARAM_KEY_MAX_DOCUMENT_LEN]])

        self.build_model(self.tensor_tokens, self.tensor_labels)

    def preprocess_token_transform(self, tokens):
        token_ids = self.token_vocab.transform(tokens)
        return np.array(list(token_ids))

    def preprocess_label_transform(self, labels):
        label_ids = self.label_vocab.transform(labels)
        return np.array(list(label_ids))

    def build_model(self, x, y):

        TOKEN_VOCAB_SIZE = self.params[constants.PARAM_KEY_TOKEN_VOCAB_SIZE]
        LABEL_VOCAB_SIZE = self.params[constants.PARAM_KEY_LABEL_VOCAB_SIZE]
        MAX_DOCUMENT_LEN = self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]
        EMBEDDING_SIZE = self.params[constants.PARAM_KEY_EMBEDDING_SIZE]
        DROPOUT_PROB = self.params[constants.PARAM_KEY_DROPOUT_PROB]

        word_vectors = tf.contrib.layers.embed_sequence(
             x,vocab_size=TOKEN_VOCAB_SIZE, embed_dim=EMBEDDING_SIZE, scope='words')

        fw_cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=DROPOUT_PROB)
        bw_cell = tf.contrib.rnn.LSTMCell(EMBEDDING_SIZE)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=DROPOUT_PROB)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_vectors, dtype=tf.float32)
        output = tf.concat(outputs, 2)
        output = tf.reshape(output, [-1, 2 * EMBEDDING_SIZE])

        logits = tf.contrib.layers.fully_connected(output, LABEL_VOCAB_SIZE)
        self.logits_tensor = logits = tf.reshape(logits, [-1, MAX_DOCUMENT_LEN, LABEL_VOCAB_SIZE],
                                                 name=constants.TENSOR_NAME_LOGITS)
        # targets = tf.one_hot(y, LABEL_VOCAB_SIZE, 1, 0)
        self.sequence_lengths = tf.count_nonzero(x, axis=1)
        log_likelihood, self.tensor_transition_params = tf.contrib.crf.crf_log_likelihood(logits, tf.cast(y, tf.int32),
                                                                                          self.sequence_lengths)

        loss = tf.reduce_mean(-log_likelihood)
        self.tensor_loss = tf.identity(loss, name=constants.TENSOR_NAME_LOSS)
        self.summ_training_loss = tf.summary.scalar("training_loss", self.tensor_loss)
        self.summ_validation_loss = tf.summary.scalar("validation_loss", self.tensor_loss)

        # Create a training op.
        self.tensor_optimizer = tf.contrib.layers.optimize_loss(
            self.tensor_loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            learning_rate=0.02)

        # self.tensor_prediction = []
        # for logit in logits:
        #     prediction, srore = tf.contrib.crf.viterbi_decode(logit, self.transition_params)
        #     self.tensor_prediction.append(prediction)
        # self.tensor_prediction, viterbi_score = tf.contrib.crf.viterbi_decode(logits, transition_params)
        #  self.tensor_prediction = tf.argmax(logits, 2, name=constants.TENSOR_NAME_PREDICTION)

        self.summ = tf.summary.merge_all()

    def fit(self, steps, batch_size, training_data_path, validation_data_path=None):

        self.preprocess(training_data_path)

        validation_token_ids = None
        validation_label_ids = None
        if validation_data_path is not None:
            tokens, labels = read_data(validation_data_path)
            validation_token_ids = self.preprocess_token_transform(tokens)
            validation_label_ids = self.preprocess_label_transform(labels)

        self.sess.run(tf.global_variables_initializer())
        self.restore(self.sess)

        logdir = constants.SUMMARY_FILE_PATH + '/' + time.strftime("%Y-%m-%d:%H:%M:%S", time.localtime())
        writer = tf.summary.FileWriter(logdir)
        writer.add_graph(self.sess.graph)
        logging.info("logdir: " + logdir)

        for i in xrange(steps):
            curr_row_ids = np.random.choice(self.token_ids.shape[0], batch_size)
            curr_token_ids = self.token_ids[curr_row_ids]
            curr_label_ids = self.label_ids[curr_row_ids]

            self.transition_matrix, _ = self.sess.run([self.tensor_transition_params, self.tensor_optimizer],
                                                      feed_dict={self.tensor_tokens: curr_token_ids,
                                                                 self.tensor_labels: curr_label_ids})

            if (i + 1) % 50 == 0:
                c, s = self.sess.run([self.tensor_loss, self.summ_training_loss],
                                     feed_dict={self.tensor_tokens: curr_token_ids, self.tensor_labels: curr_label_ids})
                writer.add_summary(s, i + 1)
                logging.info("step: %d, training loss: %.2f", i + 1, c)

                if validation_data_path is not None:
                    c, s = self.sess.run([self.tensor_loss, self.summ_validation_loss],
                                         feed_dict={self.tensor_tokens: validation_token_ids,
                                                    self.tensor_labels: validation_label_ids})
                    writer.add_summary(s, i + 1)
                    logging.info("step: %d, validation loss: %.2f", i + 1, c)

                self.save(self.sess)

    def pre_restore(self):
        self.params[constants.PARAM_KEY_TOKEN_VOCAB_SIZE] = len(self.token_vocab.vocabulary_)
        self.params[constants.PARAM_KEY_LABEL_VOCAB_SIZE] = len(self.label_vocab.vocabulary_)
        self.token_ids = self.preprocess_token_transform([[u'^', u'你', u'有', u'娘', u'子', u'吗', u'$']])
        self.label_ids = self.preprocess_label_transform([['','','']])
        self.tensor_tokens = tf.placeholder_with_default(self.token_ids,
                                                         name=constants.TENSOR_NAME_TOKENS,
                                                         shape=[None,
                                                                self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]])
        self.tensor_labels = tf.placeholder_with_default(self.label_ids,
                                                         name=constants.TENSOR_NAME_LABELS,
                                                         shape=[None,
                                                                self.params[constants.PARAM_KEY_MAX_DOCUMENT_LEN]])
        self.build_model(self.tensor_tokens,self.tensor_labels)

    def predict(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        tf_unary_scores, tf_sequence_lengths,self.transition_matrix = self.sess.run(
            [self.logits_tensor, self.sequence_lengths,self.tensor_transition_params],
            feed_dict={self.tensor_tokens: tokens_transform}
        )
        labels = []
        for tf_unary_score_, sentence_length_ in zip(tf_unary_scores, tf_sequence_lengths):
            tf_unary_score_ = tf_unary_score_[:sentence_length_]
            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_score_, self.transition_matrix)
            if True:
                # print viterbi_sequence
                labels_str = ''.join(str(i) for i in viterbi_sequence)
                # print labels_str
                try:
                    i = labels_str.index('121')
                    viterbi_sequence[i+1] = 4
                except:
                    pass
                try:
                    i = labels_str.index('131')
                    viterbi_sequence[i + 1] = 5
                except:
                    pass
            try:
                labels.append(list(self.label_vocab.reverse([viterbi_sequence])))
            except:
                labels.append(list(u'O' for i in range(tf_sequence_lengths)))
        return labels

    def logits(self, tokens):
        tokens_transform = self.preprocess_token_transform(tokens)
        logits = self.sess.run(self.logits_tensor, feed_dict={
            self.tensor_tokens: tokens_transform
        })
        entropy_list = []
        for i in xrange(len(tokens)):
            max_entropy = 0
            for j in xrange(len(tokens[i])):
                cur_entropy = entropy(logits[i][j])
                max_entropy = max(cur_entropy, max_entropy)
            entropy_list.append(max_entropy)
        return entropy_list

    def evaluate(self, testing_data_path):
        tokens, labels = read_data(testing_data_path)

        tokens_transform = self.preprocess_token_transform(tokens)
        labels_transform = self.preprocess_label_transform(labels)

        tf_unary_scores, tf_transition_params, tf_sequence_lengths = self.sess.run(
            [self.logits_tensor, self.tensor_transition_params, self.sequence_lengths],
            feed_dict={
                self.tensor_tokens: tokens_transform
            })

        label_corre_cnt = 0
        label_total_cnt = 0
        sentence_corre_cnt = 0
        sentence_total_cnt = len(labels)

        err_dir = self.params[constants.PARAM_KEY_MODEL_ERR_DIR]
        err_file = err_dir+'_new_one_entity_oeo2oso.txt'
        err_file = open(err_file,'w')

        # true_dir = self.params[constants.PARAM_KEY_MODEL_ERR_DIR]
        #  true_file = '/Users/liuxiaoan/Downloads/fund_raw_data_new/true_data_new.txt'
        #  true_file = open(true_file, 'w')

        #  err_entity = self.read_errdata('/Users/liuxiaoan/Downloads/raw_data/data/music_valid_data_new_entity.txt')
        #  # print len(err_entity)
        for tf_unary_scores_, y_, sequence_length_, data_index in zip(tf_unary_scores, labels_transform, tf_sequence_lengths, range(len(labels_transform))):
            # Remove padding from the scores and tag sequence.
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]

            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, tf_transition_params)
            # Evaluate accuracy.
            # change o e o to o s o
            if True:
                labels_str = ''.join(str(i) for i in viterbi_sequence)
                try:
                    i = labels_str.index('121')
                    viterbi_sequence[i+1] = 4
                except:
                    pass
                try:
                    i = labels_str.index('131')
                    viterbi_sequence[i + 1] = 5
                except:
                    pass
            sentence_corre = True
            label_total_cnt += sequence_length_
            sencence_label_corre_ = np.sum(np.equal(viterbi_sequence, y_))
            label_corre_cnt += sencence_label_corre_

            if sencence_label_corre_ == sequence_length_:
                sentence_corre_cnt += 1
                # true_file.write(" ".join(tokens[data_index]) + '\n')
                # true_file.write(" ".join(labels[data_index]) + '\n')
                # true_file.write(" ".join(self.predict([tokens[data_index]])[0]).strip() + '\n')
                # # err_file.write(err_entity[data_index] + '\n')
            else:
                if True:
                    try:
                        err_file.write(",".join(tokens[data_index])+ '\n')
                        err_file.write(",".join(labels[data_index]) + '\n')
                        err_file.write(",".join(self.predict([tokens[data_index]])[0]).strip().replace(' ',',') + '\n')
                        err_file.write(err_entity[data_index]+'\n')
                    except:
                        continue
        err_file.close()
       #  true_file.close()


        logging.info("total label count: %d, label accuracy: %.2f", label_total_cnt,
                     1.0 * label_corre_cnt / label_total_cnt)
        logging.info("total sentence count: %d, sentence accuracy: %.2f", sentence_total_cnt,
                     1.0 * sentence_corre_cnt / sentence_total_cnt)

    def read_errdata(self,filename):
        entity = []
        with open(filename) as f:
            for l in f:
                entity.append(l.strip())
        return entity
