#-*- coding:utf-8 -*-

import logging
import re
import time
import os
import sys
import json

# print sys.path
sys.path.append('/home/lxa/Route_Extraction')
FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

from deeptext.models.sequence_labeling.sequence_labeling import SequenceLabeling
from deeptext.models.sequence_labeling.biLSTM_sequence_labeling import BidirectionalSequenceLabeling
from deeptext.models.sequence_labeling.biLSTM_crf_sequence_labeling import BiCrfSequenceLabeling
from deeptext.models.sequence_labeling.word2vec_biLSTM_crf_sequence_labeling import Word2VecBiCrfSequenceLabeling
from deeptext.models.sequence_labeling.self_word2vec_biLSTM_crf_sequence_labeling import SeWord2VecBiCrfSequenceLabeling

from text_classification.classifier_new import Classifier

params = {}
params["max_document_len"] = 25
params["embedding_size"] = 100
params["dropout_prob"] = 0.5
params["model_name"] = 'travel_model_new'
params["model_dir"] = '../model_weight/ner_model_v_test'
params["err_dir"] = '../err/ner_err'

model = BiCrfSequenceLabeling(params)

# #
# model.fit(
#         steps=700,
#         batch_size=256,
#         training_data_path='../data/NER_train_data.txt',
#         validation_data_path='../data/NER_test_data.txt'
#         )

# model.pre_restore()
# model.restore(model.sess)

# clf = Classifier('../text_classification/user_dict.txt')
# clf.preload('../text_classification/models/sentence_classify_svc')

#start = time.time()
#
# for i in range(12):
#     idx = ''
#     if i < 10:
#         idx = '0'+ str(i)
#     else:
#         idx = str(i)
#
#     model.evaluate(
#            testing_data_path='../data/NER_test_000' + idx
#            )


def get_route(filename, model, out_file):
    entitys = []
    with open(filename) as f:
        for line in f:
            l = unicode(line.strip(), 'utf-8')
            sublines = re.split(ur'[！？｡。，]', l)
            for sentence in sublines:
                sentence = '^' + sentence + '$'
                sentence_list = [list(sentence)]
                # print sentence_list
                labels = model.predict(sentence_list)
                # print labels
                labels = labels[0][0].split()
                for i, char in zip(range(len(labels)-1), sentence_list[0]):
                    if labels[i] == 'E' and labels[i - 1] == 'O':
                        new_entity = ''
                        new_entity = new_entity + char
                    elif labels[i] == 'E' and labels[i + 1] == 'E':
                        new_entity = new_entity + char
                    elif labels[i] == 'E' and labels[i + 1] == 'O':
                        new_entity = new_entity + char
                        logging.info('sentence: %s' % (sentence))
                        logging.info('labels: %s' % ' '.join(labels))
                        if len(entitys) == 0:
                            entitys.append(new_entity)
                        if new_entity != entitys[-1]:
                            entitys.append(new_entity)
    out_f = open(out_file,'w')
    out_f.write('-'.join(entitys)+'\n')
    out_f.close()

def build_train_data(model, record_filename, route_filename, out_train_filename):
    route_entity = []
    with open(route_filename) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            line_list = line.split(u'-')
            route_entity = route_entity + line_list

    # print ' '.join(route_entity)

    out_train_file = open(out_train_filename,'a')
    subline_count = 0
    useful_count = 0
    with open(record_filename) as f:
        for line in f:
            l = unicode(line.strip(), 'utf-8')
            sublines = re.split(ur'[！？｡。，]', l)
            for sentence in sublines:
                if sentence == '':
                    continue
                sentence = '^' + sentence + '$'
                sentence_list = [list(sentence)]
                # print sentence_list
                labels = model.predict(sentence_list)
                # print labels
                labels = labels[0][0].split()
                start_index = -1
                end_index1 = -1
                if len(labels) != len(sentence_list[0]):
                    continue
                if 'E' in labels:
                    useful_count += 1
                for i, char in zip(range(len(labels)-1), sentence_list[0]):
                    if labels[i] == 'E' and labels[i - 1] == 'O':
                        new_entity = ''
                        start_index = i
                        new_entity = new_entity + char
                    elif labels[i] == 'E' and labels[i + 1] == 'E':
                        new_entity = new_entity + char
                    elif labels[i] == 'E' and labels[i + 1] == 'O':
                        new_entity = new_entity + char
                        end_index1 = i + 1
                        if new_entity not in route_entity:
                            for i in range(start_index, end_index1):
                                labels[i] = 'O'
                            start_index = -1
                            end_index1 = -1
                        # write to file
                sentence_str = u' '.join(sentence_list[0])
                label_str = u' '.join(labels)

                out_train_file.write(sentence_str + '\n')
                out_train_file.write(label_str + '\n')
                subline_count += 1
    out_train_file.close()
    return subline_count,useful_count


def build_all_data(record_dir, route_dir, out_file):
    if os.path.exists(route_dir):
        logging.info('loading article files...')

        files = os.listdir(route_dir)
        logging.info('file number: %d' % len(files))

        sentence_count = 0
        article_count = 0
        use_cout = 0
        for tmp_file in files:
            if os.path.exists(record_dir + '/'+ tmp_file[:-4] + '.txt'):
                tmp_sentence_count,tmp_use_cout = build_train_data(model,record_dir + '/'+ tmp_file[:-4] + '.txt',
                                 route_dir+'/'+tmp_file, out_file)
                sentence_count = sentence_count + tmp_sentence_count
                use_cout = use_cout + tmp_use_cout
                article_count += 1

            logging.info('handle article number %d, sentence count : %d, useful_count : %d'
                         '' % (article_count,sentence_count,use_cout))

def test_model(model, record_dir, route_dir, out_dir):
    if os.path.exists(route_dir):
        logging.info('loading article files...')

        files = os.listdir(route_dir)
        logging.info('route_dir file number: %d' % len(files))

        file_nums = [tmp_f[:-4] for tmp_f in files]

        files = os.listdir(record_dir)
        logging.info('record_dir file number: %d' % len(files))
        count = 0
        for tmp_file in files:
            tmp_index = -1
            if re.findall(r'\d+', tmp_file):
                tmp_index = re.findall(r'\d+', tmp_file)[0]
            else:
                continue
            if tmp_index in file_nums:
                continue
            else:
                get_route(record_dir+'/'+tmp_file,model, out_dir+'/'+tmp_index+'.txt')
                count += 1
    print 'test article count: %d' % count

def get_precision(true_dir, pre_dir):
    if os.path.exists(pre_dir):
        logging.info('loading prediction files...')

        files = os.listdir(pre_dir)
        logging.info('pre_dir file number: %d' % len(files))

        right_count = 0
        all_count = 0
        pre_count = 0
        min_pre = 0.5
        for tmp_file in files:
            # pre_entitys = set()
            # true_entitys = set()
            pre_entitys = []
            true_entitys = []
            tmp_index = -1

            if re.findall(r'\d+', tmp_file):
                tmp_index = re.findall(r'\d+', tmp_file)[0]
            else:
                continue

            # read pre entitys
            with open(pre_dir+'/'+tmp_file) as f:
                for line in f:
                    l = unicode(line.strip(), 'utf-8')
                    # pre_entitys = pre_entitys | set(l.strip().split('-'))
                    pre_entitys = pre_entitys + l.strip().split('-')
                    pre_count = pre_count + len(pre_entitys)

            # read true entitys
            if os.path.exists(true_dir + '/œﬂ¬∑' + tmp_file[:-4] + '.txt'):
                with open(true_dir+'/œﬂ¬∑'+tmp_file) as f:
                    for line in f:
                        l = unicode(line.strip(), 'utf-8')
                        # true_entitys = true_entitys | set(l.strip().split('-'))
                        true_entitys = true_entitys + l.strip().split('-')
                        all_count = all_count + len(true_entitys)

            else:
                print 'not find file: %s' % true_dir + '/œﬂ¬∑' + tmp_file[:-4] + '.txt'

            # panduan
            # print ' '.join(pre_entitys)
            # print ' '.join(true_entitys)
            tmp_right = 0
            for pre_entity in pre_entitys:
                if pre_entity in true_entitys:
                    right_count += 1
                    tmp_right += 1
            if len(true_entitys) == 0:
                continue
            if min_pre < (tmp_right*1.0/len(true_entitys)):
                logging.info(true_dir+'/œﬂ¬∑' + tmp_file[:-4] + '.txt   presicion: %.2f'
                             % (tmp_right*1.0/len(true_entitys)))
                # min_pre = (tmp_right*1.0/len(true_entitys))
            # print 'handle a file : '+true_dir+'/œﬂ¬∑' + tmp_file[:-4] + '.txt'

        print 'article count: %d ; ture entity count: %d ; predict entity count %d ' \
              '; precision: %.2f' %(len(files),all_count,pre_count,right_count*1.0/all_count)
        print 'article count: %d ; ture entity count: %d ; predict entity count %d ' \
              '; recall: %.2f' % (len(files), all_count, pre_count, right_count * 1.0 / pre_count)


# ------------------for ctrip article and route-------------
def get_route_new(article, model):
    pois = []
    templates = []

    paragraphs = article.split('\n')
    for paragraph in paragraphs:
        if len(paragraph) <= 5:
            continue
        sublines = re.split(ur'[！!?？｡。；;,，]', paragraph)
        for sentence in sublines:
            if len(sentence) <= 5:
                continue
            sentence = '^' + sentence + '$'
            char_list = [list(sentence)]

            labels = model.predict(char_list)
            labels = labels[0][0].split()

            template = list(sentence)
            for i, char in zip(range(len(labels) - 1), char_list[0]):
                if labels[i] == 'E' and labels[i - 1] == 'O':
                    new_entity = ''
                    new_entity = new_entity + char
                    template[i] = ''
                elif labels[i] == 'E' and labels[i + 1] == 'E':
                    new_entity = new_entity + char
                    template[i] = ''
                elif labels[i] == 'E' and labels[i + 1] == 'O':
                    new_entity = new_entity + char
                    template[i] = '@entity'

                    # pois.append(new_entity)
                    # templates.append(''.join(template[1:-1]))

                    # pois.append(new_entity)
                    # store once if entitiy show continues
                    if len(pois) == 0:
                        pois.append(new_entity)
                        templates.append(''.join(template[1:-1]))
                    elif new_entity not in pois:
                        pois.append(new_entity)


                    # if len(pois) == 0:
                    #     pois.append(new_entity)
                    # elif new_entity != pois[-1]:
                    #     clf_result = clf.predict(''.join(template[1:-1]))
                    #     if json.loads(clf_result)['label'] == '1':
                    #         pois.append(new_entity)

    return pois,templates

def get_route_new2(article, model):
    pois = []
    pois_sentence = []
    paragraphs = article.split('\n')
    for paragraph in paragraphs:
        if len(paragraph) < 2:
            continue
        sublines = re.split(ur'[！!?？｡。；;,，]', paragraph)
        for sentence in sublines:
            if len(sentence) < 2:
                continue
            sentence = '^' + sentence + '$'
            char_list = [list(sentence)]

            labels = model.predict(char_list)
            labels = labels[0][0].split()

            for i, char in zip(range(len(labels) - 1), char_list[0]):
                if labels[i] == 'E' and labels[i - 1] == 'O':
                    new_entity = ''
                    new_entity = new_entity + char
                elif labels[i] == 'E' and labels[i + 1] == 'E':
                    new_entity = new_entity + char
                elif labels[i] == 'E' and labels[i + 1] == 'O':
                    new_entity = new_entity + char
                    pois.append(new_entity)
                    pois_sentence.append(sentence[1:-1])

    return pois,pois_sentence

def test_model_new(filename,outname, model):
    i = 0
    bar = ProgressBar()
    out_f = open(outname,'w')
    with open(filename, 'r') as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            record = json.loads(line.strip())
            predict_route,templates = get_route_new(record['article'], model)

            out_f.write('-'.join(predict_route)+'\n')
            out_f.write('-'.join(record['route']) + '\n\n')
            bar.log()
            i += 1
            if i % 100 == 0:
                logging.info('handle article number: %d' % i)

    out_f.close()

# ner route and sentence
def test_model_new2(filename,outfilename,model):
    i = 0
    bar = ProgressBar()
    out_file = open(outfilename, 'w')
    with open(filename, 'r') as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            record = json.loads(line.strip())
            predict_route,poi_sentences = get_route_new2(record['article'], model)
            record['predict_route'] = predict_route

            ner_data = []
            for poi, sentence in zip(predict_route, poi_sentences):
                ner_data.append((poi, sentence))

            record['ner_data'] = ner_data
            out_file.write(json.dumps(record)+'\n')
            bar.log()

            i += 1
            if i % 100 == 0:
                logging.info('handle article number: %d' % i)

    out_file.close()

# ner original route
def test_model_new3(filename,outfilename,model):
    i = 0
    bar = ProgressBar()
    out_f = open(outfilename, 'w')
    with open(filename, 'r') as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            record = json.loads(line.strip())
            predict_route,poi_sentences = get_route_new2(record['article'], model)

            out_f.write('-'.join(predict_route)+'\n')
            out_f.write('-'.join(record['route']) + '\n\n')
          
            bar.log()

            i += 1
            if i % 100 == 0:
                logging.info('handle article number: %d' % i)

    out_f.close()


#  get_precision('/Users/liuxiaoan/Downloads/data/travel/route',
#               '/Users/liuxiaoan/Downloads/data/travel/predict')


# test_model(model,'/Users/liuxiaoan/Downloads/data/travel/record',
#            '/Users/liuxiaoan/Downloads/data/travel/route_useful',
#            '/Users/liuxiaoan/Downloads/data/travel/predict')


# build_all_data('/Users/liuxiaoan/Downloads/data/travel/record_useful',
#                '/Users/liuxiaoan/Downloads/data/travel/route_useful',
#                '/Users/liuxiaoan/deeptext/trivel/data/travel_train_data_new.txt')

# print build_train_data(model,'/Users/liuxiaoan/Downloads/data/travel/record_useful/1.txt',
#                  '/Users/liuxiaoan/Downloads/data/travel/route_useful/1.txt',
#                  '/Users/liuxiaoan/deeptext/trivel/data/travel_train_data_new.txt')


# get_route('/Users/liuxiaoan/Downloads/data/travel/record/Œƒ’¬1.txt',
#           model,'/Users/liuxiaoan/Downloads/data/travel/predict/1.txt')



# ------------------for ctrip article and route-------------

# test_model_new('../data/test_data.json', model)
class ProgressBar:
    def __init__(self):
        self.count = 0
    def log(self):
        self.count += 1
        sys.stdout.write(' ' * (self.count + 1) + '\r')
        sys.stdout.flush()
        sys.stdout.write('.' * self.count + '\r')
        sys.stdout.flush()

if __name__ == '__main__':
    # train
    flag = 'Train'
    if flag == 'Train':
        model.fit(
                steps=700,
                batch_size=256,
                training_data_path='../data/NER_train_data_without_others_all_in_v3.txt',
                validation_data_path='../data/NER_test_data_demo.txt'
                )
    elif flag == 'Test':
        model.pre_restore()
        model.restore(model.sess)
        test_model_new3('../data/test_data_without_others_and_all_in.json','../data/predict_ner_test_v3_original.txt', model)

    elif flag == 'Evaluate':
        model.pre_restore()
        model.restore(model.sess)
        model.evaluate(
                   testing_data_path='../data/NER_test_data_demo.txt'
                   )
