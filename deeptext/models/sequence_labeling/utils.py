from deeptext.utils.csv import read_csv
import os
from gensim.models import word2vec

def read_data(data_path):
    rows = read_csv(data_path)

    row_num = 0

    tokens = []
    labels = []
    for row in rows:
        if row_num % 2 == 0:
            tokens.append(row)
        else:
            labels.append(row)
        row_num += 1

    return tokens, labels        

def read_data_fund(data_path):
    tokens = []
    labels = []
    i = 0
    with open(data_path) as f:
        for line in f:
            line = line.decode('utf-8').strip().split(' ')
            if i % 2 == 0:
                tokens.append(line)
            else:
                if not (u'O' in line):
                    print i
                labels.append(line)
            i += 1
    return tokens, labels



class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                words = line.split()
                yield words

def fit(sentences,model_name,num_workers=5,num_features=50,min_word_count=4,context=4,downsampling=1e-3):
    # load sentence from the director
    # sentences = MySentences(data_path)  # a memory-friendly iterator
    print 'train word2vec ...'
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    print 'save model...'
    model.save(model_name)
    print 'finish.'
    return model