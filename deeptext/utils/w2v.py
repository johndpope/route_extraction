#coding=utf-8
import os
from gensim.models import word2vec

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                words = line.split()
                yield words

def fit(sentences,model_name,num_workers=5,num_features=50,min_word_count=40,context=4,downsampling=1e-3):
    # load sentence from the director
    # sentences = MySentences(data_path)  # a memory-friendly iterator
    print 'train word2vec ...'
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    print 'save model...'
    model.save(model_name)
    print 'finish.'
    return model
