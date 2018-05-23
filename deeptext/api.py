#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, jsonify
from flask import request
from flask import make_response
import json
import re
import sys
sys.path.append('../')

from models.sequence_labeling.biLSTM_crf_sequence_labeling import BiCrfSequenceLabeling

app = Flask(__name__)

params = {}
params["max_document_len"] = 25
params["embedding_size"] = 100
params["dropout_prob"] = 0.5
params["model_name"] = 'travel_model_new'
params["model_dir"] = '../model_weight/ner_model_v3'
params["err_dir"] = '../err/ner_err'

model = BiCrfSequenceLabeling(params)
model.pre_restore()
model.restore(model.sess)

def get_route_new3(article, model):
    new_article = ''
    pois = []
    sentence_poi = []
    paragraphs = article.split('\n')
    for paragraph in paragraphs:
        if len(paragraph) <= 2:
            continue
        sublines = re.split(ur'[！!?？｡。；;,，]', paragraph)

        tmp_sentence = []
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
                    sentence_poi.append(sentence[1:-1])
                    if sentence[1:-1].startswith(new_entity):
                        sentence = sentence.replace(new_entity,'<span style="color:blue">'+new_entity+'</span>')
                    else:
                        sentence = sentence.replace(new_entity, '<span style="color:red">' + new_entity + '</span>')
                    # print new_entity

            tmp_sentence.append(sentence[1:-1])
        new_article = new_article + '||'.join(tmp_sentence) + '<br></br>'
    return new_article,pois,sentence_poi

@app.route('/deep_ner/', methods=['GET', 'POST'])
def ner_api():
    if request.method == 'POST':
        article = request.form['article']
        new_article,pois,sentence_poi = get_route_new3(article,model)
        # 过滤POI开头的句子
        no_start_pois = []
        for poi,sentence in zip(pois,sentence_poi):
            if sentence.startswith(poi):
                continue
            else:
                no_start_pois.append(poi)

        # 去除重复的路线
        new_pois = []
        new_pois.append(pois[0])
        j = 0
        for i in range(1,len(pois)):
            if new_pois[j] == pois[i]:
                continue
            else:
                new_pois.append(pois[i])
                j += 1

        # 取第一次出现
        first_pois = []
        for poi in no_start_pois:
            if poi not in first_pois:
                first_pois.append(poi)

        res = {"new_article": new_article, "original_pois": '-'.join(pois),
               "no_start_pois":'-'.join(no_start_pois),"del_continue_pois": '-'.join(new_pois),
               "first_pois": '-'.join(first_pois)}

        response = make_response(json.dumps(res))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
        return response
    else:
        return 'entry.html'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
