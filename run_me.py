# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm

import codecs
import collections
from copy import copy

import jieba.analyse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans

WORD2VEC_SIZE = 1000
MAX_NUM_WORDS = 20
NUM_CLUSTERS = 5
MAX_NUM_CLUSTERS = min(MAX_NUM_WORDS, 20)
USE_ELBOW_TO_FIND_BEST_NUM_CLUSTERS = False

if __name__ == '__main__':
    pd.set_option('max_colwidth', 1000)
    with open('mao.txt') as f:
        rows = f.readlines()[0]

    stopwords = [line.strip() for line in codecs.open('stopped.txt', 'r', 'utf-8').readlines()]
    jieba.analyse.set_stop_words('stopped.txt')
    content = copy(rows)
    words = jieba.cut(content)
    # words = jieba.analyse.textrank(content, topK=30, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

    counter_dict = collections.Counter([w for w in words if w not in stopwords])
    num_all_words = len(counter_dict)
    most_freq_words = counter_dict.most_common(MAX_NUM_WORDS)

    split_str = ''
    max_num_str = 0
    for word, freq in most_freq_words:
        split_str += word + ' '
        max_num_str += 1
        if max_num_str > MAX_NUM_WORDS:
            break
    with open('split_str.txt', 'w', encoding='utf-8') as f:
        f.write(split_str)

    model = Word2Vec(LineSentence('split_str.txt'), size=WORD2VEC_SIZE, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format('word_model.txt', binary=False)
    words = model.wv.vocab.keys()

    # 获取词对于的词向量
    word_vector = [model[w] for w in words]

    # 聚类
    classes = dict()
    if USE_ELBOW_TO_FIND_BEST_NUM_CLUSTERS:
        inertias = []
        for i in range(1, MAX_NUM_CLUSTERS + 1):
            clf = KMeans(n_clusters=i)
            classes[i] = clf.fit_predict(word_vector)
            inertias.append(clf.inertia_)
        best = np.argmax(np.diff(np.log(np.abs(np.diff(inertias))))) + 1  # find elbow
    else:
        clf = KMeans(n_clusters=NUM_CLUSTERS)
        classes[NUM_CLUSTERS] = clf.fit_predict(word_vector)
        best = NUM_CLUSTERS

    df = pd.DataFrame(columns=['class', 'word', 'freq'])
    row = 0
    for c, w in sorted(zip(classes[best], words), key=lambda t: t[0]):
        df.loc[row, 'class'] = c
        df.loc[row, 'word'] = w
        df.loc[row, 'freq'] = counter_dict[w] / num_all_words
        row += 1
        print(f'{c}: {w}, {counter_dict[w]}')
    df.to_excel('result.xlsx')
