# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm

import codecs
from copy import copy

import jieba.analyse
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans

if __name__ == '__main__':
    pd.set_option('max_colwidth', 1000)
    with open('mao.txt') as f:
        rows = f.readlines()[0]
    segments = []
    stopwords = [line.strip() for line in codecs.open('stopped.txt', 'r', 'utf-8').readlines()]
    jieba.analyse.set_stop_words('stopped.txt')
    content = copy(rows)
    words = jieba.cut(content)
    # words = jieba.analyse.textrank(content, topK=30, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    split_str = ''
    for word in words:
        if word not in stopwords:
            segments.append({'word': word, 'count': 1})
            split_str += word + ' '

    with open('split_str.txt', 'w', encoding='utf-8') as f:
        f.write(split_str)
    model = Word2Vec(LineSentence('split_str.txt'), size=300, window=5, min_count=3, workers=4)
    model.wv.save_word2vec_format('word_model.txt', binary=False)
    words = model.wv.vocab.keys()

    # 获取词对于的词向量
    word_vector = [model[w] for w in words]

    # 聚类
    clf = KMeans(n_clusters=100)
    classes = clf.fit_predict(word_vector)
    for c, w in sorted(zip(classes, words), key=lambda t: t[0]):
        print(f'{c}: {w}')
