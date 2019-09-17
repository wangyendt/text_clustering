# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm


import codecs
import collections
from copy import copy

import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE

WORD2VEC_SIZE = 1000
MAX_NUM_WORDS = 100000
NUM_CLUSTERS = 300
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
    segment = []
    max_num_str = 0
    for word, freq in most_freq_words:
        split_str += word + ' '
        segment.append(word)
        max_num_str += 1
        # if max_num_str > MAX_NUM_WORDS:
        #     break
    print(segment)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(segment))
    word = vectorizer.get_feature_names()
    tfidf_weight = tfidf.toarray()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(tfidf_weight)
    print(kmeans.cluster_centers_)
    for index, label in enumerate(kmeans.labels_, 1):
        print("index: {}, label: {}".format(index, label))
    print("inertia: {}".format(kmeans.inertia_))
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tfidf_weight)

    x = []
    y = []

    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=kmeans.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    exit()

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
