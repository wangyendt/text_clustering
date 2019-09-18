# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm


import codecs

import jieba.analyse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE

WORD2VEC_SIZE = 1000
NUM_CLUSTERS = 10
USE_ELBOW_TO_FIND_BEST_NUM_CLUSTERS = False

if __name__ == '__main__':
    pd.set_option('max_colwidth', 1000)
    with open('hefei.txt', encoding='utf-8') as f:
        rows = f.readlines()

    stopwords = [line.strip() for line in codecs.open('stopped_words.txt', 'r', 'utf-8').readlines()]
    jieba.analyse.set_stop_words('stopped_words.txt')
    corpus = []
    for content in rows:
        words = jieba.cut(content)
        split_str = ''
        for word in words:
            if word not in stopwords:
                split_str += word + ' '
        corpus.append(split_str)
    # words = jieba.analyse.textrank(content, topK=30, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(word)))
    tfidf_weight = tfidf.toarray()
    kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(tfidf_weight)
    print(kmeans.cluster_centers_)
    # sentence_and_label = [(i,l) for i,l in enumerate]
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
