# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm

import codecs
import collections

import jieba.analyse
import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE

WORD2VEC_SIZE = 300
NUM_KEYWORDS_PER_DOC = 10
NUM_CLUSTERS = 10
KEY_WORD_TFIDF_THD = 0.28
FREQ_THD = 0.00001
USE_TFIDF = True
USE_FREQ = True


def tsne_plot(data, cls):
    tsne = TSNE(n_components=2,perplexity=20,learning_rate=100)
    decomposition_data = tsne.fit_transform(data)

    x = []
    y = []

    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=cls.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    pd.set_option('max_colwidth', 1000)
    with open('hefei.txt', encoding='utf-8') as f:
        rows = f.readlines()

    stopwords = [line.strip() for line in codecs.open('stopped_words.txt', 'r', 'utf-8').readlines()]
    jieba.analyse.set_stop_words('stopped_words.txt')
    corpus = []
    whole_words = []
    counter_dict = collections.defaultdict(int)
    num_all_words = 0
    for r in rows:
        content = r.strip(' ').replace('\r', '').replace('\n', '')
        if not content: continue
        whole_words.append([])
        words = jieba.cut(content)
        split_str = ''
        for word in words:
            if word not in stopwords:
                counter_dict[word] += 1
                num_all_words += 1
                whole_words[-1].append(word)
                split_str += word + ' '
        corpus.append(split_str)
    # words = jieba.analyse.textrank(content, topK=30, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    all_words = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(all_words)))
    tfidf_weight = tfidf.toarray()

    split_str = ''
    weight_dict = dict()
    for weight in tfidf_weight:
        loc = np.argsort(-weight)
        for i in range(NUM_KEYWORDS_PER_DOC):
            # print(all_words[loc[i]], counter_dict[all_words[loc[i]]], num_all_words)
            if USE_TFIDF:
                if weight[loc[i]] <= KEY_WORD_TFIDF_THD:
                    continue
            if USE_FREQ:
                if counter_dict[all_words[loc[i]]] / num_all_words <= FREQ_THD:
                    continue
            split_str += all_words[loc[i]] + ' '
            weight_dict[all_words[loc[i]]] = weight
        #     print(f'{i + 1}:{all_words[loc[i]], weight[loc[i]]}')
        # print('\n')
    with open('split_str.txt', 'w', encoding='utf-8') as f:
        f.write(split_str)

    model = Word2Vec(LineSentence('split_str.txt'), size=WORD2VEC_SIZE, window=5, min_count=1, workers=4)
    model.wv.save_word2vec_format('word_model.txt', binary=False)
    word_keys = model.wv.vocab.keys()
    print(len(word_keys), word_keys)

    # 获取词对于的词向量
    word_vector = [model[w] for w in word_keys]

    # 聚类
    clf = KMeans(n_clusters=NUM_CLUSTERS)
    classes = clf.fit_predict(word_vector)
    tsne_weight_input = np.array([weight_dict[w] for w in word_keys])
    df = pd.DataFrame(columns=['class', 'word', 'freq'])
    row = 0
    for c, w in sorted(zip(classes, word_keys), key=lambda t: t[0]):
        df.loc[row, 'class'] = c
        df.loc[row, 'word'] = w
        df.loc[row, 'freq'] = counter_dict[w] / num_all_words
        row += 1
        # print(f'{c}: {w}, 频数:{counter_dict[w]},频率:{counter_dict[w] / num_all_words}')
    df.to_excel('result.xlsx')

    tsne_plot(tsne_weight_input, clf)
