# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm

import codecs
import collections
import functools
import os
import random
import time

import jieba.analyse
import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

WORD2VEC_SIZE = 1025
NUM_KEYWORDS_PER_DOC = 10
NUM_CLUSTERS = 8
KEY_WORD_TFIDF_THD = 0.28
FREQ_THD = 0.00001
USE_TFIDF = True
USE_FREQ = True
MAX_NUM_FEATURES = 1000


def func_timer(func):
    """
    用于计算函数执行时间
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        r = func(*args, **kw)
        print('%s excute in %.3f s' % (func.__name__, (time.time() - start)))
        return r

    return wrapper


@func_timer
def add(x, y):
    return x + y


def list_all_files(root: str, keys=[], outliers=[], full_path=False):
    """
    列出某个文件下所有文件的全路径

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: 根目录
    :param keys: 所有关键字
    :param outliers: 所有排除关键字
    :param full_path: 是否返回全路径，True为全路径
    :return:
            所有根目录下包含关键字的文件全路径
    """
    _files = []
    _list = os.listdir(root)
    for i in range(len(_list)):
        path = os.path.join(root, _list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path, keys, outliers, full_path))
        if os.path.isfile(path) \
                and all([k in path for k in keys]) \
                and not any([o in path for o in outliers]):
            _files.append(os.path.abspath(path) if full_path else path)
    return _files


def tsne_plot(data, cls):
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=100)
    decomposition_data = tsne.fit_transform(data)

    x = []
    y = []

    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=cls.labels_, marker="o")
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    pd.set_option('max_colwidth', 1000)
    np.random.seed(1)
    root = r'D:\Code\Github\python\text_clustering\data'
    stop_word_path = r'D:\Code\Github\python\text_clustering\stopped_words.txt'
    for file in list_all_files(root, keys=['article']):
        print(file)
        with open(file, 'r', encoding='utf-8') as f:
            rows = f.readlines()
    # stopwords用set存储，查找效率高于list
    stopwords = set([line.strip() for line in codecs.open(stop_word_path, 'r', 'utf-8').readlines()])
    jieba.analyse.set_stop_words(stop_word_path)
    corpus = []
    whole_words = []
    counter_dict = collections.defaultdict(int)
    num_all_words = 0
    rows = rows[:len(rows) // 2]
    for ri, r in enumerate(rows):
        # if ri % 8 != 0: continue
        if ri % 1000 == 0:
            print(ri, len(rows))
        content = r.strip(' ').replace('\r', '').replace('\n', '')
        if not content: continue
        whole_words.append([])
        words = jieba.cut(content)
        # words = jieba.analyse.textrank(content, topK=30, withWeight=False, allowPOS=('ns', 'n', 'vn'))
        split_str = ''
        for word in words:
            if len(word) > 1 and word not in stopwords and not any([d in word for d in '0123456789一二三四五六七八九']):
                counter_dict[word] += 1
                num_all_words += 1
                whole_words[-1].append(word)
                split_str += word + ' '
        if split_str:
            corpus.append(split_str)

    import random
    cor = random.sample(corpus, 50000)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(cor))
    all_words = vectorizer.get_feature_names()
    print("word feature length: {}".format(len(all_words)))
    svd = TruncatedSVD(MAX_NUM_FEATURES)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    tfidf_weight = lsa.fit_transform(tfidf)
    print(tfidf_weight.shape)

    # use pca:
    # tfidf_weight = tfidf.toarray()
    # pca = PCA(n_components=MAX_NUM_FEATURES)
    # tfidf_weight = pca.fit_transform(tfidf)
    # print(tfidf_weight.shape)
    # pca = PCA(n_components=100)
    # tfidf_weight = pca.fit_transform(tfidf_weight)

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

    model = Word2Vec(LineSentence('split_str.txt'), size=WORD2VEC_SIZE, window=5, min_count=1, workers=8)
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
    save = collections.defaultdict(list)
    for d in df.iterrows():
        save[d[1]['class']].append({d[1]['word']: round(d[1]['freq'], 5)})
    with open('result.txt', 'w') as f:
        print(save.keys())
        for k, v in save.items():
            # print(type(k), type(map(str, v)))
            f.writelines(str(k) + ' ' + ''.join(list(map(str, v))) + '\n')
    df.to_excel('result.xlsx')
    print(df.head(4))

    tsne_plot(tsne_weight_input, clf)
