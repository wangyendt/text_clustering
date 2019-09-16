# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/16 22:27
# software: PyCharm

import codecs
from copy import copy

import jieba.analyse
import pandas as pd

if __name__ == '__main__':
    pd.set_option('max_colwidth', 1000)
    # rows = pd.read_excel('sth.xlsx', use_cols=[1, 2], encoding='utf-8')
    with open('mao.txt') as f:
        rows = f.readlines()[0]
    # rows = rows.astype(str)
    print(rows)
    segments = []
    stopwords = [line.strip() for line in codecs.open('stopped.txt', 'r', 'utf-8').readlines()]
    # stopwords = ['习近平','胡锦涛']
    print(stopwords)
    jieba.analyse.set_stop_words('stopped.txt')
    content = copy(rows)
    words = jieba.cut(content)
    # words = jieba.analyse.textrank(content, topK=30, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    splitedStr = ''
    for word in words:
        if word not in stopwords:
            segments.append({'word': word, 'count': 1})
            splitedStr += word + ' '

    splitedStr = ''
    for word in words:
        segments.append({'word': word, 'count': 1})
        splitedStr += word + ' '

    dfSg = pd.DataFrame(segments)
    dfWord = dfSg.groupby('word')['count'].sum()
    a = pd.DataFrame({'char': dfWord.index, 'num': dfWord.values})
    a.to_excel('result.xlsx')
