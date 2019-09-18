# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/9/18 1:22
# software: PyCharm


if __name__ == '__main__':
    with open('mao.txt','r') as f:
        rows = f.readlines()[0]
    sentences = rows.split('。')
    with open('mao2.txt','w') as f:
        for sen in sentences:
            f.writelines(sen + '\n')
