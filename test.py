#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14:13:08 2015-10-24

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import softmaxreg as sr

#case = sr.create(size=10,classNum=2)


case = sr.SoftmaxReg()
case.read_train_file(r"trainDigits.data")
case.loadFeatSize(1024,10)
case.printinfo()
#print case.label_set
case.train_batch(learn_rate=0.1,lamb=0.2)