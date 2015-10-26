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
case.read_train_file(r"train.data")
case.loadFeatSize(784,10)
case.printinfo()
#print case.label_set
case.train_sgd(max_iter=200,learn_rate=0.01,lamb=0.0,delta=0.2515)

x_test, y_test = sr.make_data(r"test.data")

y = case.batch_classify(x_test)

acc = sr.calc_acc(y,y_test)
print "The accuracy on testDigits:"
print acc   

case.saveModel()