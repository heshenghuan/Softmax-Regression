#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 10:44:02 2015-10-23

@author: heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import codecs
import numpy as np
import math
from cPickle import dump
from cPickle import load

def calc_acc(label_list1, label_list2):
    same = [int(x==y) for x,y in zip(label_list1, label_list2)]
    acc = float(same.count(1))/len(same)
    return acc

def sigmoid_func(x):
    return 1/(1+math.exp(-x/5000))

class SoftmaxReg:
    """
    Softmax regression algorithm implemented by python 2.7

    Using numpy.mat to store matrixes.
    """
    def __init__(self):
        """
        Initialization function, returns an instance of SoftmaxReg.

        x = SoftmaxReg(), x is an instance of class SoftmaxReg.
        """
        # the index of a label in label_set also be the index of label in Theta
        self.label_set = []
        self.Theta = None
        self.feat_dimension = 0
        self.sample_list = []
        self.label_list = []

    def saveModel(self, path=None):
        """
        Stores the model under given folder path.

        x.saveModel(r)
        """
        if not path:
            print "Using default path(./)."
            path = r"./"
        else:
            if not os.path.exists(path):
                os.makedirs(path)
                print "Folder doesn\'t exist, program automatically create the folder."
            print "Storing model file under folder:",path,'.'

        output1 = open(path+r"label_set.pkl",'wb')
        dump(self.label_set, output1, -1)
        output1.close()
        output2 = open(path+r"Theta.pkl",'wb')
        dump(self.Theta, output2, -1)
        output2.close()
        #release the memory
        self.label_set = []
        self.Theta = None
        self.sample_list = []
        self.label_list = []

    def loadLabelset(self, label_set=None):
        """
        Loads label_set from file under given file path.

        If file not exist, returns an IOError.
        """
        if not label_set:
            print "Not given any file path, load label_set from default path."
            print "Please make sure corresponding file exist!"
            label_set = r"./label_set.pkl"
        try:
            inputs = open(label_set, 'rb')
            self.label_set = load(inputs)
            return True
        except IOError:
            print "Corresponding file \"label_set.pkl\" doesn\'t exist!"
            return False