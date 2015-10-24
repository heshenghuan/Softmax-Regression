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

    Using numpy.array to store matrixes.
    """
    def __init__(self):
        """
        Initialization function, returns an instance of SoftmaxReg with all the
        field empty.

        x = SoftmaxReg(), x is an instance of class SoftmaxReg.
        """
        # the index of a label in label_set also be the index of label in Theta
        self.label_set = []
        self.Theta = None
        self.feat_dimension = 0
        self.class_num = 0
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

        If file does not exist, reports an IOError then returns False.

        If loads successed, returns True
        """
        if not label_set:
            print "Not given any file path, load label_set from default path."
            print "Please make sure corresponding file exist!"
            label_set = r"./label_set.pkl"
        
        try:
            inputs = open(label_set, 'rb')
            self.label_set = load(inputs)
            self.class_num = len(self.label_set)
            return True
        except IOError:
            print "Corresponding file \"label_set.pkl\" doesn\'t exist!"
            return False

    def loadTheta(self, Theta=None):
        """
        Loads Theta from file under given file path.

        If file does not exist, reports an IOError then returns False.

        If loads successed, returns True
        """
        if not Theta:
            print "Not given any file path, load Theta from default path."
            print "Please make sure corresponding file exist!"
            theta = r"./Theta.pkl"
        
        try:
            inputs = open(theta, 'rb')
            self.Theta = load(inputs)
            # self.feat_dimension = self.Theta.shape[1] - 1
            return True
        except IOError:
            print "Error:File does"
            print "Corresponding file \"Theta.pkl\" doesn\'t exist!"
            return False

    def loadFeatSize(self, feat_size, classNum):
        """
        A combination of function setFeatSize, setClassNum and initTheta.
        In order to be compatible with old API of MultiPerceptron of mine.

        It has same result with the following sentences:
            >>> x.setFeatSize(feat_size)
            >>> x.setClassNum(classNum)
            >>> x.initTheta()
        """
        self.setFeatSize(feat_size)
        self.setClassNum(classNum)
        flag = self.initTheta()
        return flag

    def setFeatSize(self, size=0):
        """
        Sets feature dimensions by the given size.
        """
        if size==0:
            print "Warning: ZERO dimensions of feature will be set!"
            print "         This would causes some trouble unpredictable!"
            print "         Please make sure the dimension of feature is 0!"
        self.feat_dimension = size

    def setClassNum(self, classNum=0):
        """
        Sets number of label classies by given classNum.
        """
        if classNum==0:
            print "Warning: ZERO class of samples will be set!"
            print "         This would causes some trouble unpredictable!"
            print "         Please make sure the number of classies is 0!"
        self.class_num = classNum

    def initTheta(self):
        """
        Initializes the Theta matrix.

        If the dimension of feature and number of classies are not be set, or
        are 0, this function will not Initializes Theta.

        Initialization successed, returns True. If not, returns False. 
        """
        if self.feat_dimension!=0 and self.class_num!=0:
            # theta = []
            # for i in xrange(self.class_num):
            #     theta.append([0.]*(self.feat_dimension+1))
            self.Theta = np.zeros((self.class_num,self.feat_dimension+1))
            return True
        else:
            print "Error: The dimension of feature and number of classies can"
            print "       not be ZERO!"
            return False

    def __COST(self, lamb):
        """
        Cost function of SoftmaxReg,.
        Returns the value of J(Theta) by the given sample, and gradient of 
        J(Theta).
        """
        prb = []
        # y =[]
        m = len(self.sample_list)
        k = len(self.label_set)

        J = 0.
        grad = np.zeros(self.Theta.shape)
        for i in xrange(m):
            prb.append(self.predict(sample))
            # get the predict label y
            # y.append(self.label_set.index(max(prb.items(),key=lambda a: a[1])[0]))
            label = self.label_list[i] # the label of sample[i] not the index
            J += math.log(prb[i][label])

            x = self.__getSampleVec(sample)
            y = self.label_set.index(label)
            for j in xrange(k):
                grad[j] += ((1 if y==j else 0) - prb[i][self.label_set[j]]) * x

        J = -J/m + lamb*sum(sum(self.Theta*self.Theta))/2 
        grad = -grad/m + lamb*self.Theta

        return J,grad

    def __getSampleVec(self, sample):
        """
        Returns a row vector by 1*(n+1).
        """
        sample_vec = np.zeros((1,self.feat_dimension+1))
        for i in sample.keys():
            sample_vec[i] = sample[i]

        return sample_vec

    def predict(self, sample):
        """
        Returns the predict vector of probabilities.
        """
        X = self.__getSampleVec(sample).T
        pred = {}
        for j in range(self.classNum):
            pred[self.label_set[j]] = np.dot(self.Theta[j,:],X)[0][0]
        
        return normalize(pred)

    def read_train_file(self, filepath):
        """
        Make traing set from file 
        Returns sample_set, label_set
        """
        data = codecs.open(filepath, 'r')
        for line in data.readlines():
            val = line.strip().split('\t')
            self.label_list.append(val[0])
            # max_index = 0
            sample_vec = {}
            val = val[-1].split(" ")
            for i in range(len(val)):
                [index, value] = val[i].split(':')
                sample_vec[int(index)] = float(value)
            self.sample_list.append(sample_vec)
        self.label_set = list(set(self.label_list))


def normalize(X):
    """
    Normalize the X.
    X is a dict type.
    """
    max_val = max(X.values())
    out = {}
    expsum = 0.
    for key in X.keys():
        out[key] = math.exp(X[key]-max_val)
        expsum += out[key]
    for key in X.keys():
        out[key] /= expsum
    return out


def create(self, size=0, classNum=0, label_list=None, sample_list=None):
    """
    Creates an instance of SoftmaxReg with given parameters.

    size: feature dimension
    classNum: label classies number 
    label_list: list of the samples' label
    sample_list: list of samples
    """
    tmp = SoftmaxReg()
    if tmp.loadFeatSize(size, classNum):
        # initialization successed
    else:
        # not successed
    return tmp