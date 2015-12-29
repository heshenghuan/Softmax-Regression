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
import random
from cPickle import dump
from cPickle import load


def calc_acc(label_list1, label_list2):
    same = [int(x == y) for x, y in zip(label_list1, label_list2)]
    acc = float(same.count(1)) / len(same)
    return acc


def sigmoid(x):
    return 1 / (1 + math.exp(-x / 5000))


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

    def printinfo(self):
        print "sample size:      ", len(self.sample_list)
        print "label size:       ", len(self.label_list)
        print "label set size:   ", len(self.label_set)
        print "feature dimension:", self.feat_dimension

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
                print "Folder doesn\'t exist, program will create the folder."
            print "Storing model file under folder:", path, '.'

        output1 = open(path + r"label_set.pkl", 'wb')
        dump(self.label_set, output1, -1)
        output1.close()
        output2 = open(path + r"Theta.pkl", 'wb')
        dump(self.Theta, output2, -1)
        output2.close()
        # release the memory
        self.label_set = []
        self.Theta = None
        self.sample_list = []
        self.label_list = []

    def loadLabelSet(self, label_set=None):
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
        if size == 0:
            print "Warning: ZERO dimensions of feature will be set!"
            print "         This would causes some trouble unpredictable!"
            print "         Please make sure the dimension of feature is 0!"
        self.feat_dimension = size

    def setClassNum(self, classNum=0):
        """
        Sets number of label classies by given classNum.
        """
        if classNum == 0:
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
        if self.feat_dimension != 0 and self.class_num != 0:
            # theta = []
            # for i in xrange(self.class_num):
            #     theta.append([0.]*(self.feat_dimension+1))
            self.Theta = np.zeros((self.class_num, self.feat_dimension + 1))
            return True
        else:
            print "Error: The dimension of feature and number of classies can"
            print "       not be ZERO!"
            return False

    def __COST(self, lamb):
        """
        Cost function of SoftmaxReg,.
        Returns the value of J(Theta) by the given sample list, and gradient of
        J(Theta).
        """
        prb = []
        # y =[]
        m = len(self.sample_list)
        k = len(self.label_set)

        J = 0.
        grad = np.zeros(self.Theta.shape)
        error = 0
        for i in xrange(m):
            prb.append(self.predict(self.sample_list[i]))
            # get the predict label y
            label = self.label_list[i]  # the label of sample[i] not the index
            pred = prb[i].index(max(prb[i]))
            if label != self.label_set[pred]:
                error += 1
            J += math.log(prb[i][self.label_set.index(label)])

            x = self.__getSampleVec(self.sample_list[i])
            y = self.label_set.index(label)  # the index of sample[i]'s label
            for j in xrange(k):
                grad[j] += ((1 if y == j else 0) - prb[i][j]) * x[0]

        # compute the cost function value, gradient, and accuracy of
        # classification
        J = -J / m + lamb * sum(sum(self.Theta * self.Theta)) / 2
        grad = -grad / m + lamb * self.Theta
        acc = 1 - error / float(m)

        return (J, grad, acc)

    def __getSampleVec(self, sample):
        """
        Returns a row vector by 1*(n+1).
        """
        sample_vec = np.zeros((1, self.feat_dimension + 1))
        for i in sample.keys():
            sample_vec[0][i] = sample[i]

        return sample_vec

    def predict(self, sample):
        """
        Returns the predict vector of probabilities.
        """
        X = self.__getSampleVec(sample).T
        pred = []
        for j in range(self.class_num):
            pred.append(np.dot(self.Theta[j, :], X)[0])

        return normalize(pred)

    def train_batch(self, max_iter=100, learn_rate=0.1, lamb=0.1, delta=0.01):
        """
        Training a softmax regression model, the samples and labels should be
        already assigned to field self.sample_list and self.label_list.

        max_iter: the maximum number of iteration(default 100).
        learn_rate: the learning rate of train process(default 0.1).
        lamb: the coefficient of weight decay(default 0.1).
        delta: the threshold of cost function value(default 0.01), and the
        signal of training finished.
        """
        print '-' * 60
        print "START TRAIN BATCH:"

        # training process
        J = J_pre = 0.
        rd = 0
        while rd < max_iter:
            J, grad, acc = self.__COST(lamb)
            self.Theta -= learn_rate * grad
            rd += 1
            print "Iter %4d    Cost:%4.4f    Acc:%4.4f" % (rd, J, acc)
            if rd != 0 and (J_pre - J) <= delta and J < J_pre:
                print "\n\nReach the minimal cost value threshold!"
                break
            J_pre = J

        if rd == max_iter:
            print "Train loop has reached the maximum of iteration."

        print "Training process finished."

    def train_sgd(self, max_iter=100, learn_rate=0.01, lamb=0.1, delta=0.01):
        """
        Training a Softmax regression model in stochastic gradient descent
        method.

        max_iter: the maximum number of iteration(default 100).
        learn_rate: the learning rate of train process(default 0.01).
        lamb: the coefficient of weight decay(default 0.1).
        delta: the threshold of cost function value(default 0.01), and the
        signal of training finished.
        """
        print '-' * 60
        print "START TRAIN SGD:"
        # y =[]
        m = len(self.sample_list)
        n = self.feat_dimension + 1
        k = len(self.label_set)

        J = 0.
        J_pre = 0.
        # grad = np.zeros(self.Theta.shape)
        rd = 0
        while rd < max_iter * m:
            if rd % m == 0 and rd != 0:
                loop = rd / m
                error = 0
                J = 0.
                prb = []
                for i in xrange(m):
                    prb.append(self.predict(self.sample_list[i]))
                    # the label of sample[i] not the index
                    label = self.label_list[i]
                    pred = prb[i].index(max(prb[i]))
                    if label != self.label_set[pred]:
                        error += 1
                    J += math.log(prb[i][self.label_set.index(label)])

                J = -J / m + lamb * sum(sum(self.Theta * self.Theta)) / 2
                acc = 1 - error / float(m)
                print "Iter %4d    Cost:%4.4f    Acc:%4.4f" % (loop, J, acc)
                if loop != 1 and (J_pre - J) <= delta and J < J_pre:
                    print "\n\nReach the minimal cost value threshold!"
                    break
                J_pre = J

            i = random.randint(0, m - 1)
            label = self.label_list[i]  # the label of sample[i] not the index
            pred_prb = self.predict(self.sample_list[i])
            pred = pred_prb.index(max(pred_prb))
            x = self.__getSampleVec(self.sample_list[i])
            y = self.label_set.index(label)  # the index of sample[i]'s label
            for j in xrange(k):
                # grad[j] += ((1 if y==j else 0) - pred_prb[j]) * x[0]
                self.Theta[j] += learn_rate * \
                    (((1 if y == j else 0) -
                      pred_prb[j]) * x[0] - lamb * self.Theta[j])

            # grad = -grad+lamb*self.Theta
            # self.Theta -= learn_rate * grad
            rd += 1

        if rd == max_iter * m:
            print "Train loop has reached the maximum of iteration."

        print "Training process finished."

    def train_mini_batch(self, batch_num=10, max_iter=100, learn_rate=0.01,
                         lamb=0.1, delta=0.01):
        """
        Training a Softmax regression model in stochastic gradient descent
        method with mini batch.

        max_iter: the maximum number of iteration(default 100).
        learn_rate: the learning rate of train process(default 0.01).
        lamb: the coefficient of weight decay(default 0.1).
        delta: the threshold of cost function value(default 0.01), and the
        signal of training finished.
        """
        print '-' * 60
        print "START TRAIN MINI BATCH:"
        # y =[]
        m = len(self.sample_list)
        n = self.feat_dimension + 1
        k = len(self.label_set)

        J = J_pre = 0.
        # grad = np.zeros(self.Theta.shape)
        rd = 0
        while rd < max_iter:
            batch_list = []
            while len(batch_list) < batch_num:
                index = random.randint(0, m - 1)
                if index not in batch_list:
                    batch_list.append(index)
            grad = np.zeros(self.Theta.shape)
            for i in batch_list:
                # the label of sample[i] not the index
                label = self.label_list[i]
                pred_prb = self.predict(self.sample_list[i])
                pred = pred_prb.index(max(pred_prb))
                x = self.__getSampleVec(self.sample_list[i])
                # the index of sample[i]'s label
                y = self.label_set.index(label)
                for j in xrange(k):
                    grad[j] += ((1 if y == j else 0) - pred_prb[j]) * x[0]
            self.Theta += learn_rate * (grad / batch_num + lamb * self.Theta)
            rd += 1

            error = 0
            J = 0.
            prb = []
            for i in xrange(m):
                prb.append(self.predict(self.sample_list[i]))
                # the label of sample[i] not the index
                label = self.label_list[i]
                pred = prb[i].index(max(prb[i]))
                if label != self.label_set[pred]:
                    error += 1
                J += math.log(prb[i][self.label_set.index(label)])

            J = -J / m + lamb * sum(sum(self.Theta * self.Theta)) / 2
            acc = 1 - error / float(m)
            print "Iter %4d    Cost:%4.4f    Acc:%4.4f" % (rd, J, acc)
            if rd != 0 and (J_pre - J) <= delta and J < J_pre:
                print "\n\nReach the minimal cost value threshold!"
                break
            J_pre = J

        if rd == max_iter:
            print "Train loop has reached the maximum of iteration."

        print "Training process finished."

    def classify(self, sample_test):
        """Classify the sample_test, returns the most likely label."""
        prb = self.predict(sample_test)
        index = prb.index(max(prb))
        label = self.label_set[index]
        return label

    def batch_classify(self, sample_test_list):
        """
        Doing classification for a list of sample.

        Returns a list of predicted label for each test sample.
        """
        labels = []
        for sample in sample_test_list:
            labels.append(self.classify(sample))
        return labels

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
    Normalize the X, a list of float value.
    """
    max_val = max(X)
    out = []
    expsum = 0.
    for key in range(len(X)):
        out.append(math.exp(X[key] - max_val))
        expsum += out[key]
    for key in range(len(X)):
        out[key] /= expsum
    return out


def make_data(filepath):
    """
    Makes sample list and label list from file.
    Returns a tuple of sample list and label list.
    """
    samples = []
    labels = []
    data = codecs.open(filepath, 'r')
    for line in data.readlines():
        val = line.strip().split('\t')
        labels.append(val[0])
        sample_vec = {}
        val = val[-1].split(" ")
        for i in range(0, len(val)):
            [index, value] = val[i].split(':')
            sample_vec[int(index)] = float(value)
        samples.append(sample_vec)
    return samples, labels


def create(size=0, classNum=0, label_list=None, sample_list=None):
    """
    Creates an instance of SoftmaxReg with given parameters.

    size: feature dimension
    classNum: label classies number
    label_list: list of the samples' label
    sample_list: list of samples
    """
    tmp = SoftmaxReg()
    tag = True
    if tmp.loadFeatSize(size, classNum):
        # initialization successed
        print "Initialization successed!"
    else:
        # not successed
        tag = False
        print "Initialization failed! Please checked your parameters."
    return (tmp, tag)
