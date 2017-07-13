# Softmax-Regression

### Introduction
> A python implementation of softmax-regression.<br>
> Using numpy.array model to represent matrix and vector.<br>
> In the usage, we used [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to show you how to use this algorithm.

## Data format

The format of training and testing data file must be:

```
<label> \t <index1>:<value1> <index2>:<value2> . . .
.
.
.
```

Each line contains an instance and is ended by a '\n' character. `<label>` and `<index>:<value>`are sperated by a '\t' character. But `<index>:<value>` and `<index>:<value>` are sperated by a space.

1. `<label>` is an indicator indicating the class id. Usually the indicator is an integer or a character.

   - Integer:

     The range of class id should be from 0 to the size of classes subtracting one. For example, the class id is 0, 1, 2 or 3 for a 4-class classification problem.

   - Character: 

     The class id should be a single character. For example, the class id can be 'B', 'M', 'E', 'S'.

2. `<index>` is a postive integer denoting the feature id. The range of feature id should be from 1 to the size of feature set.

   - For example, the feature id is 1, 2, ... 9 or 10 if the dimension of feature set is 10. 

3. `<value>` is a float denoting the value of feature.

   If the feature value equals 0, the `<index>:<value>` is encourged to be neglected for the consideration of storage space and computational speed.

Labels in the testing file are only used to calculate accuracy or errors.  If they are unknown, just fill the first column with any class labels.

## Usage

#### Create an instance

```
# create an instance of softmaxreg
>>> case = sr.SoftmaxReg()
```


#### Read training file

```
# read training file
>>> case.read_train_file(r"train.data")

# load basic information, include feature dimension and number of labels
# in this example, I used MNIST data. MNIST is a hand-writing digits
# classification problem dataset, each sample is a 28*28 image.
# so after converted it into a vector, the dimension is 784.
>>> case.loadFeatSize(784,10)
```

#### Print information
```
# print information of this model
>>> case.printinfo()
sample size:       50000
label size:        50000
label set size:    10
feature dimension: 784
```

#### Training a model

1. Using gradient descent

```
# train with batch gradient descent
>>> case.train_batch(max_iter=200,learn_rate=0.01,lamb=0.0001,delta=0.2515)
------------------------------------------------------------
START TRAIN BATCH:
Iter    1    Cost:2.3026    Acc:0.1136
Iter    2    Cost:2.2920    Acc:0.6691
Iter    3    Cost:2.2815    Acc:0.6726
Iter    4    Cost:2.2710    Acc:0.6757
Iter    5    Cost:2.2607    Acc:0.6791
Iter    6    Cost:2.2505    Acc:0.6819
Iter    7    Cost:2.2404    Acc:0.6849
Iter    8    Cost:2.2303    Acc:0.6884
Iter    9    Cost:2.2204    Acc:0.6909
Iter   10    Cost:2.2105    Acc:0.6931
.
.
.
Reach the minimal cost value threshold!
Training process finished.
```

2. Using SGD

```
#train with stochastic gradient descent
>>> case.train_sgd(max_iter=200,learn_rate=0.01,lamb=0.0001,delta=0.2515)
------------------------------------------------------------
START TRAIN SGD:
Iter    1    Cost:0.3104    Acc:0.9119
Iter    2    Cost:0.3101    Acc:0.9118
Iter    3    Cost:0.3000    Acc:0.9160
Iter    4    Cost:0.2972    Acc:0.9151
Iter    5    Cost:0.2837    Acc:0.9202
Iter    6    Cost:0.2767    Acc:0.9239
Iter    7    Cost:0.2836    Acc:0.9210
Iter    8    Cost:0.2757    Acc:0.9236
Iter    9    Cost:0.2777    Acc:0.9231
Iter   10    Cost:0.2809    Acc:0.9193	
.
.
.
Reach the minimal cost value threshold!
Training process finished.
```

3. Using mini-batch gradient descent

```
#train with mini batch gradient descent
>>> case.train_mini_batch(batch_num=200,max_iter=200,learn_rate=0.01,lamb=0.0001,delta=0.2515)
------------------------------------------------------------
START TRAIN MINI BATCH:
Iter    1    Cost:2.2928    Acc:0.3617
Iter    2    Cost:2.2818    Acc:0.5922
Iter    3    Cost:2.2713    Acc:0.5454
Iter    4    Cost:2.2605    Acc:0.5694
Iter    5    Cost:2.2514    Acc:0.6286
.
.
.
Reach the minimal cost value threshold!
Training process finished.
```

#### Classification
```
# classify a test file
# 1. make usable data
>>> x_test, y_test = make_data(r"test.data")
# 2. get the predict label set
>>> y = case.batch_classify(x_test)
# 3. calculate the accuracy
>>> acc = calc_acc(y,y_test)
>>> print "The accuracy on testDigits: %.4f"%acc
The accuracy on testDigits: 0.xxxx
```

#### Save & load  model
```
# save the model
# you can read detail by type help(case.saveModel)
>>> case.saveModel()

# load a model from file
>>> case.loadTheta(path=r"....")
>>> case.loadLabelSet(path=r"....")
```
