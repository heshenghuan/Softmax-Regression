# Softmax Regression

## CPP version design plan

- This c++ program must support non-numeric label. For example, it can use any single ASCII character to represent a label.

- Program should use text file to save model file. Then, make python version SoftmaxReg program compatible with those file.

## Solution

- Using STL map to keep a mapping relation of label to index of classifier.

		For example:
		
		There are 11 classes samples, we cannot use single Arabic number to 
		represent all the classes. So in the original data file, we used English
		letter A, B, ... ,K (11 letters) to represent 11 classes label.
		
		Assuming the degree of feature vector is 100. Then we will have a 11*100
		matrix of weight Theta. And we will also keep a mapping relation like:
		
		label - index
			A - 0
			B - 1
			C - 2
			...
			K - 10
			
- The mapping relation should be save as a text file.

-  Theta the weight matrix also should be save as a text file.
