CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC

all: smrtrain smrclassify

smrtrain: smr_train.cpp softmaxreg.o
	$(CXX) $(CFLAGS) smr_train.cpp softmaxreg.o -o smr_train -lm
smrclassify: smr_classify.cpp softmaxreg.o
	$(CXX) $(CFLAGS) smr_classify.cpp softmaxreg.o -o smr_classify -lm
clean:
	rm -f softmaxreg.o smr_train smr_classify
