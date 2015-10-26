# data set: iris
# sample number: 150
# modle used: fisher linear discriminant
# number of features: 4
# training set: 135 samples
# test set: 15 samples

# result: 
# precison: 93.33%

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pdb


testN = 15
samples = datasets.load_iris()

#print samples.data[0]
#pdb.set_trace()

#print(samples.data.shape)
#print(samples.target)

clf = GaussianNB()
x,y = samples.data[:-testN], samples.target[:-testN]
clf.fit(x,y)

count = 0
for i in range(-15, -1):
    print "test sample i:"
    print "predicted lable:"
    p = str(clf.predict(samples.data[i]))[1]
    
    print(p)
    print "real lable:"
    r = str(samples.target[i])
    
    print samples.target[i]
    if p == r:
        count += 1

print "test samples:"
print testN
print "precision:"
print float(count)/float(testN)
