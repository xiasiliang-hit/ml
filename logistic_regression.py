# dataset: forest fire
# sample: 
# number of features: 10

# model: linear regression
# test: 10 fold cross validation
# train set size: 406
# test set size: 56

# result:
# variance of prediction : 0.404

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import sklearn.datasets as ds
import sklearn as sk
from sklearn import linear_model
import pdb
#from sklearn.metrics import mean_absolute_error


#samples = ds.load_boston()

sum_var = 0.0
sum_abs_error = 0.0
round = 10

x = []
y = []

f = open("forestfires.csv")
for line in f.readlines():

    l = line.split(',')
    
    if (l[0] == 'X'):
        continue

    label = l[-1]
    del l[-1]
    del l[2]
    del l[2]

    #print l
    w = [float(a) for a in l]

    x.append(w)
    y.append(label)
    
n = len(x)

    
for i in range(0, round):
#    print "iteration:" + str(i)

    ind  = n/10*i
    end = n/10*(i+1)

#    print "ind"
#    print ind
#    print "end"
#    print end

#    print samples.data[end:]
    train_x = []
    train_y = []
    for i in x[:ind]:
        train_x.append(i)

    for j in x[end:]:
        train_x.append(j)

    for i in y[:ind]:
        train_y.append(i)

    for j in y[end:]:
        train_y.append(j)
        
    
    
    test_x = x[ind:end] 
    test_y = y[ind:end]

    regr = linear_model.LogisticRegression()
    regr.fit(train_x, train_y)
    
    
    sum_var += regr.score(test_x, test_y)
    #    sum_abs_error += sk.metrics.mean_absolute_error(test_y, regr.predict(test_x))
#print('Coefficients: ', regr.coef_)    

    #for j in len(test_x):
    #    sum_var += np.mean((regr.predict(test_x[j]) - test_y[j]) ** 2


    if i == 1:        
        plt.scatter(test_x, test_y,  color='black')
        plt.plot(test_x, regr.predict(test_x), color='blue',
             linewidth=3)
        

        plt.show()    


#print "average error:"
#print float(sum_abs_error)/float(round)

print "variance:"
print(float(sum_var)/float(round))

