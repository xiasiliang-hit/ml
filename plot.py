
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import pdb


testN = 15
samples = datasets.load_iris()

#print samples.data[0]
#pdb.set_trace()

#print(samples.data.shape)
#print(samples.target)
x = [0, 1, 2, 3]
xtext = ['SVM', 'fisher LDA', 'naive bayesian', 'decision tree']
y = [0.87, 0.93, 0.93, 0.93]


plt.scatter(x, y)
plt.xticks(x, xtext)
plt.xlabel("model")
plt.ylabel("precision")

plt.show()
