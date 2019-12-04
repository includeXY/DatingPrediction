import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
group,labels = kNN.createDataSet()
print(group)
print(labels)

label = kNN.classify0([0,0], group, labels, 3)
print(label)

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels[0:20])

"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),array(datingLabels))          #第三个参数设置点的大小，第四个参数设置点的颜色
plt.show()
"""
#d,e = kNN.file2matrix('datingTestSet.txt')
a,b,c = kNN.autoNorm(datingDataMat)
print(a)
print(b)
print(c)

kNN.datingClassTest()

kNN.classifyPerson()
