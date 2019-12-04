# DatingPrediction
Using kNN to predict wether a person suitable for dating，Python 3.7

kNN.py内的函数描述：
classify0函数的功能是使用k-近邻算法将每组数据划分到某个类中
file2matrix函数将文本记录转换成numpy类型
autoNorm函数归一化特征值，是每个特征值缩放到范围[0,1]内
datingClassTest是分类器针对约会网站的测试代码
classifyPerson约会网站预测函数

test.py是用来测试kNN.py中算法的正确性

datingTestSet2.txt是训练数据集，里面的样本随机排列。
