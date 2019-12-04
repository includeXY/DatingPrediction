from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    """
    :param inX: 未知类别属性点 ，是array类型
    :param dataSet: 训练数据集中的属性点，是array类型
    :param labels: 训练数据集中的标签，是list类型
    :param k: 选择与inX距离最小的前k个点
    :return: 返回前k个点中同一类别最多的那一类标签
    """
    dataSetSize = dataSet.shape[0]       #array_name.shape返回数组的行列(m,n)元组
    diffMat = tile(inX,(dataSetSize,1)) - dataSet       #得到未知类别点与所有已知类别点的坐标差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                #axis=1表示按行求和，=0表示按列求和
    distances = sqDistances**0.5
    sortedDisIndicies = distances.argsort()            #返回递增排序的索引，类型为array
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]      #选择与inX距离最小的k个点的类别
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #查找字典classCount中是否存在关键值voteIlabel，没有返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #key=operator.itemgetter(1)表示根据第二个域（类别数量）逆序排序,classCount.items()返回字典的键值对列表
    return sortedClassCount[0][0]

def file2matrix(filename):
    """
    :param filename:文件名字符串
    :return: 训练样本矩阵和类标签向量
    """
    fr = open(filename)                              #打开一个文件，并创建file对象
    arrayOfLines = fr.readlines()                      #读取所有行并返回一个列表
    numeOfLines = len(arrayOfLines)                  #行数
    returnMat = zeros((numeOfLines,3))               #创建一个矩阵，大小是numeOfLines*3，元素起始全为0
    classLabelVector = []                            #用以保存类标签向量
    index = 0
    for line in arrayOfLines:                        #访问列表的每一行
        line = line.strip()                          #移除字符串头尾的字符，默认为空格和换行
        listFromLine = line.split('\t')              #通过制定分割符对字符串进行切割
        returnMat[index,:] = listFromLine[0:3]       #[0:3]表示左闭右开
        classLabelVector.append(int(listFromLine[-1]))     #-1表示最后一列元素
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    """
    :param dataSet:训练集
    :return: 归一化后的训练集
    """
    minVals = dataSet.min(0)     #参数0表示从列中选出最小值,返回每一列中最小值组成的矩阵
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))   #使用numpy库中的函数tile（）将变量内容复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.1                     #从测试集中选取10%的数据作为测试集
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(hoRatio*m)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)   #对每个测试样本，通过分类器得到分类标签
        print('the classifier came back with: %d, the real answer is: %d' % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount +=1
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all!', 'in small doses!', 'in large doses!']
    percentTats = float(input('percentags of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of icecream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArry = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArry - minVals)/ranges,normMat,datingLabels,4)
    print('You will probably like this person: ',\
          resultList[classifierResult-1])