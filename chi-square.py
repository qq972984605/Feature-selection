from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer  # 处理缺失值
from sklearn.preprocessing import MinMaxScaler  # 归一化
import numpy as np
import pandas as pd

# 读取数据
fPath = 'D:\E盘\数据\microarray data\Ovary.csv'
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))

rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
sampleData = []
sampleClass = []

for i in range(rowNum):
    tempList = list(dataMatrix[i, :])
    sampleClass.append(tempList[-1])  # 最后一列是类别
    sampleData.append(tempList[:-1])  # 其余是特征

sampleM = np.array(sampleData)  # 特征矩阵
classM = np.array(sampleClass)  # 类别向量

# 处理缺失值（用均值填补）
imputer = SimpleImputer(strategy='mean')
sampleM = imputer.fit_transform(sampleM)

# 删除方差为 0 的特征（常数特征）
var = sampleM.var(axis=0)
constant_features = np.where(var == 0)[0]
sampleM = np.delete(sampleM, constant_features, axis=1)

# 对特征进行 Min-Max 归一化（卡方检验要求非负数）
scaler = MinMaxScaler()
sampleM = scaler.fit_transform(sampleM)

# 使用 SelectKBest 和 chi2 计算卡方分数
selector = SelectKBest(chi2, k='all')  # 计算所有特征的卡方得分
selector.fit(sampleM, classM)

# 获取所有特征的 Chi2 分数
scores = selector.scores_
indices = np.arange(len(scores))

# 对 Chi2 分数进行降序排序
sorted_indices = np.argsort(scores)[::-1]
sorted_scores = scores[sorted_indices]
sorted_features = indices[sorted_indices]

# 获取前 200 个特征
top_200_scores = sorted_scores[:200]
top_200_features = sorted_features[:200]

# 输出结果
print("Top 200 Features' Indices and Chi2 Scores:")
print(list(top_200_features))
