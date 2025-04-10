from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer  # 使用 SimpleImputer
from sklearn.preprocessing import MinMaxScaler  # 归一化
import numpy as np
import pandas as pd

# 读取数据
fPath = 'D:\E盘\数据\microarray data\Ovary.csv'
dataMatrix = np.array(
    pd.read_csv(fPath, header=None, skiprows=1))
rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
sampleData = []
sampleClass = []
for i in range(0, rowNum):
    tempList = list(dataMatrix[i, :])
    sampleClass.append(tempList[-1])
    sampleData.append(tempList[:-1])
sampleM = np.array(sampleData)  # 特征矩阵
classM = np.array(sampleClass)  # 类别向量

# 处理缺失值：用均值填补
imputer = SimpleImputer(strategy='mean')  # 使用 SimpleImputer
sampleM = imputer.fit_transform(sampleM)

# 删除常数特征
# 计算每个特征的方差
var = sampleM.var(axis=0)
# 找到方差为零的特征索引
constant_features = np.where(var == 0)[0]
# 删除这些特征
sampleM = np.delete(sampleM, constant_features, axis=1)

# 对特征进行归一化（Min-Max 归一化到 [0,1]）
scaler = MinMaxScaler()
sampleM = scaler.fit_transform(sampleM)

# 使用 SelectKBest 和 f_classif 计算 F-Score
selector = SelectKBest(f_classif, k='all')  # 保留所有特征计算分数
selector.fit(sampleM, classM)

# 获取所有特征的 F-score 和对应的索引
scores = selector.scores_  # F-Scores for all features
indices = np.arange(len(scores))  # Indices of all features

# 对 F-Scores 进行排序
sorted_indices = np.argsort(scores)[::-1]  # Descending order
sorted_scores = scores[sorted_indices]
sorted_features = indices[sorted_indices]

# 获取前 200 个特征
top_200_scores = sorted_scores[:200]
top_200_features = sorted_features[:200]

# 输出结果
print("Top 200 Features' Indices and F-Scores:")
print(list(top_200_features))