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

# 计算皮尔逊相关系数
correlation_scores = np.array([np.corrcoef(sampleM[:, i], classM)[0, 1] for i in range(sampleM.shape[1])])

# 取绝对值并排序（相关性越大，特征越重要）
abs_correlation_scores = np.abs(correlation_scores)
sorted_indices = np.argsort(abs_correlation_scores)[::-1]  # 降序排序

# 选择前 200 个特征
top_200_features = sorted_indices[:200]
top_200_scores = abs_correlation_scores[top_200_features]

# 输出结果
print("Top 200 Features' Indices and Correlation Scores:")
print(list(top_200_features))
