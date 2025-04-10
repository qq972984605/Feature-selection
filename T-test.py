import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer  # 处理缺失值
from sklearn.preprocessing import MinMaxScaler  # 归一化

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

# 对特征进行 Min-Max 归一化
scaler = MinMaxScaler()
sampleM = scaler.fit_transform(sampleM)

# 计算每个特征的 T-test 统计量
unique_classes = np.unique(classM)  # 获取所有类别
group1 = sampleM[classM == unique_classes[0]]  # 第一个类别的样本
group2 = sampleM[classM == unique_classes[1]]  # 第二个类别的样本

t_values = []
p_values = []

for i in range(sampleM.shape[1]):  # 遍历每个特征
    t_stat, p_val = ttest_ind(group1[:, i], group2[:, i], equal_var=False)  # 独立样本 T 检验
    t_values.append(abs(t_stat))  # 取绝对值，保证排名
    p_values.append(p_val)

t_values = np.array(t_values)
p_values = np.array(p_values)
indices = np.arange(len(t_values))

# 对 T 值进行降序排序
sorted_indices = np.argsort(t_values)[::-1]
sorted_t_values = t_values[sorted_indices]
sorted_features = indices[sorted_indices]

# 获取前 200 个特征
top_200_t_values = sorted_t_values[:200]
top_200_features = sorted_features[:200]

# 输出结果
print("Top 200 Features' Indices and T Scores (T-test):")
print(list(top_200_features))
