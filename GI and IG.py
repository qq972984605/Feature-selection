import numpy as np
import pandas as pd
from collections import Counter
from sklearn.impute import SimpleImputer  # 处理缺失值
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer  # 归一化 & 离散化


# 读取数据
def load_data(file_path, discretize=False, bins=5):
    dataMatrix = np.array(pd.read_csv(file_path, header=None, skiprows=1))

    # 分离特征和类别
    sampleData = dataMatrix[:, :-1]  # 所有行，去掉最后一列
    sampleClass = dataMatrix[:, -1]  # 只取最后一列作为类别标签

    # 处理缺失值（用均值填补）
    imputer = SimpleImputer(strategy='mean')
    sampleData = imputer.fit_transform(sampleData)

    # 删除方差为 0 的特征（常数特征）
    var = sampleData.var(axis=0)
    sampleData = sampleData[:, var > 0]

    # 归一化特征
    scaler = MinMaxScaler()
    sampleData = scaler.fit_transform(sampleData)

    # 是否进行离散化（避免 IG 和 GI 结果过于相似）
    if discretize:
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        sampleData = discretizer.fit_transform(sampleData)

    return sampleData, sampleClass


# 计算熵 H(Y)
def entropy(y):
    counts = np.bincount(y.astype(int))  # 计算类别频数
    probs = counts[counts > 0] / len(y)  # 计算概率
    return -np.sum(probs * np.log2(probs))  # 熵公式


# 计算条件熵 H(Y | X)
def conditional_entropy(x, y):
    unique_values, counts = np.unique(x, return_counts=True)
    cond_entropy = 0
    for val, count in zip(unique_values, counts):
        subset_y = y[x == val]
        cond_entropy += (count / len(x)) * entropy(subset_y)
    return cond_entropy


# 计算信息增益 IG
def compute_information_gain(X, y):
    H_Y = entropy(y)
    return np.array([H_Y - conditional_entropy(X[:, i], y) for i in range(X.shape[1])])


# 计算基尼不纯度 Gini(Y)
def gini_impurity(y):
    counts = np.bincount(y.astype(int))
    probs = counts[counts > 0] / len(y)
    return 1 - np.sum(probs ** 2)


# 计算条件基尼不纯度 Gini(Y | X)
def conditional_gini(x, y):
    unique_values, counts = np.unique(x, return_counts=True)
    cond_gini = 0
    for val, count in zip(unique_values, counts):
        subset_y = y[x == val]
        cond_gini += (count / len(x)) * gini_impurity(subset_y)
    return cond_gini


# 计算基尼增益 GI
def compute_gini_gain(X, y):
    Gini_Y = gini_impurity(y)
    return np.array([Gini_Y - conditional_gini(X[:, i], y) for i in range(X.shape[1])])


# 计算并排序特征重要性
def rank_features(scores):
    sorted_indices = np.argsort(scores)[::-1]
    return sorted_indices, scores[sorted_indices]


# 主程序
file_path = 'D:\\E盘\\数据\\microarray data\\Ovary.csv'
discretize = True  # 是否进行离散化（默认为 True）

# 加载数据
sampleM, classM = load_data(file_path, discretize=discretize, bins=5)

# 计算 IG 和 GI
info_gains = compute_information_gain(sampleM, classM)
gini_gains = compute_gini_gain(sampleM, classM)

# 排序特征
sorted_ig_indices, sorted_ig_scores = rank_features(info_gains)
sorted_gi_indices, sorted_gi_scores = rank_features(gini_gains)

# 选择前 200 个特征
top_200_ig = sorted_ig_indices[:200]
top_200_gi = sorted_gi_indices[:200]

# 输出结果
print("Top 200 Features' Indices based on Information Gain:")
print(list(top_200_ig))

print("\nTop 200 Features' Indices based on Gini Gain:")
print(list(top_200_gi))
