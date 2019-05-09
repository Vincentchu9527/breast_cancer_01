# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)

# 读取并探索数据
inputfile = './data.csv'
data = pd.read_csv(inputfile)
print(data.head())
print(data.describe())
print(data.info())
print(data.diagnosis.value_counts())

# 数据清洗与分类
# print(data.columns)
data_y = data.diagnosis.replace(['B', 'M'], [0, 1])
# print(data_y.value_counts())
# 把平均值数据, 方差数据与最坏值数据分开
data_mean = data[data.columns[2:12]]
# print(data_mean.columns)
data_se = data[data.columns[12:22]]
# print(data_se.columns)
data_worst = data[data.columns[22:32]]
# print(data_worst.columns)

# 1. 相关性分析降维
# 画相关性系数图
mean_corr = data_mean.corr()
sns.heatmap(mean_corr, annot=True)
plt.show()
# 特征选择
features1 = data[['radius_mean', 'texture_mean', 'smoothness_mean',
                  'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']]
# 标准化
ss = StandardScaler()
features1 = ss.fit_transform(features1)
# 训练集数据集分割
train_x, test_x, train_y, test_y = train_test_split(features1, data_y, test_size=.3, random_state=33)
# LinearSVC模型
model1 = LinearSVC()
model1.fit(train_x, train_y)
pre_1 = model1.predict(test_x)
print('模型1得分:', accuracy_score(test_y, pre_1))

# 2. 主成分分析降维
# PCA 模型
pca = PCA(n_components=6)
# pca = PCA(n_components='mle')
model2 = Pipeline([
    ('StandardScaler', StandardScaler()),
    ('PCA', pca)])
features2 = model2.fit_transform(data_mean)
print(features2.shape)
print('贡献率:', pca.explained_variance_ratio_)
# 训练数据集分割
train_x, test_x, train_y, test_y = train_test_split(features2, data_y, test_size=.3, random_state=33)
# LinearSVC模型
model2 = LinearSVC()
model2.fit(train_x, train_y)
pre_2 = model2.predict(test_x)
print('模型2得分:', accuracy_score(test_y, pre_2))
