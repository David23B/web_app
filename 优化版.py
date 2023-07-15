#!/usr/bin/env python
# coding: utf-8

# ### 基本思路
# 观察出显著特征，作为训练使用的特征。
#     
#     1.对训练集的label0和label1进行噪声去除
#     2.对训练集进行Knn填补缺失值
#     3.选择出训练集的group1特征，使用lightgbm或其他模型进行训练
#     4.k折交叉验证，画出混淆矩阵，查看在训练集上的效果（一般较好）。检验label0、label1、label2的分类结果（一般较差）
#     5.对验证集的label2进行噪声去除
#     6.对验证集进行Knn填补缺失值
#     7.直接对验证集进行预测（只关注group1的特征）
#     8.画出混淆矩阵，查看在验证集上的效果（一般较差），查看是否为0、1、2分类较差。
#     9.使用高斯混合拟合出训练集中group2的特征
#     10.使用概率密度取值作为分类概率，直接将验证集上分类为0、1、2的样本重新通过高斯混合概率进行k次判断（k为group2的特征数）。规则为：任意的x>50，则判定为2。x<50，则连续比较0和1谁的概率密度更大，多数获胜。
#     

# ## 第一步，读取文件，进行噪声去除

# In[265]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as n
# 读取原始数据
orig_df = pd.read_csv('train_10000.csv')
orig_df


# #### 选择我们预先准备好的特征，去除噪声

# In[376]:


import numpy as np

features_group=['sample_id', 'label', 'feature2','feature5', 'feature10', 'feature15', 'feature19', 'feature22', 'feature31', 'feature49', 'feature71', 'feature85','feature13', 'feature14', 'feature27', 'feature30', 'feature33', 'feature35', 'feature44', 'feature46', 'feature59', 'feature76', 'feature97', 'feature105']
df_group1=orig_df[features_group]

for column in df_group1.columns:
    if column != 'sample_id' and column != 'label':
        print("当前正在处理",column)
        # 先分别处理 label 0 和 1 的数据
        df_label_0_1 = df_group1[(df_group1['label'] == 0) | (df_group1['label'] == 1)]
        Q1 = df_label_0_1[column].quantile(0.25)
        Q3 = df_label_0_1[column].quantile(0.75)
        IQR = Q3 - Q1
        df_label_0_1[column] = df_label_0_1[column].where(((df_label_0_1[column] >= (Q1 - 1.5 * IQR)) & (df_label_0_1[column] <= (Q3 + 1.5 * IQR))), np.nan)

        # 对于其他类别的数据，直接保留
        df_other_labels = df_group1[(df_group1['label'] != 0) & (df_group1['label'] != 1)]
        
        # 将两部分数据合并为最终的清理后数据
        cleaned_df = pd.concat([df_label_0_1, df_other_labels])

df_group1=cleaned_df


# In[377]:


df_group1


# ## 第二步，训练集knn填补缺失值

# In[378]:


from sklearn.impute import KNNImputer

# Create imputer
imputer = KNNImputer(n_neighbors=5)

# Only select columns that should be imputed
cols_to_impute = [col for col in df_group1.columns if col != 'sample_id' and col != 'label']
print(df_group1['label'].isnull().any())
# Perform imputation on the dataframe
df_group1[cols_to_impute] = imputer.fit_transform(df_group1[cols_to_impute])

# The dataframe 'cleaned_df' now have missing values filled with KNN imputation
df_group1


# ## 第三步，选择模型进行训练

# In[379]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Separate features and target
X = df_group1.drop(columns=['label', 'sample_id'])
y = df_group1['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42) # 0.25 x 0.8 = 0.2

# Create a LGBM classifier instance
clf = lgb.LGBMClassifier()

# Train the model with early stopping
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])

# Predict the test set results
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))


# ## 第四步、K折交叉验证，画出混淆矩阵，查看得分

# In[380]:


y_pred = clf.predict(X_test)
#Print classification report
print(classification_report(y_test, y_pred))

# # Print confusion matrix
# print(confusion_matrix(y_test, y_pred))


# 果然，模型在0、1、2的分类上比较糟糕，我们想办法解决这个问题。不过首先看看其在官方验证集上的泛化能力。

# ## 第五步、加载验证集，查看效果。

# In[381]:


val_df = pd.read_csv('validate_1000.csv')
val_df = val_df[features_group]

from sklearn.impute import KNNImputer

# Select samples with label 2
val_df_label_2 = val_df[val_df['label'] == 2]

# Define IQR
Q1 = val_df_label_2['feature2'].quantile(0.25)
Q3 = val_df_label_2['feature2'].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = val_df_label_2[(val_df_label_2['feature2'] < Q1 - 1.5*IQR) | (val_df_label_2['feature2'] > Q3 + 1.5*IQR)]

# Replace outliers with median
val_df_label_2.loc[outliers.index, 'feature2'] = val_df_label_2['feature2'].median()

# Merge the treated samples back to the original dataframe
val_df[val_df['label'] == 2] = val_df_label_2

# Create KNN imputer
imputer = KNNImputer(n_neighbors=6)

# Define columns to be imputed
cols_to_impute = ['feature2', 'feature5', 'feature10', 'feature15', 'feature19', 'feature22', 'feature31', 'feature49', 'feature71', 'feature85']

# Apply KNN imputation
val_df[cols_to_impute] = imputer.fit_transform(val_df[cols_to_impute])

val_df


# In[382]:


# 提取验证集的特征
X_val = val_df.drop(columns=['label', 'sample_id'])

# 提取验证集的标签
y_val = val_df['label']

# 使用训练的模型预测验证集的标签
y_val_pred = clf.predict(X_val)

# 输出分类报告
print(classification_report(y_val, y_val_pred))

# 输出混淆矩阵
print(confusion_matrix(y_val, y_val_pred))


# ## 第六步 开始优化！
# 我们希望进一步分离出0、1、2三类。我们采取混合高斯拟合不同分布的概率密度，通过观察选择出最适合做分类的特征。我们通过样本x的特征值对应在0、1、2三个特征值上的概率，写出相应的规则，对预测结果进行调整。
#     
#     下面是我们选择出的特征和验证集中某一行的相交情况和概率展示。

# In[438]:


from sklearn.mixture import GaussianMixture
import numpy as np

def fit_gmm(data, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))
    return gmm

def predict_proba(gmm, x):
    log_proba = gmm.score_samples(np.array([x]).reshape(-1, 1))  # 计算对数概率密度
    proba = np.exp(log_proba)  # 转换为概率密度
    return proba

def process_features(df, features, new_xs, n_components=2):
    for feature, new_x in zip(features, new_xs):
        # 定义6个类别对应的颜色
        colors = ['red', 'yellow', 'blue', 'green', 'black', 'orange']
        plt.figure(figsize=(10, 6))

        # 拟合每个类别的数据，并计算新数据的概率密度
        gmms = []
        probas = []

        if pd.isna(new_x):
            print(f"{feature} is NaN")
            probas = [1/6] * 6
        else:
            for i in range(3):
                try:
                    data = df[df['label'] == i][feature].dropna().values
                    gmm = fit_gmm(data, n_components)
                    gmms.append(gmm)
                    proba = predict_proba(gmm, new_x)
                    probas.append(proba)

                    # 画出概率密度函数
                    xs = np.linspace(np.min(data), np.max(data), 1000)
                    ys = np.exp(gmm.score_samples(xs.reshape(-1, 1)))
                    plt.plot(xs, ys, color=colors[i])

                    # 标记新数据的位置
                    plt.axvline(new_x, color='gray', linestyle='--')

                except Exception as e:
                    print(f"Error occurred while processing feature {feature} and label {i}: {e}")

        total_proba = np.sum(probas)
        if total_proba == 0:  # 避免分母为0
            probas = [1/6] * 6
        else:
            probas = probas / total_proba
            
        plt.title(f'Probability Density Functions for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Probability Density')
        plt.show()

        # 计算新数据属于每个类别的概率
        for i in range(3):
            print(f'Probability of class {i} for {feature} at value {new_x}: {probas[i]}')

# 选取第行（记住，Python的索引从0开始，所以第k行应该是k-1）
k =56
# 例如，你想选取第10行
selected_row = val_df.iloc[k-1]
print(selected_row['label'])
# 将选取的行转换为列表
new_xs = np.array(selected_row).flatten().tolist() 
features_group3 = ['label','feature2', 'feature5', 'feature10', 'feature15', 'feature19', 'feature13', 'feature14', 'feature27', 'feature30', 'feature33', 'feature35', 'feature44', 'feature46', 'feature59', 'feature76', 'feature97', 'feature105']
df_group2 = cleaned_df[features_group3]

# 使用新的处理函数
process_features(df_group2,features_group3[1:],new_xs[2:])       


# #### 下面修改一下process_features函数，不再画图，让它更方便使用。

# In[412]:


def fit_gmms(df, features, n_components=2):
    gmms = {}
    for feature in features:
        gmms[feature] = []
        for i in range(3):
            data = df[df['label'] == i][feature].dropna().values
            gmm = fit_gmm(data, n_components)
            gmms[feature].append(gmm)
    return gmms

def process_features2(df, gmms, features, new_xs, n_components=2):
    probas_list = []
    for feature, new_x in zip(features, new_xs):
        probas = []

        if pd.isna(new_x):
            print(f"{feature} is NaN")
            probas = [1/3] * 3
        else:
            for i in range(3):
                try:
                    gmm = gmms[feature][i]
                    proba = predict_proba(gmm, new_x)
                    probas.append(proba)
                except Exception as e:
                    print(f"Error occurred while processing feature {feature} and label {i}: {e}")
        
        total_proba = np.sum(probas)
        if total_proba == 0:  # 避免分母为0
            probas = [1/3] * 3
        else:
            probas = probas / total_proba

        probas_list.append(probas)
    
    return probas_list


# #### 回顾一下验证集

# In[411]:


# 提取验证集的特征
X_val = val_df.drop(columns=['label', 'sample_id'])

# 提取验证集的标签
y_val = val_df['label']

# 使用训练的模型预测验证集的标签
y_val_pred = clf.predict(X_val)

# 输出分类报告
print(classification_report(y_val, y_val_pred))

# 输出混淆矩阵
print(confusion_matrix(y_val, y_val_pred))


# ### 定义我们的规则
#     (1)对于那些现在被判定为1的样本，feature5、feature10、feature15、feature19中的probas[1]和probas[0]进行对比，如果probas[0]更大的数量更多，则将当前判定由1类改为0类。如果数量持平或者probas[1]大的数量更多，则保持为1。规则1）优先执行.
#     (2)对于当前被判为2的样本，如果feature2、feature10，feature15、feature59、feature76所有的概率probas[2]都各自没有超过0.5，则将这个标签置为2，否则置为1。规则2）需要再规则1）完成后再进行.
#     (3)对于当前判别为0的样本，feature19、feature5、feature15的probas[2]与probas[0]进行对比，如果probas[2]更大的数量更多，则将当前0改为2，否则不变
#     

# In[473]:


# 定义规则1和规则2对应的特征
rule1_features = ['feature5', 'feature10', 'feature15', 'feature19','feature59','feature76']
rule2_features = ['feature2', 'feature10', 'feature15', 'feature59', 'feature76']
# 添加规则3的特征
rule3_features = ['feature19', 'feature5', 'feature15', 'feature13', 'feature14', 'feature33', 'feature97', 'feature105']

df_group2

# 拟合GMMs
gmms_rule1 = fit_gmms(df_group2, rule1_features)
gmms_rule2 = fit_gmms(df_group2, rule2_features)
gmms_rule3 = fit_gmms(df_group2, rule3_features)
# 初始化一个数组来存储调整后的预测值
corrected_preds = y_val_pred.copy()

from sklearn.metrics import confusion_matrix

# 执行规则1
indices = np.where(y_val_pred == 1)[0]
selected_samples = val_df.iloc[indices]
for idx, sample in selected_samples.iterrows():
    new_xs = [sample[feature] for feature in df_group2.columns]
    rule1_probas = process_features2(df_group2, gmms_rule1, rule1_features, [new_xs[df_group2.columns.tolist().index(feature)] for feature in rule1_features])
    if sum(proba[0] > proba[1] for proba in rule1_probas) >= len(rule1_probas) / 2:
        corrected_preds[idx] = 0

print("Classification report after applying Rule 1:")
print(classification_report(y_val, corrected_preds))
print("Confusion matrix after applying Rule 1:")
print(confusion_matrix(y_val, corrected_preds))

# 执行规则2
indices = np.where(y_val_pred == 2)[0]
selected_samples = val_df.iloc[indices]
for idx, sample in selected_samples.iterrows():
    new_xs = [sample[feature] for feature in df_group2.columns]
    rule2_probas = process_features2(df_group2, gmms_rule2, rule2_features, [new_xs[df_group2.columns.tolist().index(feature)] for feature in rule2_features])
    if all(proba[2] < 0.705 for proba in rule2_probas):
        corrected_preds[idx] = 1

print("Classification report after applying Rule 2:")
print(classification_report(y_val, corrected_preds))
print("Confusion matrix after applying Rule 2:")
print(confusion_matrix(y_val, corrected_preds))




# 执行规则3
indices = np.where(corrected_preds == 0)[0]
selected_samples = val_df.iloc[indices]
for idx, sample in selected_samples.iterrows():
    new_xs = [sample[feature] for feature in df_group2.columns]
    rule3_probas = process_features2(df_group2, gmms_rule3, rule3_features, [new_xs[df_group2.columns.tolist().index(feature)] for feature in rule3_features])
    if sum(proba[2] > proba[0] for proba in rule3_probas) >= len(rule3_probas) / 2:
        corrected_preds[idx] = 2

print("Classification report after applying Rule 3:")
print(classification_report(y_val, corrected_preds))
print("Confusion matrix after applying Rule 3:")
print(confusion_matrix(y_val, corrected_preds))


# ### 看看测试集

# In[453]:


test_df=pd.read_csv('test_2000_x.csv')


# In[465]:


# 提取所需的特征
features_group=['sample_id', 'label', 'feature2','feature5', 'feature10', 'feature15', 'feature19', 'feature22', 'feature31', 'feature49', 'feature71', 'feature85','feature13', 'feature14', 'feature27', 'feature30', 'feature33', 'feature35', 'feature44', 'feature46', 'feature59', 'feature76', 'feature97', 'feature105']
features_group.remove('label')
feature_test=features_group

test_df_selected = test_df[feature_test]
# 使用KNN填充缺失值
imputer = KNNImputer(n_neighbors=5)
test_df_imputed = imputer.fit_transform(test_df_selected)

# 转换回 DataFrame
test_df_imputed = pd.DataFrame(test_df_imputed, columns=test_df_selected.columns)
test_df=test_df_imputed


# In[466]:


# 再次使用训练的模型预测测试集的标签
X_test = test_df_imputed.drop(columns=['sample_id'])
y_test_pred = clf.predict(X_test)


# In[472]:


# 输出模型预测的各个类的数量
unique, counts = np.unique(y_test_pred, return_counts=True)
print("Predicted class counts by model:")
print(dict(zip(unique, counts)))

# 定位到y_test_pred中所有被分类为0、1、2的样本在test_df中的位置
indices = np.where(np.isin(y_test_pred, [0, 1, 2]))[0]

# 将被预测为0，1，2的标签的样本提取出来
selected_samples = test_df.iloc[indices]

# 初始化一个数组来存储调整后的预测值
corrected_preds = y_test_pred.copy()

# 对每个样本进行处理，应用规则
for idx, sample in selected_samples.iterrows():
    # 生成new_xs，确保元素的顺序与features_group中的顺序一致
    new_xs = [sample[feature] for feature in features_group]

    # 应用规则1
    if corrected_preds[idx] == 1:
        rule1_probas = process_features2(df_group2, gmms_rule1, rule1_features, [new_xs[features_group.index(feature)] for feature in rule1_features])
        if sum(proba[0] > proba[1] for proba in rule1_probas) >= len(rule1_probas) / 2:
            corrected_preds[idx] = 0

    # 应用规则2
    if corrected_preds[idx] == 2:
        rule2_probas = process_features2(df_group2, gmms_rule2, rule2_features, [new_xs[features_group.index(feature)] for feature in rule2_features])
        if all(proba[2] < 0.7 for proba in rule2_probas):
            corrected_preds[idx] = 1

    # 应用规则3
    if corrected_preds[idx] == 0:
        rule3_probas = process_features2(df_group2, gmms_rule3, rule3_features, [new_xs[features_group.index(feature)] for feature in rule3_features])
        if sum(proba[2] > proba[0] for proba in rule3_probas) >= len(rule3_probas) / 2:
            corrected_preds[idx] = 2


# 输出调整后的各个类的数量
unique, counts = np.unique(corrected_preds, return_counts=True)
print("Predicted class counts after applying rules:")
print(dict(zip(unique, counts)))


# In[ ]:




