import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.neural_network import MLPClassifier
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import pickle
import json


# 读取数据集，处理缺省值，用处理后的结果替换原训练集
def load_dataset(dataset="train"):
    # 加载数据
    df = pd.read_csv('./dataset/train.csv')
    if dataset=="test":
        df_test = pd.read_csv('./dataset/test.csv')

    # 处理缺失值
    for label in range(6):
        for feature in range(107):
            feature_name = f'feature{feature}'
            mean_value = df[df['label'] == label][feature_name].mean()
            if dataset=="train":
                df.loc[(df['label'] == label) & (df[feature_name].isnull()), feature_name] = mean_value
                df.to_csv('./dataset/train.csv', index=False)
            if dataset=="test":
                df_test.loc[(df_test['label'] == label) & (df_test[feature_name].isnull()), feature_name] = mean_value
                df_test.to_csv("./dataset/test.csv", index=False)


# 选出每个label关键的6个特征，并进行整合得到最终模型用到的特征
def key_fea(num=6):
    # 初始一个空的DataFrame用于存储所有故障类型的特征重要性
    df = pd.read_csv('./dataset/train.csv')

    # 获取特征和标签
    X = df.iloc[:, 1:-1]  # 取feature0到feature106作为特征
    y = df['label']  # 取'label'列作为标签

    all_importances = pd.DataFrame(index=X.columns)

    # 使用分层K折交叉验证
    skf = StratifiedKFold(n_splits=5)

    for fault_type in range(6):
        # 对于每个故障类型，我们将该类型标记为1，其他类型标记为0
        y_binary = (y == fault_type).astype(int)

        feature_importances = []

        for train_index, test_index in skf.split(X, y_binary):
            # 训练随机森林分类器
            clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=5, min_samples_split=10)
            clf.fit(X.iloc[train_index], y_binary.iloc[train_index])

            # 获取并存储特征重要性
            feature_importances.append(clf.feature_importances_)

        # 通过所有交叉验证的平均值获取特征重要性
        all_importances[f'FaultType {fault_type}'] = np.mean(feature_importances, axis=0)
        
    # 为每个故障类型选择前6个最重要的特征
    important_features_per_fault = {}
    for fault_type in all_importances.columns:
        top_6_features = all_importances[fault_type].sort_values(ascending=False)[:num]
        important_features_per_fault[fault_type] = list(top_6_features.index)

    # 合并所有的重要特征到一个集合
    selected_features = set()

    for features in important_features_per_fault.values():
        selected_features.update(features)
    
    with open("./model/key_fea.txt", 'w') as f:
        f.write(str(selected_features))

    
def train():
    # 加载数据
    df_train = pd.read_csv('./dataset/train.csv')
    with open("./model/key_fea.txt", 'r') as f:
        selected_features = f.readline()
        selected_features = eval(selected_features)

    # 选择重要的特征
    X_train = df_train[selected_features].values
    y_train = df_train['label'].values

    # 标准化数据
    scaler = StandardScaler()
    x_training_data_final = scaler.fit_transform(X_train)

    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(64,64,64), random_state=42,max_iter=30,verbose=False,learning_rate_init=0.05,learning_rate="adaptive")
    mlp.fit(x_training_data_final, y_train)
    # print("The model is trained!")
    
    model_filename = './model/model.pkl'  # 设定文件名
    pickle.dump(mlp, open(model_filename,'wb'))  # 对模型进行“腌制”
    # print("The model is saved!")


def val(X_test, y_test):
    model = pickle.load(open('./model/model.pkl','rb'))#加载“腌制”好的模型
    # 在测试集上检查性能
    y_test_pre = model.predict(X_test)
    print("f1_score: ", f1_score(y_test, y_test_pre, average='macro'))
    print("accuracy_score: ", accuracy_score(y_test, y_test_pre))
    

def test(type="file"):
    # 加载数据和模型
    df_train = pd.read_csv('./dataset/train.csv')
    with open("./model/key_fea.txt", 'r') as f:
        selected_features = f.readline()
        selected_features = eval(selected_features)
    df_test = pd.read_csv('./dataset/test.csv')
    model = pickle.load(open('./model/model.pkl','rb'))#加载“腌制”好的模型

    # 选择重要的特征
    X_train = df_train[selected_features].values
    X_test = df_test[selected_features].values

    # 标准化数据
    scaler = StandardScaler()
    x_training_data_final = scaler.fit(X_train)
    x_test = scaler.transform(X_test)
    
    res_dict = {}
    for i,label in enumerate(model.predict([x_test])):
        res_dict[i] = label
    json.dump(res_dict, open("./result/result.json",'w'))
    
    
    