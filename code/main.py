import os
import pprint
import pickle
import random
import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import time

####数据集提取
#打开pkl文件
def load_compact_pkl_dataset(dataset_path,dataset_name):
    with open(dataset_path+dataset_name+'.pkl','rb') as f:
        data = pickle.load(f)
    return data

#信号展平打乱处理 XGBoost 要求输入数据是二维数组(样本数, 特征数)
def data_for_xgb(signals):

    num_tx = signals.shape[0]
    num_samples_per_tx = signals.shape[1]
    X = signals.reshape(-1, 256 * 2)  # (8000,512) 8000条信号 256个信号点的实部虚部展平，按时间点顺序排列
    Y = np.repeat(np.arange(num_tx), num_samples_per_tx)  # 8000个信号的标签
    X, Y = shuffle(X, Y, random_state=42)  # 保持X和Y的数据对齐，打乱它们的顺序
    return X,Y

#读取pkl文件 SingleDay数据集
dataset=load_compact_pkl_dataset('./','SingleDay')

# 检查 dataset 的结构和键
#print("数据集包含的字段：", dataset.keys())  # 查看数据集中有哪些字段 dict_keys(['tx_list', 'rx_list', 'capture_date_list', 'equalized_list', 'max_sig', 'data'])

# 获取 'data' 和索引信息
data = dataset['data']            # 嵌套信号数据
tx_list = dataset['tx_list']      # 发射器列表
rx_list = dataset['rx_list']      # 接收器列表

#print("发射器数量：", len(tx_list))#发射器数量： 28
#print("接收器数量：", len(rx_list))#接收器数量： 10

# 定义索引
# 选择前 10 个发射器和第一个接收器作为域A信号
tx_indices_A = list(range(min(10, len(tx_list))))  # 域A发射器 10个
tx_indices_B=list(range(10,20))

rx_indices =[0,1]  # 第一个接收器
day_index = 0  # 选择第一个日期
eq_index = 0  # 选择均衡化信号，去除真实环境中的信道效应（如多径衰落、干扰、噪声等）（若非均衡化，改为 0）

#提取信号数据
selected_signals=[]
signal_tmp=[]
i=0
for rx_index in rx_indices:
    i+=1#域A本地设备信号，域B收到的来自域A设备的信号
    for tx_index in tx_indices_A:
        signal = data[tx_index][rx_index][day_index][eq_index]
        signal_tmp.append(signal)
    selected_signals.append(signal_tmp)
    signal_tmp = []
    if i==2:#域B本地设备信号
        for tx_index in tx_indices_B:
            signal = data[tx_index][rx_index][day_index][eq_index]
            signal_tmp.append(signal)
        selected_signals.append(signal_tmp)

signal_AA=np.array(selected_signals[0])#域A本地信号
signal_BA=np.array(selected_signals[1])#域B收到的来自域A设备的信号
signal_BB=np.array(selected_signals[2])#域B本地信号


print("域A/B设备数：10台  每台设备800条信号（256个采样点）")
print("数据集提取完成：域A本地设备信号，域B本地设备信号，域B收到的来自域A设备的信号")

#print(signal_BB.shape)#(10, 800, 256, 2)：10个发送器，每个发送器800条信号，每条信号256个采样点，I/Q 信号的实部和虚部


####域A模型训练

X_AA, Y_AA = data_for_xgb(signal_AA)#域A本地设备 信号展平打乱处理

#划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_AA, Y_AA, test_size=0.2)

'''
#超参数设置
# 设置参数搜索范围
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# 创建 XGBoost 分类器
model = XGBClassifier(objective='multi:softmax', num_class=10, random_state=42)

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

# 拟合网格搜索
grid_search.fit(X_train, Y_train)

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)

'''
print("开始训练源域模型...")
# 域A XGBoost 分类器
model_Source = XGBClassifier(
    objective='multi:softmax',  # 多分类任务
    num_class=10,  # 发射器数量
    n_estimators=150,
    max_depth=6,
    eval_metric='mlogloss',     # 多分类损失
    learning_rate=0.1,
    random_state=42
)
# 记录训练源模型的开始时间
start_time_source = time.time()
# 训练模型
model_Source.fit(X_train, Y_train)
# 记录训练源模型的结束时间
end_time_source = time.time()
time_source = end_time_source - start_time_source

print("训练完成，耗时：",time_source,"s")
print()

# 测试模型
y_pred = model_Source.predict(X_test)

#score=model_Souce.score(X_test,Y_test)
#print(score)

#模型准确率
accuracy = accuracy_score(Y_test, y_pred)
print("源域模型测试集分类准确率:",accuracy)

# 分类报告
#print("分类报告:")
#print(classification_report(Y_test, y_pred, target_names=[tx_list[i] for i in tx_indices_A]))#设备真实标签
#print(classification_report(Y_test, y_pred, target_names=[str(i) for i in tx_indices_A]))#类别标签:0-9

# 打印混淆矩阵
print("混淆矩阵AA:")
print(confusion_matrix(Y_test, y_pred))
print()

#混淆矩阵可视化
os.makedirs('./result', exist_ok=True)#创建文件夹
ConfusionMatrixDisplay.from_predictions(Y_test, y_pred)
plt.title("WiSig xgboost confusion matrix for AA")
plt.savefig("./result/" + str(accuracy) + ".png")
#plt.show()



###源模型在域B的表现(未经过微调)
X_BA, Y_BA = data_for_xgb(signal_BA)#域B收到的来自域A设备的信号，展平打乱处理
y_pred_BA1 = model_Source.predict(X_BA)
accuracy_BA = accuracy_score(Y_BA, y_pred_BA1)

print("源域模型跨域后，测试集分类准确率:",accuracy_BA)
print("混淆矩阵BA1:")
print(confusion_matrix(Y_BA, y_pred_BA1))
print()

ConfusionMatrixDisplay.from_predictions(Y_BA, y_pred_BA1)
plt.title("WiSig xgboost confusion matrix for BA1")
plt.savefig("./result/" + str(accuracy_BA) + ".png")
###迁移学习：在域B继续训练
#域B本地所有数据
#X_BB, Y_BB = data_for_xgb(signal_BB)#域A本地设备 信号展平打乱处理

#使用域B收到的部分（少量）数据对源模型进行调整
X_subset, X_subset_test, Y_subset, Y_subset_test= train_test_split(X_BA, Y_BA, test_size=0.1, random_state=42) #1/10的X_BA数据量
#print(X_subset.shape)
#print(Y_subset)

#继续训练
print("开始微调模型...")

target_model = xgb.XGBClassifier(
    n_estimators=10,
    max_depth=5,
    learning_rate=0.2,
    eval_metric='logloss'
)

# 使用源域模型的参数初始化目标域模型

target_model._Booster = model_Source._Booster
# 记录微调模型的开始时间

start_time_finetune = time.time()
target_model.fit(X_subset,Y_subset, xgb_model=model_Source.get_booster())
# 记录微调模型的结束时间
end_time_finetune = time.time()
# 微调模型所用的时间
time_finetune = end_time_finetune - start_time_finetune

print("微调完成,耗时：",time_finetune,"s")
print()

#1.直接训练
#model_Source.fit(X_subset, Y_subset, xgb_model=model_Source.get_booster())

'''
#2.增量训练
dtrain = xgb.DMatrix(X_subset, label=Y_subset)
params = {
    'objective': 'multi:softmax',  
    'num_class': 10,  
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss',
    'seed': 42
}

# 使用 XGBoost 的原生接口进行训练
model_Souce = xgb.train(params, dtrain, num_boost_round=150)
print("训练完成")
#验证微调后模型识别来自域A信号X_BA的准确率

y_pred_BA = model_Source.predict(X_BA)
'''
#准确率
y_pred_BA2 = target_model.predict(X_subset_test)
accuracy_BA2 = accuracy_score(Y_subset_test, y_pred_BA2)
print("微调后分类准确率:", accuracy_BA2)

#打印混淆矩阵
print("混淆矩阵BA2:")
print(confusion_matrix(Y_subset_test, y_pred_BA2))
print()

#混淆矩阵可视化
ConfusionMatrixDisplay.from_predictions(Y_subset_test, y_pred_BA2)
plt.title("WiSig xgboost confusion matrix for BA2")
plt.savefig("./result/" + str(accuracy_BA2) + ".png")

 #保存微调后的模型
print("保存模型...")
target_model.save_model('target_model.ubj')
model_loaded = XGBClassifier()

# 加载已经保存的模型
print("加载模型...")
model_loaded.load_model('target_model.ubj')

# 使用加载的模型进行预测
y_pred_loaded = model_loaded.predict(X_BA)
accuracy_BA_load = accuracy_score(Y_BA, y_pred_loaded)
print("加载模型的分类准确率BA3:", accuracy_BA_load)