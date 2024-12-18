# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:01:54 2022

@author: yls
"""
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from imblearn.over_sampling import SMOTE

filename = 'D:/src/lwx/eccDNA/04/data/30V80.csv'
raw = pd.read_csv(filename)
# raw.drop(columns=['Unnamed: 0'],inplace=True)

y = raw['label']#.values
X = raw.drop(columns=['label'])#.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)


scaler = preprocessing.StandardScaler().fit(X_train)
X_train.loc[:,:] = scaler.transform(X_train)
X_test.loc[:,:] = scaler.transform(X_test)




"""
建模
"""
linearReg = LinearRegression().fit(X_train, y_train)# 训练
linearReg.score(X_train, y_train), linearReg.score(X_test, y_test)# 测试

lasso = Lasso(alpha=0.001).fit(X_train, y_train)# 训练
lasso.score(X_train, y_train), lasso.score(X_test, y_test)# 测试

logisticReg = LogisticRegression().fit(X_train, y_train)# 训练
logisticReg.score(X_train, y_train), logisticReg.score(X_test, y_test)# 测试

GNB = GaussianNB().fit(X_train, y_train)# 训练
GNB.score(X_train, y_train), GNB.score(X_test, y_test)# 测试

# XGB = XGBClassifier(random_state=1,n_estimators=400,max_depth=4,learning_rate=0.01,subsample=0.3,min_child_weight=3,gamma=5,colsample_bytree=0.9, colsample_bylevel =0.3 ).fit(X_train, y_train)
XGB = XGBClassifier(random_state=1,n_estimators=400,max_depth=5,learning_rate=0.01,subsample=0.8,min_child_weight=1,gamma =5).fit(X_train, y_train)
XGB.score(X_train, y_train), XGB.score(X_test, y_test)


SVM = SVC(gamma='auto',probability=True).fit(X_train, y_train)#训练
# SVM = SVC(kernel='linear',C=0.5,gamma=0.001,probability=True).fit(X_train, y_train)#训练
# SVM = SVC(kernel='sigmoid',C=2,gamma=5).fit(X_train, y_train)#训练
SVM.score(X_train, y_train), SVM.score(X_test, y_test)# 测试

RF =RandomForestClassifier(random_state=23).fit(X_train, y_train)
# RF =RandomForestClassifier(n_estimators=800).fit(X_train, y_train)
RF.score(X_train, y_train), RF.score(X_test, y_test)

# KNN = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)

KNN = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
KNN.score(X_train, y_train), KNN.score(X_test, y_test)
"""
可视化
"""
clf = SVM
list(X_train.columns)



f,ax = plt.subplots(figsize=(15, 10),dpi=300)# 新建一块画布
plot_roc_curve(clf, X_test, y_test, ax=ax,color='#E995C9', name='test roc', lw=5,linestyle= '-.')# clf是已经拟合过的分类器
plot_roc_curve(clf, X_train, y_train, ax=ax, color='#9F9ADF',name='train roc', lw=5,linestyle= '-.')
plt.plot([0, 1], [0, 1], '--', lw=5, color = 'grey')
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve',fontsize=25)
plt.legend(loc='lower right',fontsize=20)
# plt.savefig('D:/src/lwx/eccDNA/05/11.pdf')
plt.show()




perm = PermutationImportance(clf, n_iter=15, random_state=2021)
perm.fit(X_train, y_train)
weight_df = eli5.explain_weights_df(perm, feature_names=list(X_train.columns))
# display(weight_df)
feature_names = weight_df['feature']# list(X_test.columns)

plt.figure(dpi=100,figsize=(15,5))
plt.barh(y=feature_names,width=weight_df['weight'])
plt.xlabel('Permutation Importance',FontSize=12)
plt.ylabel('Feature',FontSize=12)
plt.savefig('D:/src/lwx/eccDNA/05/Importance.pdf')
plt.show()

"""
混淆矩阵
"""
y_pred = clf.predict(X_test)#预测
# 修改图标、字体大小等属性
fig,ax = plt.subplots(dpi=60)
disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
plt.savefig('D:/src/lwx/eccDNA/05/dier.pdf')
plt.show()



f,ax = plt.subplots(dpi=100)# 新建一块画布
plot_roc_curve(clf, X_test, y_test, ax=ax, name='test roc')# clf是已经拟合过的分类器
plot_roc_curve(clf, X_train, y_train, ax=ax, name='train roc')
plt.show()


"""
保存模型
"""
import pickle
with open('svm_model.pickle', 'wb') as f:
    pickle.dump(clf, f)

# 加载模型
with open('svm_model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)


# 使用加载的模型进行预测

newtext = pd.read_csv('./test0325-1.csv')
newtest = newtext.drop(columns=['label'])
newtest.loc[:,:] = scaler.transform(newtest)


result = loaded_model.predict(newtest)
print(result)

"""
95置信区间
"""
###训练集
from sklearn.metrics import roc_auc_score
y_score = clf.predict_proba(X_train)[:, 1]#预测
# y_score = y_score.tolist()
y_label = y_train.reset_index(drop=True)
# y_pred_score = pd.Series(y_score, index=X_test.index)
# 定义 Bootstrap 函数来计算 AUC
def bootstrap_auc(y_true, y_score, n_bootstrap=1000, alpha=0.05):
    aucs = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[indices]
        y_score_sample = y_score[indices]
        auc = roc_auc_score(y_true_sample, y_score_sample)
        aucs.append(auc)
    
    # 计算置信区间
    lower_percentile = (alpha / 2) * 100
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(aucs, lower_percentile)
    upper_bound = np.percentile(aucs, upper_percentile)
    
    return lower_bound, upper_bound

# 计算 AUC 的置信区间`
lower_bound, upper_bound = bootstrap_auc(y_label, y_score)

print("95% 置信区间:", lower_bound, "-", upper_bound)

###测试集
from sklearn.metrics import roc_auc_score
y_score = clf.predict_proba(X_test)[:, 1]#预测
# y_score = y_score.tolist()
y_label = y_test.reset_index(drop=True)
# y_pred_score = pd.Series(y_score, index=X_test.index)
# 定义 Bootstrap 函数来计算 AUC
def bootstrap_auc(y_true, y_score, n_bootstrap=1000, alpha=0.05):
    aucs = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[indices]
        y_score_sample = y_score[indices]
        auc = roc_auc_score(y_true_sample, y_score_sample)
        aucs.append(auc)
    
    # 计算置信区间
    lower_percentile = (alpha / 2) * 100
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(aucs, lower_percentile)
    upper_bound = np.percentile(aucs, upper_percentile)
    
    return lower_bound, upper_bound

# 计算 AUC 的置信区间`
lower_bound, upper_bound = bootstrap_auc(y_label, y_score)

print("95% 置信区间:", lower_bound, "-", upper_bound)






"""
95置信区间模型
"""
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)


n_bootstraps = 1000
rng_seed = 42  # 设置随机数种子以便重现
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # bootstrap采样
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:
        # 确保至少有两个类
        continue

    y_true_resampled = y_test[indices]
    y_prob_resampled = y_probs[indices]

    fpr_resampled, tpr_resampled, _ = roc_curve(y_true_resampled, y_prob_resampled)
    roc_auc_resampled = auc(fpr_resampled, tpr_resampled)
    bootstrapped_scores.append(roc_auc_resampled)

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# 计算95%置信区间
confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

plt.figure()

# 绘制原始ROC曲线
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

# 绘制随机分类的参考线
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

# 绘制置信区间
tprs_bootstrapped = []
base_fpr = np.linspace(0, 1, 101)

for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:
        continue

    y_true_resampled = y_test[indices]
    y_prob_resampled = y_probs[indices]
    fpr_resampled, tpr_resampled, _ = roc_curve(y_true_resampled, y_prob_resampled)
    tpr_resampled = np.interp(base_fpr, fpr_resampled, tpr_resampled)
    tpr_resampled[0] = 0.0
    tprs_bootstrapped.append(tpr_resampled)

tprs_bootstrapped = np.array(tprs_bootstrapped)
mean_tprs = tprs_bootstrapped.mean(axis=0)
std_tprs = tprs_bootstrapped.std(axis=0)

tprs_upper = np.minimum(mean_tprs + 1.96 * std_tprs, 1)
tprs_lower = np.maximum(mean_tprs - 1.96 * std_tprs, 0)

plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='95% CI')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic with 95% CI')
plt.legend(loc="lower right")
plt.show()























