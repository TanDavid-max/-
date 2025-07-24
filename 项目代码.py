import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot
from xgboost import plot_importance
from easygraphics.dialog import show_objects as show
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import tree
from turtle import shape
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#数据导入
train = pd.read_csv(r'C:\Users\86138\Desktop\house_price\train.csv')
test = pd.read_csv(r'C:\Users\86138\Desktop\house_price\test.csv')
train_id = train['Id']
test_id = test['Id']
train.drop(columns='Id', inplace=True)
test.drop(columns='Id', inplace=True)
all_data = pd.concat([train, test], axis=0, ignore_index=True)
sale_price = train['SalePrice']
all_data.drop(columns='SalePrice', inplace=True)
print(all_data.info())
sum(train.isna().sum())

#数据预处理
#统计缺失值的数量
def calc_mis_val(df):
    cols = df.columns
    mis_val = df.isnull().sum()
    mis_val_pct = round(100 * mis_val / df.shape[0], 2)
    mis_val_df = pd.DataFrame({'mis_val':mis_val, 'mis_val_pct(%)':mis_val_pct}, index=cols)
    mis_val_df = mis_val_df[mis_val_df['mis_val'] != 0].sort_values('mis_val', ascending=False)
    print('总列数：', df.shape[1])
    print('含缺失值列数：', mis_val_df.shape[0])
    return mis_val_df
all_mis_val = calc_mis_val(all_data)
print(all_mis_val)

#缺失值的处理函数
def process_missing(df):
    #缺少值过多，需要去除
    df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'])
    #数值型缺失值用0代替，类别型用None代替
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[col] = df[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    objects = []
    for i in df.columns:
        if df[i].dtype == object:
            objects.append(i)
    df.update(df[objects].fillna('None'))
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numeric.append(i)
    df.update(df[numeric].fillna(0))
    return df

#检查缺失值是否存在
train = process_missing(train)
test = process_missing(test)
print(train.isnull().sum().value_counts())
print(test.isnull().sum().value_counts())
all_data = pd.concat([train, test], axis=0, ignore_index=True)
all_data.drop(columns='SalePrice', inplace=True)
print(all_data.isnull().sum().value_counts())

#数据相关性分析
plt.figure(figsize=(20, 16))
train_corr = train.corr()
sns.heatmap(train_corr, square=True, vmax=0.8, cmap='YlGnBu')
plt.show()
train_corr = train.corr()
plt.figure(figsize=(10, 10))
top_cols = train_corr['SalePrice'].nlargest(10).index
train_corr_top = train.loc[:, top_cols].corr()
sns.heatmap(train_corr_top, annot=True, square=True,
            fmt='.2f', cmap='hot_r', vmax=0.8)
plt.show()

#重要变量散点图
fig = plt.figure(figsize=(16,20))
cols = ['OverallQual','TotalBsmtSF','GrLivArea','YearBuilt','GarageCars','FullBath']
for col in cols:
    ax = fig.add_subplot(2, 3, cols.index(col)+1)
    ax.scatter(train[col], train['SalePrice'])
    ax.set_xlabel(col)
plt.show()
outlier1 = train[(train['OverallQual']==4) & (train['SalePrice']>200000)].index.tolist()
outlier2 = train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index.tolist()
outlier3 = train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index.tolist()
outlier4 = train[(train['YearBuilt']<1900) & (train['SalePrice']>400000)].index.tolist()
outliers = outlier1 + outlier2 + outlier3 + outlier4
outliers = list(set(outliers))
print('离群点个数为{}，其索引为{}'.format(len(outliers), outliers))
all_data.drop(index=outliers, inplace=True)
sale_price.drop(index=outliers, inplace=True)
#完工质量与材料的饼状图
'''
Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10 = 0,0,0,0,0,0,0,0,0,0
for i in all_data['OverallQual']:
    if i == 1:
        Q1 += 1
    elif i == 2:
        Q2 += 1
    elif i == 3:
        Q3 += 1
    elif i == 4:
        Q4 += 1
    elif i == 5:
        Q5 += 1
    elif i == 6:
        Q6 += 1
    elif i == 7:
        Q7 += 1
    elif i == 8:
        Q8 += 1
    elif i == 9:
        Q9 += 1
    elif i == 10:
        Q10 += 1
X = np.array([Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10])
plt.pie(X,labels = ['Very Poor','Poor','Fair','Below Average','Average','Above Average',
                          'Good','Very Good','Excellent','Very Excellent'], autopct='%3.1f%%')
plt.title('完工质量与材料各层次占比')
plt.show()
'''
# 顺序特征编码
def order_coding(col):
    if col == 'Ex':
        code = 0
    elif col == 'Gd':
        code = 1
    elif col == 'TA':
        code = 2
    elif col == 'Fa':
        code = 3
    elif col == 'Po':
        code = 4
    else:
        code = 5
    return code
order_cols = ['BsmtCond','BsmtQual','ExterCond','ExterQual','FireplaceQu',
              'GarageCond','GarageQual','HeatingQC','KitchenQual']
for order_col in order_cols:
    all_data[order_col] = all_data[order_col].apply(order_coding).astype(int)

#部分特征转化为字符串
all_data['MSSubClass'] = all_data['MSSubClass'].astype(object)
all_data['YrSold'] = all_data['YrSold'].astype(object)
all_data['MoSold'] = all_data['MoSold'].astype(object)
#时间序列数据进行LabelEncoder编码
time_cols = ['GarageYrBlt','YearBuilt','YearRemodAdd','YrSold']
for time_col in time_cols:
    all_data[time_col] = LabelEncoder().fit_transform(all_data[time_col])
numeric_df = all_data.select_dtypes(['float64','int32','int64'])
numeric_cols = numeric_df.columns.tolist()

# 计算各数值型特征的偏度
skewed_cols = all_data[numeric_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_df = pd.DataFrame({'skew':skewed_cols})
print(skewed_df)
# 对偏度绝对值大于1的特征进行对数变换
skew_cols = skewed_df[skewed_df['skew'].abs()>1].index.tolist()
for col in skew_cols:
    all_data[col] = np.log1p(all_data[col])
# SalePrice属于偏态分布
sns.distplot(sale_price)
fig = plt.figure()
res = stats.probplot(sale_price, plot=plt)
plt.show()
# 对其进行对数变换并绘制分布图
sale_price = np.log1p(sale_price)
sns.distplot(sale_price)
fig = plt.figure()
res = stats.probplot(sale_price, plot=plt)
plt.show()

#剩余文本类型进行编码
all_data = pd.get_dummies(all_data)
all_data.info()

# 还原训练集和测试集
clean_train = all_data.iloc[:1456, :]
clean_test = all_data.iloc[1456:, :]
# 加上去除离群点后的标签列
clean_train = pd.concat([clean_train, sale_price], axis=1)
print('处理后的训练集大小：', clean_train.shape)
print('处理后的测试集大小：', clean_test.shape)

#拆分train_data
X=clean_train.drop("SalePrice",axis=1)
Y= clean_train["SalePrice"]
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
result=[]
#定义绘制拟合度的曲线
def plot_learning_curves(model,X_train,X_test,Y_train,Y_test):
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], Y_train[:m])
        Y_train_predict = model.predict(X_train[:m])
        Y_test_predict = model.predict(X_test)
        train_errors.append(mean_squared_error(Y_train_predict, Y_train[:m]))
        test_errors.append(mean_squared_error(Y_test_predict, Y_test))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="test")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.show()

#线性回归预测得分
linearmodel=LinearRegression()
linearmodel.fit(X_train,Y_train)
score1=["Linear Regression", linearmodel.score(X_train,Y_train),linearmodel.score(X_test,Y_test),
       abs(linearmodel.score(X_train,Y_train)-linearmodel.score(X_test,Y_test))]
result.append(score1)
print(result)
result1 = linearmodel.predict(X_test)
score = linearmodel.score(X_test,Y_test)
plt.figure()
plt.plot(np.arange(437), Y_test, "go-", label="True Value")
plt.plot(np.arange(437), result1, "ro-", label="Predict Value")
plt.title(f"LinearRegression---score:{score}")
plt.legend(loc="best")
plt.show()
#plot_learning_curves(linearmodel,X_train,X_test,Y_train,Y_test)

#随机森林回归模型
#随机森林回归模型调参
ScoreAll = []
for i in range(10, 200, 1):  # criterion = 'entropy'
    DT = RandomForestRegressor(n_estimators=i, random_state=66)
    score = cross_val_score(DT, X_train, Y_train, cv=10).mean()
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
max_score = np.where(ScoreAll == np.max(ScoreAll[:, 1]))[0][0] 
print("最优参数以及最高得分:", ScoreAll[max_score])
plt.figure(figsize=[20, 5])
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()
ScoreAll = []  
for i in range(10,30,3):  
    DT = RandomForestRegressor(n_estimators = 137,random_state = 66,max_depth =i)
    score = cross_val_score(DT,X_train,Y_train,cv=10).mean()
    ScoreAll.append([i,score])
ScoreAll = np.array(ScoreAll)    
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] 
print("最优参数以及最高得分:",ScoreAll[max_score])    
plt.figure(figsize=[20,5])  
plt.plot(ScoreAll[:,0],ScoreAll[:,1])  
plt.show()
ScoreAll = []  
for i in range(2,10,1):  
	DT = RandomForestRegressor(n_estimators = 137,random_state = 66,max_depth =16,min_samples_split =i) 
	score = cross_val_score(DT,X_train,Y_train,cv=10).mean()  
	ScoreAll.append([i,score])  
ScoreAll = np.array(ScoreAll) 
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] 
print("最优参数以及最高得分:",ScoreAll[max_score])    
plt.figure(figsize=[20,5])  
plt.plot(ScoreAll[:,0],ScoreAll[:,1])  
plt.show()
ScoreAll = []  
for i in range(2,10,1):  
	DT = RandomForestRegressor(n_estimators = 137,random_state = 66,max_depth =16,
	min_samples_leaf = i,min_samples_split = 2,) 
	score = cross_val_score(DT,X_train,Y_train,cv=10).mean()  
	ScoreAll.append([i,score])  
ScoreAll = np.array(ScoreAll)  
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] 
print("最优参数以及最高得分:",ScoreAll[max_score])    
plt.figure(figsize=[20,5])  
plt.plot(ScoreAll[:,0],ScoreAll[:,1])  
plt.show()
grid = {
'max_features':np.arange(0.1, 1),  
'min_samples_leaf':np.arange(5,15),  
'min_samples_split':np.arange(2,10),  
}  
rfc = RandomForestRegressor(random_state=66,n_estimators = 137,max_depth = 16 )  
GS = GridSearchCV(rfc,param_grid,cv=10)  
GS.fit(X_train,Y_train)  
print(GS.best_params_)  
print(GS.best_score_)

#随机森林回归建模
rmodel=RandomForestRegressor(n_estimators=137, bootstrap=True, random_state=66,
                             max_depth = 16, min_samples_leaf =1 ,min_samples_split =2)
rmodel.fit(X_train,Y_train)
score1=["Random Regression", rmodel.score(X_train,Y_train),rmodel.score(X_test,Y_test),
        abs(rmodel.score(X_train,Y_train)-rmodel.score(X_test,Y_test))]
result.append(score1)
print(result)
result2 = rmodel.predict(X_test)
score = rmodel.score(X_test,Y_test)
plt.figure()
plt.plot(np.arange(437), Y_test, "go-", label="True Value")
plt.plot(np.arange(437), result2, "ro-", label="Predict Value")
plt.title(f"RandomForestryRegression---score:{score}")
plt.legend(loc="best")
plt.show()
plot_learning_curves(rmodel,X_train,X_test,Y_train,Y_test)
#岭回归模型
#参数优化
ScoreAll = []
for i in range(1,20,1):
	DT = Ridge()
	score = cross_val_score(DT,X_train,Y_train,cv=10).mean()
	ScoreAll.append([i,score])
ScoreAll = np.array(ScoreAll)
max_score = np.where(ScoreAll == np.max(ScoreAll[:, 1]))[0][0]
print("最优参数以及最高得分:", ScoreAll[max_score])
plt.figure(figsize=[20, 5])
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()
#模型构建
clf = Ridge(fit_intercept=True, normalize=False, copy_X=True, max_iter=None,
            solver='auto', random_state=None,alpha=11)
clf.fit(X_train,Y_train)
score1=["Ridge Regression", clf.score(X_train,Y_train),clf.score(X_test,Y_test),
        abs(clf.score(X_train,Y_train)-clf.score(X_test,Y_test))]
result.append(score1)
print(result)
result2 = clf.predict(X_test)
score = clf.score(X_test,Y_test)
plt.figure()
plt.plot(np.arange(437), Y_test, "go-", label="True Value")
plt.plot(np.arange(437), result2, "ro-", label="Predict Value")
plt.title(f"RidgeRegression---score:{score}")
plt.legend(loc="best")
plt.show()
plot_learning_curves(clf,X_train,X_test,Y_train,Y_test)
#进行预测
prediction=np.expm1(clf.predict(clean_test))
print(prediction)
output = pd.DataFrame({'Id': test_id,
                       'SalePrice': prediction})
output.to_csv('submission.csv', index=False)
