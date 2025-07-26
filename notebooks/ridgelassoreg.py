import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv',header=1)
print(df.head())


print(df.info())

## Dataset cleaning
print(df[df.isnull().any(axis=1)])

df.loc[:122,"Region"]=0

df.loc[122:,"Region"]=1

print(df.head())

print(df.info())

df[['Region']] = df[['Region']].astype(int)
df.head()
print(df.isnull().sum())
df = df.dropna().reset_index(drop=True)
print(df.head())
print(df.iloc[[122]])
print(df.columns)
df =df.drop(122).reset_index(drop=True)
df.columns = df.columns.str.strip()
print(df.columns)
 
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)


print(df.info())
print(df.head())

objects=[features for features in df.columns if df[features].dtypes=='O']
for i in objects:
    if i!='Classes':
        df[i]=df[i].astype(float)
    
df_copy= df
print(df.head())

print(df.describe())

df.to_csv('Algerian_forest_fires_cleaned_dataset.csv',index=False)

df = df.drop(['day','month','year'],axis=1)
print(df.head())

print(df['Classes'].value_counts())
df['Classes']= np.where(df['Classes'].str.contains('not fire'),0,1)
print(df['Classes'].value_counts())


df['Classes'].value_counts()

plt.style.use('seaborn-v0_8')
df.hist(bins=50,figsize=(20,15))
plt.show()

print(plt.style.available)

## Percentage for Pie Chart

percentage= df['Classes'].value_counts()
classlabels = ['Fire','Not Fire']
plt.figure(figsize=(12,7))
plt.pie(percentage,labels= classlabels,autopct='%1.1f%%')
plt.title('Pie chart of Classes')
plt.show()


sns.heatmap(df.corr(),annot=True)
plt.show()

## Monthly Fire Analysis

df_copy['Classes'] = np.where(df_copy['Classes'].str.contains('not fire'),'not fire','fire')

dftemp = df.loc[df['Region']==1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df_copy)
plt.xlabel('Number of Fires',weight='bold')
plt.ylabel('Months',weight='bold')
plt.title('Fire Analysis of Sidi-Bel regions',weight='bold')
plt.show()

dftemp = df.loc[df['Region']==0]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df_copy)
plt.xlabel('Number of Fires',weight='bold')
plt.ylabel('Months',weight='bold')
plt.title('Fire Analysis of Sidi-Bel regions',weight='bold')
plt.show()

print(df.head())

## Independent and dependent features

X= df.drop('FWI',axis=1)
y = df['FWI']

## TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)
print(y_train.shape)

print(X_train.corr())

sns.heatmap(X_train.corr(),annot=True)
plt.show()

def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)


    return col_corr

corr_features= correlation(X_train,0.85)

print(corr_features)

## drop 
X_train.drop(corr_features,axis=1,inplace=True)
X_test.drop(corr_features,axis=1,inplace=True)

print(X_train.head())


## Feature Scaling or Standardization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)

## Box plot to understand standard scaler

plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=X_train)
plt.title('X train before scaling')
plt.show()
plt.subplot(1,2,2)
sns.boxplot(data=X_train_scaled)
plt.title('X train after scaling')
plt.show()

## Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

linreg = LinearRegression()
linreg.fit(X_train_scaled,y_train)
y_pred = linreg.predict(X_test_scaled)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2= r2_score(y_test, y_pred)

print('Lin Reg MAE' , mae)
print('Lin Reg MSE',mse)
print('Lin Reg R2 score', r2)
plt.scatter(y_test,y_pred)
plt.show()

## Lasso Regression
from sklearn.linear_model import Lasso

lasso =Lasso()
lasso.fit(X_train_scaled,y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

mae = mean_absolute_error(y_test,y_pred_lasso)
mse = mean_squared_error(y_test,y_pred_lasso)
r2= r2_score(y_test, y_pred_lasso)

print('Lasso MAE' , mae)
print('Lasso MSE',mse)
print('Lasso R2 score', r2)
plt.scatter(y_test,y_pred_lasso)
plt.show()

##  Cross validation Lasso
from sklearn.linear_model import LassoCV

lassocv =LassoCV()
lassocv.fit(X_train_scaled,y_train)
y_pred_lassocv = lassocv.predict(X_test_scaled)

mae = mean_absolute_error(y_test,y_pred_lassocv)
mse = mean_squared_error(y_test,y_pred_lasso)
r2= r2_score(y_test, y_pred_lassocv)

print('Lasso CV MAE' , mae)
print('Lasso CV MSE',mse)
print('Lasso CV R2 score', r2)
plt.scatter(y_test,y_pred_lassocv)
plt.show()

## Ridge Regression
from sklearn.linear_model import Ridge

ridge =Ridge()
ridge.fit(X_train_scaled,y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mae = mean_absolute_error(y_test,y_pred_ridge)
mse = mean_squared_error(y_test,y_pred_ridge)
r2= r2_score(y_test, y_pred_ridge)

print('Ridge MAE' , mae)
print('Ridge MSE',mse)
print('Ridge R2 score', r2)
plt.scatter(y_test,y_pred_ridge)
plt.show()

## Ridge Regression CV
from sklearn.linear_model import RidgeCV

ridgecv =RidgeCV(cv=5)
ridgecv.fit(X_train_scaled,y_train)
y_pred_ridgecv = ridgecv.predict(X_test_scaled)

mae = mean_absolute_error(y_test,y_pred_ridgecv)
mse = mean_squared_error(y_test,y_pred_ridgecv)
r2= r2_score(y_test, y_pred_ridgecv)

print('Ridge CV MAE' , mae)
print('Ridge CV MSE',mse)
print('Ridge CV R2 score', r2)
plt.scatter(y_test,y_pred_ridgecv)
plt.show()

## ElasticNet Regression
from sklearn.linear_model import ElasticNet

elastic =ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred_elastic = elastic.predict(X_test_scaled)

mae = mean_absolute_error(y_test,y_pred_elastic)
mse = mean_squared_error(y_test,y_pred_elastic)
r2= r2_score(y_test, y_pred_elastic)

print('Elastic Net MAE' , mae)
print('Elastic Net MSE',mse)
print('Elastic Net R2 score', r2)
plt.scatter(y_test,y_pred_elastic)
plt.show()


## ElasticNetCV Regression
from sklearn.linear_model import ElasticNetCV

elasticv =ElasticNetCV()
elasticv.fit(X_train_scaled,y_train)
y_pred_elasticv = elasticv.predict(X_test_scaled)

mae = mean_absolute_error(y_test,y_pred_elasticv)
mse = mean_squared_error(y_test,y_pred_elasticv)
r2= r2_score(y_test, y_pred_elasticv)

print('Elastic Net CV MAE' , mae)
print('Elastic Net CV MSE',mse)
print('Elastic Net CV R2 score', r2)
plt.scatter(y_test,y_pred_elasticv)
plt.show()


## pickle the machine learning models , preprocessing models standardscaler

import pickle 

pickle.dump(scaler,open('sclaer.pkl','wb'))
pickle.dump(ridge,open('ridge.pkl','wb'))

## ML Project Lifecycle




