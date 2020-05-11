# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:47:09 2020

@author: Chandramouli
"""

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.drop(['SalePrice'], axis=1)
y_train = train.SalePrice
x_test = test


total = x_train.isnull().sum().sort_values(ascending=False)#to find no of null values in each  column
total1=x_train.isnull().count().sort_values(ascending=False)
x_train.info()

percent = x_train.isnull().sum()/x_train.isnull().count().sort_values(
        ascending=False)*100#find percentage of missing values
        
#missing_data = pd.concat([total,percent], axis=1)
missing_data = pd.concat([total,percent], axis=1,
                         keys=['Total','Percent'])#keys will give column name
                         
x_train = x_train.drop(missing_data[missing_data.Percent>10].index,
                       axis=1)

x_test = x_test.drop(missing_data[missing_data.Percent>10].index,
                       axis=1)



x_train= x_train.drop(['SaleType','SaleCondition',
                       'Id','MoSold','YrSold'], axis=1)#unwanted data

x_test= x_test.drop(['SaleType','SaleCondition',
                       'Id','MoSold','YrSold'], axis=1)#unwanted data

#fill missing values for train dataset
x_train.dtypes
cat_features = x_train.dtypes[x_train.dtypes=='object'].index#to find out columns with string datatypes #.index prints along with the index
a = x_train[cat_features]
x_train = x_train.drop(a, axis=1)
print(len(a))
a.info()
#x_test.MSZoning=x_test['MSZoning'].fillna(x_test['MSZoning'].mode()[0])#filling NA in MSzoning column using mode
#a=a.fillna(a.mode())
a.info()
a.describe()
a.MasVnrType=a['MasVnrType'].fillna(a['MasVnrType'].mode()[0])
a.info()
a.BsmtQual=a['BsmtQual'].fillna(a['BsmtQual'].mode()[0])
a.BsmtExposure=a['BsmtExposure'].fillna(a['BsmtExposure'].mode()[0])
a.BsmtFinType1=a['BsmtFinType1'].fillna(a['BsmtFinType1'].mode()[0])
a.BsmtFinType2=a['BsmtFinType2'].fillna(a['BsmtFinType2'].mode()[0])
a.GarageType=a['GarageType'].fillna(a['GarageType'].mode()[0])
a.GarageFinish=a['GarageFinish'].fillna(a['GarageFinish'].mode()[0])
a.GarageQual=a['GarageQual'].fillna(a['GarageQual'].mode()[0])
a.GarageCond=a['GarageCond'].fillna(a['GarageCond'].mode()[0])
a.Electrical=a['Electrical'].fillna(a['Electrical'].mode()[0])
a.BsmtCond=a['BsmtCond'].fillna(a['BsmtCond'].mode()[0])

a.info()
#for test dataset
cat_features1 = x_test.dtypes[x_test.dtypes=='object'].index
b = x_test[cat_features1]
x_test = x_test.drop(b, axis=1)
b.MasVnrType=b['MasVnrType'].fillna(b['MasVnrType'].mode()[0])
b.info()
b.BsmtQual=b['BsmtQual'].fillna(b['BsmtQual'].mode()[0])
b.BsmtExposure=b['BsmtExposure'].fillna(b['BsmtExposure'].mode()[0])
b.BsmtFinType1=b['BsmtFinType1'].fillna(b['BsmtFinType1'].mode()[0])
b.BsmtFinType2=b['BsmtFinType2'].fillna(b['BsmtFinType2'].mode()[0])
b.GarageType=b['GarageType'].fillna(b['GarageType'].mode()[0])
b.GarageFinish=b['GarageFinish'].fillna(b['GarageFinish'].mode()[0])
b.GarageQual=b['GarageQual'].fillna(b['GarageQual'].mode()[0])
b.GarageCond=b['GarageCond'].fillna(b['GarageCond'].mode()[0])
b.Electrical=b['Electrical'].fillna(b['Electrical'].mode()[0])
b.BsmtCond=b['BsmtCond'].fillna(b['BsmtCond'].mode()[0])
b.Functional=b['Functional'].fillna(b['Functional'].mode()[0])
b.KitchenQual=b['KitchenQual'].fillna(b['KitchenQual'].mode()[0])
b.BsmtQual=b['BsmtQual'].fillna(b['BsmtQual'].mode()[0])
b.Exterior1st=b['Exterior1st'].fillna(b['Exterior1st'].mode()[0])
b.Exterior2nd=b['Exterior2nd'].fillna(b['Exterior2nd'].mode()[0])
b.Utilities=b['Utilities'].fillna(b['Utilities'].mode()[0])
b.MSZoning=b['MSZoning'].fillna(b['MSZoning'].mode()[0])
b.info()

#for train numerical data
x_train.info()
x_train.MasVnrArea=x_train['MasVnrArea'].fillna(x_train['MasVnrArea'].mean())
x_train.GarageYrBlt=x_train['GarageYrBlt'].fillna(x_train['GarageYrBlt'].mean())
x_train.info()
#x_train1=pd.concat([x_train,a], axis=1)
res_train=pd.get_dummies(a)
x_train1=pd.concat([x_train,res_train], axis=1)


#for test numerical data
x_test.info()
x_test.MasVnrArea=x_test['MasVnrArea'].fillna(x_test['MasVnrArea'].mean())
x_test.BsmtFinSF1=x_test['BsmtFinSF1'].fillna(x_test['BsmtFinSF1'].mean())
x_test.BsmtFinSF2=x_test['BsmtFinSF2'].fillna(x_test['BsmtFinSF2'].mean())
x_test.BsmtUnfSF=x_test['BsmtUnfSF'].fillna(x_test['BsmtUnfSF'].mean())
x_test.TotalBsmtSF=x_test['TotalBsmtSF'].fillna(x_test['TotalBsmtSF'].mean())
x_test.BsmtFullBath=x_test['BsmtFullBath'].fillna(x_test['BsmtFullBath'].mean())
x_test.BsmtHalfBath=x_test['BsmtHalfBath'].fillna(x_test['BsmtHalfBath'].mean())
x_test.GarageYrBlt=x_test['GarageYrBlt'].fillna(x_test['GarageYrBlt'].mean())
x_test.GarageCars=x_test['GarageCars'].fillna(x_test['GarageCars'].mean())
x_test.GarageArea=x_test['GarageArea'].fillna(x_test['GarageArea'].mean())
x_test.info()
#x_test1=pd.concat([x_test,b], axis=1)
res_test=pd.get_dummies(b)
x_test1=pd.concat([x_test,res_test], axis=1)


x_train2=x_train1.iloc[:,:]
x_test2=x_test1.iloc[:,:]
#now we have x_train1 dataset and x_test1 dataset after EDA

set(x_train1)-set(x_test1)#to find out unique columns in train compared to test
x_train1=x_train1.drop(['Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','Electrical_Mix','Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','GarageQual_Ex','Heating_Floor','Heating_OthW','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll', 'Utilities_NoSeWa'],1)

#now x_Train1 and x_Test1 have equal columns
#vif

"""def vif_calc(x):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    x_train1['intercept']=1
    vif=pd.DataFrame()
    vif['variables']=x_train1.columns#adds all columns in x_train1 in vif
    for i in range(0,x_train1.shape[1]):#here [1] is the column no ,no of columns
         vif['variables'][i]=variance_inflation_factor(x_train1.values,i)
    return(vif)
vif=vif_calc(x_train1)
print(vif)
x_train1.info()
for i in range(0,vif.shape[1]):
    vif[i]=vif.drop(vif.max(i))
    vif_drop[i]=vif_calc(vif)"""
    
    
#x_train1=x_train1.drop(['intercept'],1)
from statsmodels.stats.outliers_influence import variance_inflation_factor    

def vif_calc(X):
  import numpy as np
  thresh=3
  cols = x_train1.columns
  variables = np.arange(x_train1.shape[1])
  dropped=True
  while dropped:
       dropped=False
       c = x_train1[cols[variables]].values
       vif = [variance_inflation_factor(c, i) for i in np.arange(c.shape[1])]
       maxloc = vif.index(max(vif))
       if max(vif) > thresh:
        print('dropping \'' + x_train1[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
        variables = np.delete(variables, maxloc)
        dropped=True
  print('Remaining variables:')
  print(x_train1.columns[variables])
  return x_train1[cols[variables]]
vif=vif_calc(x_train1)

#removing all the columns dropped during vif calculation
x_train1_pred=vif
x_test1_pred=x_test1.drop(['BsmtFinSF1','1stFlrSF','MSZoning_C (all)','Street_Grvl','LotShape_IR1','LandContour_Bnk','LotConfig_Corner','LandSlope_Gtl','Neighborhood_Blmngtn','Condition1_Artery','BldgType_1Fam','RoofStyle_Flat'],1)
x_test1_pred=x_test1_pred.drop(['Exterior1st_CBlock','MasVnrType_BrkCmn','ExterQual_Ex','ExterCond_Ex','Foundation_BrkTil','BsmtQual_Ex','BsmtCond_Fa','BsmtExposure_Av','BsmtFinType1_ALQ','BsmtFinType2_ALQ','HeatingQC_Ex'],1)
x_test1_pred=x_test1_pred.drop(['CentralAir_N','KitchenQual_Ex','Functional_Maj1','GarageType_2Types','GarageFinish_Fin','GarageCond_Ex' ,'PavedDrive_N','YearBuilt','YearRemodAdd','GarageYrBlt','Electrical_SBrkr','GarageCond_TA'],1)
x_test1_pred=x_test1_pred.drop(['Utilities_AllPub','RoofMatl_CompShg','ExterCond_TA','Heating_GasA','GarageQual_TA','Condition2_Norm','RoofStyle_Gable','Street_Pave','Exterior2nd_VinylSd','Exterior1st_VinylSd'],1)
x_test1_pred=x_test1_pred.drop(['MSZoning_RL','GrLivArea','HouseStyle_1Story','GarageType_Attchd','Functional_Typ','BsmtFinType2_Unf','OverallQual','KitchenAbvGr','TotRmsAbvGrd','MSSubClass','MasVnrType_None','Exterior2nd_MetalSd'],1) 
x_test1_pred=x_test1_pred.drop(['GarageCars','OverallCond','ExterQual_TA','TotalBsmtSF','BsmtCond_TA','Condition1_Norm','BedroomAbvGr','CentralAir_Y','LandContour_Lvl','Exterior1st_CemntBd','FullBath','PavedDrive_Y' ,'BsmtQual_TA'],1)
x_test1_pred=x_test1_pred.drop(['Exterior1st_HdBoard','KitchenQual_TA','2ndFlrSF','GarageArea','Foundation_PConc','Neighborhood_NAmes','Exterior2nd_Wd Sdng','BsmtExposure_No','BsmtUnfSF','Neighborhood_Somerst','GarageFinish_Unf'],1)
x_test1_pred=x_test1_pred.drop(['LotConfig_Inside','LotArea','Exterior1st_AsbShng','MSZoning_RM','Foundation_CBlock','BsmtFinType1_Unf','BsmtQual_Gd','HouseStyle_2Story','ExterQual_Gd','Exterior1st_Plywood','Exterior2nd_Stucco','Exterior2nd_Brk Cmn','MasVnrType_BrkFace','LotShape_Reg','Fireplaces'],1)

#multiple linear regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train1_pred,y_train)#gives line of best fit
regression.coef_
regression.intercept_

y_pred=regression.predict(x_test1_pred)

#saving as csv
import pandas as pd 
y_pred1=pd.DataFrame(y_pred)
y_pred1.columns=['Predicted value']
y_pred1.to_csv('House prediction results 3.0.csv') 

