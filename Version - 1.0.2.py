# coding: utf-8
# Author: Baijia Chen



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl 


# #__STEP ONE:  Drawing design (Data Input & Check)__




#input data and check
data_input = pd.read_csv('input/nyc-rolling-sales.csv')
data_input.head(5)


# #__STEP TWO:  Digging Foundation  (Data Cleaning & Data Transformation)__

# * We need to delete some columns which is obvious rundunt
# >
# >1.We found that [__'EAST-MENT', 'APARTMENT NUMBER'__] these two columns are empty and we can drop them;
# >
# >2.We found that __'BUILDING CLASS CATEGORY '__ is same as the _'BUILDING CLASS OF TIME SALE'_;
# >
# >3.We know that sum of __'RESIDENTIAL UNITS'__ and __'COMMERCIAL UNIT'__ is __'TOTAL UNIT'__ so we can drop the former two;
# >
# >4.We now need to predict the sale price so we only need to delete the [__'BUILDING CLASS AT PRESENT'__, __'TAX CLASS AT PRESENT'__]
# 
# * We also need to combine some real estate market knowledge
# >
# >1.When we buy the apartment/house the final sale price relies on the __'GROSS SQUARE FEET'__  but not the __'LAND SQUARE FEET'__;
# >
# >2.When we wanna predict one real estate price we only need to focus on which the area(__'BOROUGH'__) and zip ocde(__'ZIP CODE'__) it belongs to, but not the specific ___'ADDRESS'___
# 



# Data Cleaning
data_input.drop(data_input[['Unnamed: 0','EASE-MENT','NEIGHBORHOOD','ADDRESS','BUILDING CLASS CATEGORY','BUILDING CLASS AT PRESENT','RESIDENTIAL UNITS','LAND SQUARE FEET','COMMERCIAL UNITS','TAX CLASS AT PRESENT','APARTMENT NUMBER']], axis=1, inplace=True)
data_input['SALE DATE'] = data_input['SALE DATE'].apply(lambda x : x[0:4])
data_input = data_input.iloc[:,:].replace(' -  ',np.nan)


# * We need to do __data type tranformation__ on some columns:
# >
# >1.We know that __'GROSS SQUARE FEET'__ & __'SALE PRICE'__ must be numerical;
# >
# >2.[__'ZIP CODE','YEAR BUILT','TAX CLASS AT TIME OF SALE'__] should be categories, so we need to transform them to be string which will cause effect from the size of value




# Data type transformation
data_input[['GROSS SQUARE FEET','SALE PRICE']] = data_input[['GROSS SQUARE FEET','SALE PRICE']].astype(float)
data_input[['ZIP CODE','YEAR BUILT','TAX CLASS AT TIME OF SALE']] = data_input[['ZIP CODE','YEAR BUILT','TAX CLASS AT TIME OF SALE']].astype(str)





#check data type
data_input.dtypes





#Seperating Input data into data_train & data_test
data_train = data_input[~pd.isna(data_input['SALE PRICE'])]
data_test = data_input[pd.isna(data_input['SALE PRICE'])]





#Let's show some inforamtion about or training data
data_train.describe()


# * From the data description we found that our target data(__'SALE PRICE'__) is unstable and we need to do something on it to make it look more smoothy !!!!




#Show plot about our target data
get_ipython().magic(u'matplotlib inline')
prices = pd.DataFrame({"Price" :data_train['SALE PRICE']})
prices.hist(figsize = (5,5))


# > The __'SALE PRICE'__ data looks so ugly, we need to make up it to make it smoothy
# >




#We can get rid of the strange sale price rows
abnomal_data = data_train.loc[data_train["SALE PRICE"] < 10000.0]
data_train = data_train[data_train['SALE PRICE'] > 1000.0]
train_sale_mean = data_train['SALE PRICE'].mean()
train_sale_std = data_train['SALE PRICE'].std()
index_list = []
for i in data_train['SALE PRICE']:
    if i - train_sale_mean > 2 * train_sale_std:
        index_list.append(i)
#print min(index_list)
data_train = data_train[data_train['SALE PRICE'] < 26500000.0]

We have just get rid the strange data points but we can not be satisfied with it, since we need to make the data have a 
lower std and relatively perfect data distibuion



# We need to find out the relatively best std and show the relatively smooth data to you
std_list = []
for i in range(1000, 5000, 50):
    data = data_train['SALE PRICE']
    value = data.apply(lambda x: np.log(x % i + 1))
    std_list.append(value.std())
#print std_list.index(min(std_list))

sns.set_palette("hls")  
mpl.rc("figure", figsize=(9, 5))    
sns.distplot(data_train['SALE PRICE'].apply(lambda x : np.log(x % 4350 + 1)))


# >__Wow !!!__ It is beautiful, isn't it ? It looks better than it before.
# >
# >__Keep Calm !!!!!__
# >
# >We still need to see is there something strange will threat our target data and we need to fall away them
# >




# Show the heatmap about the co-realtionship between var and target data
# And we can find there is no more things that we need to get rid of, they are important to us
def fall_away(df):
    dfData = df.corr()
    plt.subplots(figsize=(9, 9)) 
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('./BluesStateRelation.png')
    plt.show()
fall_away(data_train)


# #__STEP THREE: Construction (Data Feature Engineering)__




# We need to get the target data we need to do training and tests
y_train = data_train.pop('SALE PRICE').apply(lambda x : np.log(x % 4350 + 1))
y_test = data_test.pop('SALE PRICE')





# Combining the remaining data whihc we need to use in the future
all_df = pd.concat((data_train, data_test), axis = 0)
all_df.head()




# Double check the data type
all_df.dtypes


# >Here we need to waste time to do a __big project__ to tranform all category data to numerical data
# >
# >However, in order to avoid the trouble comes from the size of numerical we can use __ONE-HOT__!!!
# >
# >Luckily!!! we have __get_dummies__ of __pandas__
# >




#Tansform the data
all_df = pd.get_dummies(all_df)
all_df.head()




#Check the null value and fix them
all_df.isnull().sum().sort_values(ascending=False).head(10)





fill_it = all_df['GROSS SQUARE FEET'].mean()
all_df['GROSS SQUARE FEET'] = all_df['GROSS SQUARE FEET'].fillna(fill_it)



#Double check the null value
all_df.isnull().sum().sum()


# >__Standardized numerical data__ to make each item more smooth which I think will be better to our model fitting !!!!
# >




#standardized the numerical data
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_df.loc[:, numeric_cols].mean()
numeric_col_std = all_df.loc[:, numeric_cols].std()
all_df.loc[:, numeric_cols] = (all_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
all_df.head()


# #__STEP FOUR: Completion (Modeling)__




# varialbles we need to train and test
x_train = all_df.loc[data_train.index]
x_test = all_df.loc[data_test.index]





#check the data has the same dimension
x_train.shape, x_test.shape


# * Applying  __Ridge Regression__ model on the data
# >
# >1. For multi-factor datasets, this model can easily put all vars
# >
# >2. Ridge Regression is  better to avoid the effect from noise point
# >




from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
alphas = np.arange(10,15,0.1)
test_scores1 = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores1.append(np.mean(test_score))





import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error")


# >We can find at the point of 15 the Error gets to the minimal, so the HyperParameter to this model we can choose __15__
# >




# Applying Random Froest to do prediction
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores2 = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores2.append(np.mean(test_score)) 




# Findind the best Hyper Parameter of RF
plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error")


# __#STEP FIVE : Ensembling__




# Using the best parameters on the model
alphas_ = list[alphas]
alpha = alphas_[test_scores1.index(min(test_scores1))]
beta = max_features[test_scores2.index(min(test_scores2))]
ridge = Ridge(alpha)
rf = RandomForestRegressor(n_estimators=500, beta)





#Fiiting the data
ridge.fit(x_train, y_train)
rf.fit(x_train, y_train)
y_ridge = np.expm1(ridge.predict(x_test)) * 4350.0
y_rf = np.expm1(rf.predict(x_test)) * 4350.0




# Show the result of Prediction
y_final = (y_ridge + y_rf) / 2
result_df = pd.DataFrame(data= {'Id' : x_test.index, 'SalePrice': y_final})
result_df.head(10)

#result_df.to_csv('../Result2.csv')

