#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy.special import boxcox1p
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import csv
import os
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
os.getcwd()

PATH = '/home/william/AI_project/mid_test/Kaggle/mid_test/PM25/'
train = pd.read_csv(PATH+ 'PM25_train.csv',  engine='c')
test = pd.read_csv(PATH + 'PM25_test.csv', engine='c')


train.head()

train.info(verbose=False)

test.head() #test data no sale price


#Need to know the distribution of PM2.5¶
sns.distplot(train['PM2.5'], bins=100, kde=False)
plt.title('Price Distribution')
plt.xlabel('PM2.5')
plt.ylabel('Count')
plt.show()


# from above we could see if outlier, would be found or not
# Use Log and boxcox1p transformation to check which transformation is better then reduce the distance between outlier and mode

sns.distplot(np.log1p(train['PM2.5']), bins=200, kde=False,color='y',label='log1p')
sns.distplot(boxcox1p(train['PM2.5'],0.1), bins=200, kde=False, color='b', label='boxcox1p')
upper = train['PM2.5'].max(axis=0)
lower = train['PM2.5'].min(axis=0)
train_PM25 = np.asarray((train['PM2.5'])/(upper-lower))

sns.distplot(train_PM25,bins=200, kde=False,color='g',label='Max-Min')

# sns.distplot((np.max(train['PM2.5'])-np.min(train['PM2.5']))/(np.max(train['PM2.5'])+np.min(train['PM2.5'])),bins=200, kde=False,color='g',label='Max-Min')
# sns.distplot((np.max(train['PM2,bins=200, kde=False,color='g',label='Max-Min')
plt.title('distribution')
plt.xlabel('PM2.5')
plt.legend(['log1p', 'boxcox1p','Max-Min'])
plt.show()


fig, ax = plt.subplots()
ax.scatter(x = train['Temperature'], y = train['PM2.5'])
plt.ylabel('PM2.5', fontsize=13)
plt.xlabel('Temperature', fontsize=13)
plt.show()


sns.distplot(train['PM2.5'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['PM2.5'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['PM2.5'], plot=plt)
plt.show()



#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
#train["PM2.5"] = np.log1p(train["PM2.5"])

#Check the new distribution 
sns.distplot(np.log1p(train['PM2.5']) , fit=norm);

# Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(np.log1p(train['PM2.5'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('PM2.5 distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(np.log1p(train['PM2.5']), plot=plt)
plt.show()


#No need Id, so we need to drop Id

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['device_id']
test_ID = test['device_id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop(['device_id'], axis = 1, inplace = True)
test.drop(['device_id'], axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

train.head()
test.head()



#Correlation map to see how features are correlated with PM25
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True,annot=True)


from sklearn.model_selection import KFold

categorical_features = []

for dtype, feature in zip(train.dtypes, train.columns):
    if dtype == object:
        categorical_features.append(feature)

categorical_features


from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = False)
train1 = train
test1 = test
#train1 = pd.read_csv(PATH+ 'PM25_train.csv',  engine='c')
train1= train1.rename(columns={train1.columns[2]:'PM25'})
print(train1.head(5))
global_mean = train1['PM25'].mean()

# #df.rename(columns={ df.columns[1]: "your value" })
# ###########################################################
# feature_list =[]
# corr_list =[]
# for f_ in categorical_features:    
    
#     train1['item_target_enc'] = np.nan
#     for tr_ind, val_ind in kf.split(train1):
#         X_tr, X_val = train1.iloc[tr_ind], train1.iloc[val_ind]
#         train1.loc[train1.index[val_ind], 'item_target_enc'] = X_val[f_ ].map(X_tr.groupby(f_ ).PM25.mean())

#     train1['item_target_enc'].fillna(global_mean, inplace = True)
#     encoded_feature = train1['item_target_enc'].values
#     # You will need to compute correlation like that
#     corr = np.corrcoef(train1['PM25'].values,encoded_feature)[0][1]
#     feature_list.append(f_)
#     corr_list.append(corr)
#     corr = np.array(corr)
    
    
    
# combine = pd.DataFrame(corr_list, index=feature_list, columns=['ratio'])
# combine
# #combine = combine.sort_values(by='0' ascending=False)
# #corr_list = (combine['ratio'].index).sort_values(ascending=False)[:30]# encoded_feature[0][1]
# corr_list = sorted(combine['ratio'], reverse=True)
# #corr_list
# corr_ratio = pd.DataFrame({'corr_ratio':corr_list},index=feature_list)
# corr_ratio_y = corr_ratio.iloc[:,0]
# corr_ratio_y_index = corr_ratio_y.index
# corr_ratio_y_value = corr_ratio_y.values

# #corr_ratio[:]
# # corr_ratio.head(30)
# # corr_list = sorted(corr_list)
# #plt.tight_layout()
# plt.subplots(figsize=(15,12))
# plt.xticks(rotation='90')
# plt.tight_layout()
# sns.barplot(corr_ratio_y_index,corr_ratio_y_value,saturation=1.0,capsize=0.1)
# plt.xlabel('feature',fontsize=30)
# plt.ylabel('corr',fontsize=30)
# plt.show()


#combine train and test to the same dataframe for treatment in the same time¶


ntrain = train1.shape[0]
ntest = test1.shape[0]
y_train = train1.PM25.values
all_data = pd.concat((train1, test1)).reset_index(drop=True)
all_data.drop(['PM25'], axis=1, inplace=True)
#all_data = all_data.transpose()
print("all_data size is : {}".format(all_data.shape))


#To treat the y_train(Sale price) by log transformation(np.log1p)

y_train = np.log1p(y_train)
y_train[:8]

all_data.head()

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)



all_data.head()
all_data.describe()

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(3)

#Transforming some numerical variables that are really categorical(轉換一些真正絕對的數值變量)

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# We use the scipy function boxcox1p which computes the Box-Cox transformation of 1+x

# ###Note that setting λ=0 is equivalent to log1p used above for the target variable. ###See this page for more details on Box Cox Transformation as well as the scipy function's page

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])

#Getting dummy categorical features¶

train = all_data[:ntrain]
test = all_data[ntrain:]

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
#import lightgbm as lgb

# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
#                              learning_rate=0.05, max_depth=3, 
#                              min_child_weight=1.7817, n_estimators=100,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                              random_state =7, nthread = -1)

model_xgb = xgb.XGBRegressor(n_estimators=2000)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

ensemble = xgb_pred


sub = pd.DataFrame()
sub['device_id'] = test_ID
sub['pred_pm25'] = ensemble
sub.to_csv('submission_4.csv',index=False)

path = '/home/jovyan/Kaggle/mid_test/PM25/'
result = pd.read_csv(path + 'submission_4.csv', engine='c')
result = result.groupby('device_id').mean()
pd.DataFrame(result)

result.to_csv('/home/jovyan/Kaggle/mid_test/PM25/submission_PM25_1.csv')


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 
    

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


sub = pd.DataFrame()
sub['device_id'] = test_ID
sub['pred_pm25'] = ensemble1
sub.to_csv('submission_3.csv',index=False)


path = '/home/jovyan/Kaggle/mid_test/PM25/'
result = pd.read_csv(path + 'submission_3.csv', engine='c')
result = result.groupby('device_id').mean()
pd.DataFrame(result)

result.to_csv('/home/jovyan/Kaggle/mid_test/PM25/submission_PM25.csv')