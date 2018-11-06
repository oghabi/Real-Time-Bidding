import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os


from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy import sparse
import datetime



from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


import xgboost as xgb
from pyfm import pylibfm   # Factorization Machines
from fastFM import als

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)


# To keep randomness the same
np.random.seed(0)


df_train = pd.read_csv('./data/train.csv', encoding="ISO-8859-1")  # Shape is (2430981, 25)
df_cv = pd.read_csv('./data/validation.csv', encoding="ISO-8859-1") # (303925, 25)
# Test set doesn't have click, bid price, pay price
df_test = pd.read_csv('./data/test.csv', encoding="ISO-8859-1") # (303375, 22)


# Merge train and validation set when evaluating bids on the test set (more data better)
# df_train = pd.concat([df_train, df_cv])  # (2734906, 25)


# Shows number of ones and zeros
df_train.click.value_counts()


df_train_majority = df_train[df_train.click==0]
df_train_minority = df_train[df_train.click==1]


###### Downsample the majority class (only for train data) ######## 
w = 0.025 # Negative (class 0) downsampling rate

df_train_majority_downsampled = resample(df_train_majority, 
										replace=False,  # sample without replacement
										n_samples= int(w * df_train_majority.shape[0]),  # downsample to 2 * minority
										random_state=123) # reproducible results
# df_train_0 = df_train_0.sample(frac=0.0075, random_state=42)
 
# Combine minority class with downsampled majority class and then shuffle
df_train = pd.concat([df_train_majority_downsampled, df_train_minority]).sample(frac=1)


def preprocess_df(df):
	# df_test doesn't have the bidprice and payprice
	if 'bidprice' and 'payprice' in df.columns:
		df['bidprice'] = df['bidprice'].apply( lambda x: x/1000.0 )
		df['payprice'] = df['payprice'].apply( lambda x: x/1000.0 )
	df['slotprice'] = df['slotprice'].apply( lambda x: x/1000.0 )
	df['OS'] = df['useragent'].apply( lambda x: x.split('_')[0] )
	df['browser'] = df['useragent'].apply( lambda x: x.split('_')[1] )
	return df


df_train = preprocess_df(df_train)
df_cv = preprocess_df(df_cv)
df_test = preprocess_df(df_test)


# Group by the advertisers
# df_train[['click', 'bidprice', 'payprice', 'advertiser']].groupby(['advertiser']).agg(['sum', 'count'])
# df_train[['click', 'bidprice', 'payprice', 'advertiser']].groupby(['advertiser']).describe()
train_statistics = df_train[['click', 'bidprice', 'payprice', 'advertiser']].groupby(['advertiser']).sum()

# number of impressions
train_statistics['impressions'] = df_train.groupby(['advertiser']).size()

# Calculate CTR, CPM, and eCPC
train_statistics['CTR'] = train_statistics['click']/train_statistics['impressions']

# cost (budget) sum of bidprice or payprice
# CPM is (Cost(Budget)/# Impressions) * 1000   
train_statistics['CPM'] = (train_statistics['payprice']/train_statistics['impressions']) * 1000.0
train_statistics['eCPC'] = train_statistics['payprice']/train_statistics['click']


# For the whole campaign (all the advertisers)
total_stats_train = df_train[['click', 'bidprice', 'payprice']].sum()
total_stats_train['impressions'] = df_train.shape[0]
total_stats_train['CTR'] = total_stats_train['click']/total_stats_train['impressions']
total_stats_train['CPM'] = (total_stats_train['payprice']/total_stats_train['impressions']) * 1000.0
total_stats_train['eCPC'] = total_stats_train['payprice']/total_stats_train['click']


total_stats_cv = df_cv[['click', 'bidprice', 'payprice']].sum()
total_stats_cv['impressions'] = df_train.shape[0]
total_stats_cv['CTR'] = total_stats_cv['click']/total_stats_cv['impressions']
total_stats_cv['CPM'] = (total_stats_cv['payprice']/total_stats_cv['impressions']) * 1000.0
total_stats_cv['eCPC'] = total_stats_cv['payprice']/total_stats_cv['click']


df_train[ (df_train['bidprice'] >= df_train['slotprice']) & (df_train['bidprice'] >= df_train['payprice']) ]

############################ plotting CTR vs various features ##############################
def plot_CTR_vs_Features():
	CTR_vs_Weekday = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'weekday']].groupby(['advertiser', 'weekday']).sum()
	CTR_vs_Weekday['impressions'] = df_train.groupby(['advertiser', 'weekday']).size()
	CTR_vs_Weekday['CTR'] = CTR_vs_Weekday['click']/CTR_vs_Weekday['impressions']
	Plot_CTR_vs_Weekday = CTR_vs_Weekday.unstack('advertiser').loc[:, 'CTR'][[1458, 3358]]
	Plot_CTR_vs_Weekday.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_CTR_vs_Weekday.plot(kind="line")  #kind="bar"
	plt.xlabel('Weekday')
	plt.ylabel('CTR')
	plt.show()
	plt.clf()


	CTR_vs_Hour = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'hour']].groupby(['advertiser', 'hour']).sum()
	CTR_vs_Hour['impressions'] = df_train.groupby(['advertiser', 'hour']).size()
	CTR_vs_Hour['CTR'] = CTR_vs_Hour['click']/CTR_vs_Hour['impressions']
	Plot_CTR_vs_Hour = CTR_vs_Hour.unstack('advertiser').loc[:, 'CTR'][[1458, 3358]]
	Plot_CTR_vs_Hour.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_CTR_vs_Hour.plot(kind="line")  #kind="bar"
	plt.xlabel('Hour')
	plt.ylabel('CTR')
	plt.show()
	plt.clf()


	CTR_vs_OS = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'OS']].groupby(['advertiser', 'OS']).sum()
	CTR_vs_OS['impressions'] = df_train.groupby(['advertiser', 'OS']).size()
	CTR_vs_OS['CTR'] = CTR_vs_OS['click']/CTR_vs_OS['impressions']
	Plot_CTR_vs_OS = CTR_vs_OS.unstack('advertiser').loc[:, 'CTR'][[1458, 3358]]
	Plot_CTR_vs_OS.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_CTR_vs_OS.plot(kind="bar")  #kind="bar"
	plt.xlabel('OS')
	plt.ylabel('CTR')
	plt.show()
	plt.clf()


	CTR_vs_Browser = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'browser']].groupby(['advertiser', 'browser']).sum()
	CTR_vs_Browser['impressions'] = df_train.groupby(['advertiser', 'browser']).size()
	CTR_vs_Browser['CTR'] = CTR_vs_Browser['click']/CTR_vs_Browser['impressions']
	Plot_CTR_vs_Browser = CTR_vs_Browser.unstack('advertiser').loc[:, 'CTR'][[1458,3358]]
	Plot_CTR_vs_Browser.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_CTR_vs_Browser.plot(kind="bar")  #kind="bar"
	plt.xlabel('Browser')
	plt.ylabel('CTR')
	plt.show()
	plt.clf()


	CTR_vs_AdExchange = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'adexchange']].groupby(['advertiser', 'adexchange']).sum()
	CTR_vs_AdExchange['impressions'] = df_train.groupby(['advertiser', 'adexchange']).size()
	CTR_vs_AdExchange['CTR'] = CTR_vs_AdExchange['click']/CTR_vs_AdExchange['impressions']
	Plot_CTR_vs_AdExchange = CTR_vs_AdExchange.unstack('advertiser').loc[:, 'CTR'][[1458,3358]]
	Plot_CTR_vs_AdExchange.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_CTR_vs_AdExchange.plot(kind="bar")  #kind="bar"
	plt.xlabel('Ad Exchange')
	plt.ylabel('CTR')
	plt.show()
	plt.clf()
############################################################################################


########################### plotting Market Price (payprice) vs various features ###########
def plot_payprice_vs_Features():
	payprice_vs_Weekday = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'weekday']].groupby(['advertiser', 'weekday']).sum()
	payprice_vs_Weekday['impressions'] = df_train.groupby(['advertiser', 'weekday']).size()
	payprice_vs_Weekday['market_price'] = (payprice_vs_Weekday['payprice']/payprice_vs_Weekday['impressions']) * 1000.0  # Market price is in CPM 
	Plot_payprice_vs_Weekday = payprice_vs_Weekday.unstack('advertiser').loc[:, 'market_price'][[1458, 3358]]
	Plot_payprice_vs_Weekday.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_payprice_vs_Weekday.plot(kind="line")  #kind="bar"
	plt.xlabel('Weekday')
	plt.ylabel('Pay Price')
	plt.show()
	plt.clf()


	payprice_vs_Hour = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'hour']].groupby(['advertiser', 'hour']).sum()
	payprice_vs_Hour['impressions'] = df_train.groupby(['advertiser', 'hour']).size()
	payprice_vs_Hour['market_price'] = (payprice_vs_Hour['payprice']/payprice_vs_Hour['impressions']) * 1000.0  # Market price is in CPM 
	Plot_payprice_vs_Hour = payprice_vs_Hour.unstack('advertiser').loc[:, 'market_price'][[1458, 3358]]
	Plot_payprice_vs_Hour.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_payprice_vs_Hour.plot(kind="line")  #kind="bar"
	plt.xlabel('Hour')
	plt.ylabel('Pay Price')
	plt.show()
	plt.clf()


	payprice_vs_OS = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'OS']].groupby(['advertiser', 'OS']).sum()
	payprice_vs_OS['impressions'] = df_train.groupby(['advertiser', 'OS']).size()
	payprice_vs_OS['market_price'] = (payprice_vs_OS['payprice']/payprice_vs_OS['impressions']) * 1000.0  # Market price is in CPM 
	Plot_payprice_vs_OS = payprice_vs_OS.unstack('advertiser').loc[:, 'market_price'][[1458, 3358]]
	Plot_payprice_vs_OS.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_payprice_vs_OS.plot(kind="bar")  #kind="bar"
	plt.xlabel('OS')
	plt.ylabel('Pay Price')
	plt.show()
	plt.clf()

	payprice_vs_Browser = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'browser']].groupby(['advertiser', 'browser']).sum()
	payprice_vs_Browser['impressions'] = df_train.groupby(['advertiser', 'browser']).size()
	payprice_vs_Browser['market_price'] = (payprice_vs_Browser['payprice']/payprice_vs_Browser['impressions']) * 1000.0  # Market price is in CPM 
	Plot_payprice_vs_Browser= payprice_vs_Browser.unstack('advertiser').loc[:, 'market_price'][[1458,3358]]
	Plot_payprice_vs_Browser.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_payprice_vs_Browser.plot(kind="bar")  
	plt.xlabel('Browser')
	plt.ylabel('Pay Price')
	plt.show()
	plt.clf()


	payprice_vs_AdExchange = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'adexchange']].groupby(['advertiser', 'adexchange']).sum()
	payprice_vs_AdExchange['impressions'] = df_train.groupby(['advertiser', 'adexchange']).size()
	payprice_vs_AdExchange['market_price'] = (payprice_vs_AdExchange['payprice']/payprice_vs_AdExchange['impressions']) * 1000.0  # Market price is in CPM 
	Plot_payprice_vs_AdExchange = payprice_vs_AdExchange.unstack('advertiser').loc[:, 'market_price'][[1458,3358]]
	Plot_payprice_vs_AdExchange.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_payprice_vs_AdExchange.plot(kind="bar")  
	plt.xlabel('Ad Exchange')
	plt.ylabel('Pay Price')
	plt.show()
	plt.clf()

############################################################################################


############################ plotting eCPC vs various features #############################
def plot_eCPC_vs_Features():
	eCPC_vs_Weekday = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'weekday']].groupby(['advertiser', 'weekday']).sum()
	eCPC_vs_Weekday['impressions'] = df_train.groupby(['advertiser', 'weekday']).size()
	eCPC_vs_Weekday['eCPC'] = (eCPC_vs_Weekday['payprice']/eCPC_vs_Weekday['click']) 
	Plot_eCPC_vs_Weekday = eCPC_vs_Weekday.unstack('advertiser').loc[:, 'eCPC'][[3358]]
	Plot_eCPC_vs_Weekday.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_eCPC_vs_Weekday.plot(kind="bar")  #kind="bar"
	plt.xlabel('Weekday')
	plt.ylabel('eCPC')
	plt.show()
	plt.clf()


	eCPC_vs_Hour = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'hour']].groupby(['advertiser', 'hour']).sum()
	eCPC_vs_Hour['impressions'] = df_train.groupby(['advertiser', 'hour']).size()
	eCPC_vs_Hour['eCPC'] = (eCPC_vs_Hour['payprice']/eCPC_vs_Hour['click']) 
	Plot_eCPC_vs_Hour = eCPC_vs_Hour.unstack('advertiser').loc[:, 'eCPC'][[3358]]
	Plot_eCPC_vs_Hour.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_eCPC_vs_Hour.plot(kind="bar")  #kind="bar"
	plt.xlabel('Hour')
	plt.ylabel('eCPC')
	plt.show()
	plt.clf()


	eCPC_vs_OS = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'OS']].groupby(['advertiser', 'OS']).sum()
	eCPC_vs_OS['impressions'] = df_train.groupby(['advertiser', 'OS']).size()
	eCPC_vs_OS['eCPC'] = (eCPC_vs_OS['payprice']/eCPC_vs_OS['click']) 
	Plot_eCPC_vs_OS = eCPC_vs_OS.unstack('advertiser').loc[:, 'eCPC'][[3358]]
	Plot_eCPC_vs_OS.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_eCPC_vs_OS.plot(kind="bar")  #kind="bar"
	plt.xlabel('OS')
	plt.ylabel('eCPC')
	plt.show()
	plt.clf()


	eCPC_vs_AdExchange = df_train[['click', 'bidprice', 'payprice', 'advertiser', 'adexchange']].groupby(['advertiser', 'adexchange']).sum()
	eCPC_vs_AdExchange['impressions'] = df_train.groupby(['advertiser', 'adexchange']).size()
	eCPC_vs_AdExchange['eCPC'] = (eCPC_vs_AdExchange['payprice']/eCPC_vs_AdExchange['click']) 
	Plot_eCPC_vs_AdExchange = eCPC_vs_AdExchange.unstack('advertiser').loc[:, 'eCPC'][[3358]]
	Plot_eCPC_vs_AdExchange.fillna(0.0, inplace=True)   # Fill Nans with zero
	Plot_eCPC_vs_AdExchange.plot(kind="bar")  
	plt.xlabel('Ad Exchange')
	plt.ylabel('eCPC')
	plt.show()
	plt.clf()
############################################################################################




# Models for the CTR Esimation


# Create feature for the Slot Floor price
def slot_price_bucketing(x):
	floor_price = x*1000.0
	if floor_price >= 0 and floor_price<1:
		return 1
	elif floor_price>= 1 and floor_price<=10:
		return 2
	elif floor_price>= 11 and floor_price<=50:
		return 3
	elif floor_price>= 51 and floor_price<=100:
		return 4
	else: #[101, infinity]
		return 5   


# Add extra feature columns: "slot_price_bucket", "slot_width_height", one column for each user segmentation tag
def extra_features(df):
	df['slot_price_bucket'] = df['slotprice'].apply( slot_price_bucketing )	
	# Create feature for slot width x slot height
	df['slot_width_height'] = df['slotwidth'].apply( lambda x: str(x) ) + "_" + df['slotheight'].apply( lambda x: str(x) )
	# Create feature for user tags (add one column for each tag)
	# return df
	AllTags = np.array(df['usertag'])
	Tags_set = { tag for usertags in AllTags for tag in usertags.strip().split(',')}
	Tags = list(Tags_set)
	for t in Tags:
		df["usertag_" + t] = df['usertag'].apply( lambda x: int( t in x ) )
	Tags_list = [("usertag_" + t) for t in Tags]
	return df, Tags_list


############# Create a dictionary which contains a separate LR model for each advertiser (9 advertisers)########################
advertisers = np.array(df_train['advertiser'])
unique_elements, counts_elements = np.unique(advertisers, return_counts=True)
advertisers = list(unique_elements)
features2 = ['weekday', 'hour', 'OS', 'browser', 'region' , 'slotvisibility', 'slotformat' ,'slot_width_height', 'slot_price_bucket'] 
advertiser_LR_models = {}
advertiser_avg_CTR={}

df_train2_read = pd.read_csv('./data/train.csv', encoding="ISO-8859-1")
df_cv2_read = pd.read_csv('./data/validation.csv', encoding="ISO-8859-1")



df_train_majority = df_train2_read[df_train2_read.click==0]
df_train_minority = df_train2_read[df_train2_read.click==1]

w = 0.025 # Negative (class 0) downsampling rate
df_train_majority_downsampled = resample(df_train_majority, 
										replace=False, # sample without replacement
										n_samples= int(w * df_train_majority.shape[0]),  # downsample to 2 * minority
										random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class and then shuffle
df_train2_read = pd.concat([df_train_majority_downsampled, df_train_minority]).sample(frac=1)



for adv in advertisers:
	df_train2 = df_train2_read[ df_train2_read['advertiser'] == adv ]
	df_cv2 = df_cv2_read[ df_cv2_read['advertiser'] == adv ]
	df_train2 = preprocess_df(df_train2)
	df_cv2 = preprocess_df(df_cv2)
	df_train2, train_usertags2 = extra_features(df_train2)
	df_cv2, cv_usertags2 = extra_features(df_cv2)
	train_data_labels2 = df_train2[features2 + train_usertags2 + ['click']]
	test_data_labels2 = df_cv2[features2 + cv_usertags2 + ['click']]
	train_data2 = train_data_labels2[features2 + train_usertags2].applymap(str)
	test_data2 = test_data_labels2[features2 + cv_usertags2].applymap(str)
	train_encoding2 = pd.get_dummies(train_data2)
	temp_cols2 = list( train_encoding2.columns )
	temp_cols2 = [t for t in temp_cols2 if '_0' not in t or 'usertag' not in t]
	train_encoding2 = train_encoding2[temp_cols2]
	train_encoding_final2 = pd.DataFrame(index=range(0,len(train_encoding2)), columns=train_encoding_cols)
	# Now go through the train_encoding columns and see if they exist in the encoding
	for col in train_encoding_cols:  #It should be train_encoding_cols not train_encoding_cols2 (to have all the features)
		if col in temp_cols2:
			train_encoding_final2[col] = np.array( train_encoding2[col] )
	train_encoding_final2 = train_encoding_final2.fillna(0)
	train_encoding2 = train_encoding_final2
	train_labels2 = np.array(train_data_labels2['click'])
	train_encoding_cols2 = list( train_encoding2.columns )
	test_encoding2 = pd.get_dummies(test_data2)
	temp_cols2 = list( test_encoding2.columns )
	temp_cols2 = [t for t in temp_cols2 if '_0' not in t or 'usertag' not in t]
	test_encoding2 = test_encoding2[temp_cols2]
	test_labels2 = np.array(test_data_labels2['click'])
	test_encoding_cols2 = list( test_encoding2.columns )
	test_encoding_final2 = pd.DataFrame(index=range(0,len(test_encoding2)), columns=train_encoding_cols)   #It should be train_encoding_cols not train_encoding_cols2 (to have all the features)
	# Now go through the train_encoding columns and see if they exist in the encoding
	for col in train_encoding_cols:  #It should be train_encoding_cols not train_encoding_cols2 (to have all the features)
		if col in test_encoding_cols2:
			test_encoding_final2[col] = np.array( test_encoding2[col] )
	test_encoding_final2 = test_encoding_final2.fillna(0)
	model_LR2 = LogisticRegression(C=0.7, penalty="l2", class_weight='balanced' ,max_iter=300)
	train_encoding2 = np.array(train_encoding2)
	test_encoding_final2 = np.array( test_encoding_final2 )
	model_LR2.fit(train_encoding2, train_labels2)
	advertiser_LR_models[ adv ] = model_LR2
	advertiser_avg_CTR[ adv ] = df_train2['click'].sum()/df_train2.shape[0]
	predicted_LR2 = model_LR2.predict(test_encoding_final2)
	print (metrics.classification_report(test_labels2, predicted_LR2))
	print (metrics.confusion_matrix(test_labels2, predicted_LR2))
	print (adv, 'PD AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels2,  model_LR2.predict_proba(test_encoding_final2)[:,1] ))


df_cv['pCTR'] = np.array( [  advertiser_LR_models[df_cv['advertiser'].loc[i]].predict_proba(test_encoding_final[i:i+1])[:,1][0]  for i in range(0, len(df_cv)) ] )
df_cv['avgCTR'] = np.array( [  advertiser_avg_CTR[df_cv['advertiser'].loc[i]] for i in range(0, len(df_cv)) ] )
####################################################################


df_train, train_usertags = extra_features(df_train)
df_cv, cv_usertags = extra_features(df_cv)
df_test, test_usertags = extra_features(df_test)


features = ['weekday', 'hour', 'OS', 'browser', 'region' , 'slotvisibility', 'slotformat' ,'slot_width_height', 'slot_price_bucket']
# features = ['weekday', 'hour', 'OS', 'browser', 'region' , 'city', 'slotvisibility', 'slotformat' ,'slot_width_height', 'slot_price_bucket'] 


train_data_labels = df_train[features + train_usertags + ['click']]
test_data_labels = df_cv[features + cv_usertags + ['click']]	   # This is the validation set
submission_test_data_nolabels = df_test[features + test_usertags]  # No clicks for the actual test set

# Need to convert values to string
train_data = train_data_labels[features + train_usertags].applymap(str)
test_data = test_data_labels[features + cv_usertags].applymap(str)
submission_test_data = submission_test_data_nolabels[features + test_usertags].applymap(str)


# SKlearn dictvectorizer method
# vectorizer = DictVectorizer()
# label_encoder = LabelEncoder()

# train_data_dicts = train_data.to_dict('records')
# sk_train_encoding = vectorizer.fit_transform(train_data_dicts)
# # sk_train_encoding.toarray()
# sk_train_labels = label_encoder.fit_transform( np.array(train_data_labels['click']) )

# test_data_dicts = test_data.to_dict('records')
# sk_test_encoding = vectorizer.transform(test_data_dicts)
# # sk_test_encoding.toarray()
# sk_test_labels = np.array(test_data_labels['click'])


# Pandas Dict Vectorizer Method
train_encoding = pd.get_dummies(train_data)
temp_cols = list( train_encoding.columns )
temp_cols = [t for t in temp_cols if '_0' not in t or 'usertag' not in t]
train_encoding = train_encoding[temp_cols]
train_labels = np.array(train_data_labels['click'])
train_encoding_cols = list( train_encoding.columns )

# Validation Data Encoding
test_encoding = pd.get_dummies(test_data)
temp_cols = list( test_encoding.columns )
temp_cols = [t for t in temp_cols if '_0' not in t or 'usertag' not in t]
test_encoding = test_encoding[temp_cols]
test_labels = np.array(test_data_labels['click'])
test_encoding_cols = list( test_encoding.columns )

test_encoding_final = pd.DataFrame(0, index=range(0,len(test_encoding)), columns=train_encoding_cols)

# Now go through the train_encoding columns and see if they exist in the encoding
for col in train_encoding_cols:
	if col in test_encoding_cols:
		test_encoding_final[col] = np.array( test_encoding[col] )



# Submission Test Encoding
submission_test_encoding = pd.get_dummies(submission_test_data)
temp_cols = list( submission_test_encoding.columns )
temp_cols = [t for t in temp_cols if '_0' not in t or 'usertag' not in t]
submission_test_encoding = submission_test_encoding[temp_cols]
submission_test_encoding_cols = list( submission_test_encoding.columns )

submission_test_encoding_final = pd.DataFrame(0, index=range(0,len(submission_test_encoding)), columns=train_encoding_cols)


# Now go through the train_encoding columns and see if they exist in the encoding
for col in train_encoding_cols:
	if col in submission_test_encoding_cols:
		submission_test_encoding_final[col] = np.array( submission_test_encoding[col] )

# Already intialized with 0 instead of NA
# submission_test_encoding_final = submission_test_encoding_final.fillna(0)

# Save the CSV files to avoid doing the feature extraction again
# train_encoding.to_csv(path_or_buf='./train_encoding.csv')
# test_encoding.to_csv(path_or_buf='./test_encoding.csv')
# df_train.to_csv(path_or_buf='./df_train.csv')
# df_cv.to_csv(path_or_buf='./df_cv.csv')

# pd.DataFrame(train_encoding).to_csv(path_or_buf='./train_encoding.csv')
# pd.DataFrame(test_encoding).to_csv(path_or_buf='./test_encoding.csv')



print ('------- Logistic Regression -------')


train_encoding = np.array(train_encoding)
test_encoding_final = np.array( test_encoding_final )
submission_test_encoding_final = np.array( submission_test_encoding_final )


############## Logistic Regression #############
tuned_parameters = [{'C': np.linspace(start=0.01,stop=1,num=10),"penalty":["l2"], "class_weight":["balanced"]}]
model_LR=GridSearchCV( LogisticRegression(max_iter=500),param_grid=tuned_parameters,scoring="roc_auc",cv=5,n_jobs=-1 )  # This gives better results
model_LR.fit(train_encoding, train_labels)
print("parameters selected: ",model_LR.best_params_, "best score: ", model_LR.best_score_)
predicted_LR = model_LR.predict(test_encoding_final)
print (metrics.classification_report(test_labels, predicted_LR))
print (metrics.confusion_matrix(test_labels, predicted_LR))
print ('PD AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model_LR.predict_proba(test_encoding_final)[:,1] ))
predicted_LR_CTR = model_LR.predict_proba(test_encoding_final)[:,1]


model_LR = LogisticRegression(C=0.01, penalty="l2", class_weight='balanced' ,max_iter=500)
model_LR.fit(train_encoding, train_labels)
predicted_LR = model_LR.predict(test_encoding_final)
print (metrics.classification_report(test_labels, predicted_LR))
print (metrics.confusion_matrix(test_labels, predicted_LR))
print ('PD AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model_LR.predict_proba(test_encoding_final)[:,1] ))


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=600, learning_rate=1, max_depth=5, random_state=0).fit(train_encoding, train_labels)
print ('PD AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  clf.predict_proba(test_encoding_final)[:,1] ))

################# XGBoost #################
param_to_tuned = {'max_depth':[3,6,8],
                  'seed':[1337],
                  'silent':[1],
                  'n_estimators':[500,700,300],
                  'learning_rate':np.linspace(0.01,0.6,6),
                  'objective':['binary:logistic'],
                  'subsample':[0.1,0.8],
                  'colsample_bytree':[0.7],
                  'gamma':[0,1,0.5],
                  'min_child_weight':[0,0.5,1.5],
                  'early_stopping_rounds': 50}
xgb_model = xgb.XGBClassifier() 
model_xgb = GridSearchCV(xgb_model, param_grid = param_to_tuned, scoring='roc_auc',
						cv=5,refit=True,return_train_score=True)
# model_xgb = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)
# Svens Parameters
model_xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
							 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
							 max_depth=5, min_child_weight=1, missing=None, n_estimators=120,
							 n_jobs=3, nthread=None, objective='binary:logistic',
							 random_state=500, reg_alpha=1, reg_lambda=0.8, scale_pos_weight=1,
							 seed=None, silent=False, subsample=1, verbose=10)
model_xgb.fit(train_encoding, train_labels)
predictions_xgb = model_xgb.predict(test_encoding_final)
print (metrics.classification_report(test_labels, predictions_xgb))
print (metrics.confusion_matrix(test_labels, predictions_xgb))
print ('PD XGBoost AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model_xgb.predict_proba(test_encoding_final)[:,1] ))
print('max_depth:',model_xgb.best_params_['max_depth'])
print('n_estimators:',model_xgb.best_params_['n_estimators'])
print('learning_rate:',model_xgb.best_params_['learning_rate'])
predicted_XGB_CTR = model_xgb.predict_proba(test_encoding_final)[:,1]

################# Stacking Ensemble (this is stacking of labels from the base level classifiers) ###############
from vecstack import stacking   
from sklearn.metrics import accuracy_score, roc_auc_score


models1 = [LogisticRegression(C=0.7, penalty="l2", class_weight='balanced' ,max_iter=500),
LogisticRegression(C=0.01, penalty="l2", class_weight='balanced' ,max_iter=500),
xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)]

S_train1, S_test1 = stacking(models1, train_encoding, train_labels, test_encoding_final, 
	regression = False, metric = roc_auc_score, n_folds = 3, 
	stratified = True, shuffle = True, random_state = 0, verbose = 2)


models2 = [model]

S_train3, S_test3 = stacking(models2, train_encoding, Y_train, test_encoding_final, 
	regression = False, metric = roc_auc_score, n_folds = 3, 
	stratified = True, shuffle = True, random_state = 0, verbose = 2)



models2 = [als.FMClassification(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)]

S_train2, S_test2 = stacking(models2, sparse.csr_matrix(train_encoding), np.array([1 if i ==1 else -1 for i in train_labels ]), sparse.csr_matrix(test_encoding_final), 
	regression = False, metric = roc_auc_score, n_folds = 5, 
	stratified = True, shuffle = True, random_state = 0, verbose = 2)


#concatenate S_train 1 and 2 (because FM expects sparse scipy matrix as input so can't merge the code together)
#Shape of S_train is num_samples x (number of base classifers). Number of base classifiers is the number of features extracted
S_train = np.hstack((S_train1, np.array([1 if i ==1 else 0 for i in S_train2 ]).reshape(S_train2.shape[0],-1) ))  
S_test = np.hstack((S_test1, np.array([1 if i ==1 else 0 for i in S_test2 ]).reshape(S_test1.shape[0],-1) ))



# 2nd Level: XGBoost
model = xgb.XGBClassifier(seed = 0, n_jobs = -1, learning_rate = 0.1, n_estimators = 300, max_depth = 3)
model.fit(S_train, train_labels)
stacked_predictions= model.predict(S_test)
print (metrics.classification_report(test_labels, stacked_predictions))
print (metrics.confusion_matrix(test_labels, stacked_predictions))
print ('Stacked Ensemble XGBoost AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model.predict_proba(S_test)[:,1] ))

# 2nd Level: Factorization Machine
model =  als.FMClassification(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
model.fit(sparse.csr_matrix(S_train), np.array([1 if i ==1 else -1 for i in train_labels ]))
stacked_predictions= model.predict(sparse.csr_matrix(S_test))
stacked_predictions = np.array([1 if i ==1 else 0 for i in stacked_predictions ]) 
print (metrics.classification_report(test_labels, stacked_predictions))
print (metrics.confusion_matrix(test_labels, stacked_predictions))
print ('Stacked Ensemble FM AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model.predict_proba(sparse.csr_matrix(S_test)) ))

# 2nd Level: Logistic Regression
model = LogisticRegression(C=0.01, penalty="l2", class_weight='balanced' ,max_iter=500)
model.fit(S_train, train_labels)
stacked_predictions= model.predict(S_test)
print (metrics.classification_report(test_labels, stacked_predictions))
print (metrics.confusion_matrix(test_labels, stacked_predictions))
print ('Stacked Ensemble XGBoost AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model.predict_proba(S_test)[:,1] ))



############## Stacking Ensemble with probabilities fed into the meta-learner ##############

# Note: Probabilities work better than the discrete labels in the above stacking ensemble model

# Train Set probabilities
predicted_LR_train_probs =  model_LR.predict_proba(train_encoding)[:,1]
predicted_XGB_train_probs = model_xgb.predict_proba(train_encoding)[:,1]
predicted_MLP_train_probs = model_MLP.predict(train_encoding)[:,1]
# predicted_FM_train_probs = model_FM.predict_proba(sparse.csr_matrix(train_encoding)).reshape(-1,1) 

# Test Set probabilities
predicted_LR_test_probs =  model_LR.predict_proba(test_encoding_final)[:,1]
predicted_XGB_test_probs = model_xgb.predict_proba(test_encoding_final)[:,1]
predicted_MLP_test_probs = model_MLP.predict(test_encoding_final)[:,1]
# predicted_FM_test_probs = model_FM.predict_proba(sparse.csr_matrix(test_encoding_final)).reshape(-1,1) 


# Re-calibrate the probabilities
predicted_LR_train_probs = (predicted_LR_train_probs/(predicted_LR_train_probs+((1-predicted_LR_train_probs)/w))).reshape(-1,1) 
predicted_XGB_train_probs = (predicted_XGB_train_probs/(predicted_XGB_train_probs+((1-predicted_XGB_train_probs)/w))).reshape(-1,1)
predicted_MLP_train_probs = (predicted_MLP_train_probs/(predicted_MLP_train_probs+((1-predicted_MLP_train_probs)/w))).reshape(-1,1)


predicted_LR_test_probs = (predicted_LR_test_probs/(predicted_LR_test_probs+((1-predicted_LR_test_probs)/w))).reshape(-1,1) 
predicted_XGB_test_probs = (predicted_XGB_test_probs/(predicted_XGB_test_probs+((1-predicted_XGB_test_probs)/w))).reshape(-1,1) 
predicted_MLP_test_probs = (predicted_MLP_test_probs/(predicted_MLP_test_probs+((1-predicted_MLP_test_probs)/w))).reshape(-1,1) 


# Concatenate them
stacked_train_set = np.hstack((predicted_LR_train_probs, predicted_XGB_train_probs, predicted_MLP_train_probs))  
stacked_test_set = np.hstack((predicted_LR_test_probs, predicted_XGB_test_probs, predicted_MLP_test_probs))  


# Meta Classifier
Stacked_Meta_Clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.15)
Stacked_Meta_Clf.fit(stacked_train_set, train_labels)
predictions_xgb_stacked = Stacked_Meta_Clf.predict(stacked_test_set)
print (metrics.classification_report(test_labels, predictions_xgb_stacked))
print (metrics.confusion_matrix(test_labels, predictions_xgb_stacked))
print ('PD XGBoost AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  Stacked_Meta_Clf.predict_proba(stacked_test_set)[:,1] ))




# Building a stacked ensemble by predicting the stacked train set using K-Fold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
k_fold = KFold(n_splits=3)

train_reduced_features = []

for train_index, test_index in k_fold.split(train_encoding):
	model_LR_kfold = LogisticRegression(C=0.01, penalty="l2", class_weight='balanced' ,max_iter=500)
	model_xgb_kfold = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)
	# Train on the split train
	model_LR_kfold.fit(train_encoding[train_index], train_labels[train_index])
	model_xgb_kfold.fit(train_encoding[train_index], train_labels[train_index])
	predicted_train_kfold = np.hstack( (model_LR_kfold.predict_proba(train_encoding[test_index])[:,1].reshape(-1,1), model_xgb_kfold.predict_proba(train_encoding[test_index])[:,1].reshape(-1,1)) )
	train_reduced_features.append(predicted_train_kfold)


# Vertically stack the train_reduced_features into one
stacked_train_set = np.vstack((train_reduced_features[0], train_reduced_features[1], train_reduced_features[2]))  


# Then train on whole train set and predict stacked test set
predicted_LR_test_probs =  model_LR.predict_proba(test_encoding_final)[:,1].reshape(-1,1)  
predicted_XGB_test_probs = model_xgb.predict_proba(test_encoding_final)[:,1].reshape(-1,1) 
stacked_test_set = np.hstack((predicted_LR_test_probs, predicted_XGB_test_probs))  


Stacked_Meta_Clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)
Stacked_Meta_Clf.fit(stacked_train_set, train_labels)
predictions_xgb_stacked = Stacked_Meta_Clf.predict(stacked_test_set)
print (metrics.classification_report(test_labels, predictions_xgb_stacked))
print (metrics.confusion_matrix(test_labels, predictions_xgb_stacked))
print ('PD XGBoost AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  Stacked_Meta_Clf.predict_proba(stacked_test_set)[:,1] ))

 


##############  KERAS MLP    ##############  
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


# train_encoding.shape: (62522, 198)
num_features = train_encoding.shape[1]
num_train_samples = train_encoding.shape[0]

model_MLP = Sequential()
model_MLP.add(Dense(128, activation='relu', input_dim=num_features, kernel_initializer='he_uniform'))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model_MLP.add(Dropout(0.5))
model_MLP.add(Dense(2, activation='softmax'))
model_MLP.summary()


Y_train = np_utils.to_categorical(train_labels, 2)
Y_test = np_utils.to_categorical(test_labels, 2)


model_MLP.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer='adam',
			  metrics=['accuracy'])

model_MLP.fit(train_encoding, Y_train,
			batch_size=256,
			epochs=10,
			shuffle=True,
			verbose=1)


predicted_MLP_labels = np.argmax(model_MLP.predict(test_encoding_final), axis=1)
predicted_MLP_probs = model_MLP.predict(test_encoding_final)  # Gives 0 and 1 classes probabilities


print (metrics.classification_report(test_labels, predicted_MLP_labels))
print (metrics.confusion_matrix(test_labels, predicted_MLP_labels))
print ('MLP AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model_MLP.predict(test_encoding_final)[:,1] ))





############## Factorization Machines (Read: https://github.com/vecxoz/vecstack  ) ##############  
# rank (int): The rank of the factorization used for the second order interactions.
model_FM = als.FMClassification(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
# Expects sparse scipy matrix as input and the outputs should be 1 or -1, not 1 or 0
model_FM.fit(sparse.csr_matrix(train_encoding), np.array([1 if i ==1 else -1 for i in train_labels ]))
predicted_FM = model_FM.predict(sparse.csr_matrix(test_encoding_final))
predicted_FM = np.array([1 if i ==1 else 0 for i in predicted_FM ])  #Convert labels back to 1 and 0

print (metrics.classification_report(test_labels, predicted_FM))
print (metrics.confusion_matrix(test_labels, predicted_FM))
print ('FM AU ROC (Hold out Set): ', metrics.roc_auc_score(test_labels,  model_FM.predict_proba(sparse.csr_matrix(test_encoding_final)) ))
predicted_FM_CTR = model_FM.predict_proba(sparse.csr_matrix(test_encoding_final))

print ('-----------------------------------')



############## Constant Bidding Strategy #################
print ('------- Constant Bidding Strategy -------')
# For constant bidding (Training the parameter on the CV set)
budget = 6250 # CNY
optimal_const_bid = 0
optimal_num_clicks = 0
num_won_auctions = 0

# For plotting
bids_array = []
num_clicks_array = []

# Min pay price is 0, Max pay price is 0.3, mean is 0.08
# Number of bid requests is 303,925
df_cv['payprice'].describe()

# Do a grid search to find optimal value which achieves max KPI (# clicks) and depletes the budget completely
for bid in np.linspace(0.01, 0.30, 130):
	df_cv['bid'] = bid
	# You win the auction if bid >= slot price and pay price (2nd highest bidder)
	Won_Auctions = df_cv[ (df_cv['bid'] >= df_cv['slotprice']) & (df_cv['bid'] >= df_cv['payprice']) ]
	# Won_Auctions = df_cv.sample(frac=1, random_state=5)[ (df_cv['bid'] >= df_cv['slotprice']) & (df_cv['bid'] >= df_cv['payprice']) ]
	exceeding_indexes = np.flatnonzero(  (Won_Auctions['payprice'].cumsum().values) > budget  )
	# We exceeded the budget in the won auctions
	if len(exceeding_indexes) > 0:
		# Gives the first index as soon as the sum of the payprice exceeds the budget
		ExceedBudget_Index = exceeding_indexes[0]
		# Only consider the won auctions up to ExceedBudget_Index
		num_clicks = Won_Auctions[:ExceedBudget_Index]['click'].sum()
		budget_spent = Won_Auctions[:ExceedBudget_Index]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:ExceedBudget_Index])
	# We didn't exceed the budget
	else:
		num_clicks = Won_Auctions[:]['click'].sum()
		budget_spent = Won_Auctions[:]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:])
	# Count KPI (# clicks)
	if num_clicks > optimal_num_clicks:
		optimal_const_bid = bid
		optimal_num_clicks = num_clicks
	print ('No Budget Limit: ', Won_Auctions['payprice'].sum(), 'Num Clicks: ',  Won_Auctions['click'].sum() , 'Yes Budget Limit: ', budget_spent, "Num Clicks: ", num_clicks , "CTR: ", num_clicks/num_won_auctions, '# won impressions: ', num_won_auctions , "bid: ", bid)
	bids_array.append(bid)
	num_clicks_array.append(num_clicks)


# Plot a graph of bids vs KPI (number of click)
plt.plot(bids_array, num_clicks_array)
plt.xlabel("bid value")
plt.ylabel("KPI (# clicks)")
plt.title("Bid vs KPI (subject to budget constraint)")
plt.show()

# optimal bid is:  0.0778481012658 Optimal num clicks:  68, 146865 won impressions
print ("optimal bid is: ", optimal_const_bid, "Optimal num clicks: ", optimal_num_clicks)
print ('-----------------------------------')
########################################################



############### Random Bidding Strategy ###################
print ('------- Random Bidding Strategy -------')
# For random bidding (Training the upper and lower bound on the CV set)
budget = 6250 # CNY
optimal_lower_bound = 0
optimal_upper_bound = 0
optimal_num_clicks = 0
num_won_auctions = 0

for lower_bound in np.linspace(0.0, 0.30, 20):
	for upper_bound in np.linspace(0.001, 0.31, 20):
		if lower_bound < upper_bound:
			random_bids = np.random.uniform(lower_bound, upper_bound, size=df_cv.shape[0])
			df_cv['bid'] = random_bids
			# You win the auction if bid >= slot price and pay price (2nd highest bidder)
			Won_Auctions = df_cv[ (df_cv['bid'] >= df_cv['slotprice']) & (df_cv['bid'] >= df_cv['payprice']) ]
			exceeding_indexes = np.flatnonzero(  (Won_Auctions['payprice'].cumsum().values) > budget  )
			# We exceeded the budget in the won auctions
			if len(exceeding_indexes) > 0:
				# Gives you the first index when the sum of the payprice exceeds the budget
				ExceedBudget_Index = exceeding_indexes[0]
				# Only consider the won auctions up to ExceedBudget_Index
				num_clicks = Won_Auctions[:ExceedBudget_Index]['click'].sum()
				budget_spent = Won_Auctions[:ExceedBudget_Index]['payprice'].sum()
				num_won_auctions = len(Won_Auctions[:ExceedBudget_Index])
			# We didn't exceed the budget
			else:
				num_clicks = Won_Auctions[:]['click'].sum()
				budget_spent = Won_Auctions[:]['payprice'].sum()
				num_won_auctions = len(Won_Auctions[:])
			# Count KPI (# clicks)
			if num_clicks > optimal_num_clicks:
				optimal_lower_bound = lower_bound
				optimal_upper_bound = upper_bound
				optimal_num_clicks = num_clicks
			print ('No Budget Limit: ', Won_Auctions['payprice'].sum(), 'Num Clicks: ',  Won_Auctions['click'].sum(), 'Yes Budget Limit: ', budget_spent, "Num Clicks: ", num_clicks , "CTR: ", num_clicks/num_won_auctions, '# won impressions: ', num_won_auctions, "range bid: ", (lower_bound, upper_bound))


# Plot graph with the upper bound


# optimal bid range is:  (0.012244897959183673, 0.12081632653061225) Optimal num clicks:  80
print ("optimal bid range is: ", (optimal_lower_bound, optimal_upper_bound), "Optimal num clicks: ", optimal_num_clicks)
print ('-----------------------------------')
#########################################################


############### Linear Bidding Strategy ###################
print ('------- Linear Bidding Strategy -------')

budget = 6250 
optimal_basebid = 0
optimal_num_clicks = 0
num_won_auctions = 0

# For plotting
base_bids_array = []
num_clicks_array = []

avg_ctr = 7.375623e-04  # Avg CTR from training set (for all the advertisers)
# predicted_CTR = model_LR.predict_proba(test_encoding_final)[:,1]
predicted_CTR = model_xgb.predict_proba(test_encoding_final)[:,1]

# Perform Calibration
predicted_CTR = predicted_CTR/(predicted_CTR+((1-predicted_CTR)/w))

# Do a grid search to find optimal base bid which achieves max KPI (# clicks) and depletes the budget completely
for bid in np.linspace(0.00008, 0.2, 190):
	df_cv['bid'] = (bid/avg_ctr) * predicted_CTR
	# df_cv['bid'] = (bid/avg_ctr) * df_cv['pCTR']
	# df_cv['bid'] = (bid/df_cv['avgCTR']) * df_cv['pCTR']
	# You win the auction if bid >= slot price and pay price (2nd highest bidder)
	Won_Auctions = df_cv[ (df_cv['bid'] >= df_cv['slotprice']) & (df_cv['bid'] >= df_cv['payprice']) ]
	exceeding_indexes = np.flatnonzero(  (Won_Auctions['payprice'].cumsum().values) > budget  )
	# We exceeded the budget in the won auctions
	if len(exceeding_indexes) > 0:
		# Gives the first index as soon as the sum of the payprice exceeds the budget
		ExceedBudget_Index = exceeding_indexes[0]
		# Only consider the won auctions up to ExceedBudget_Index
		num_clicks = Won_Auctions[:ExceedBudget_Index]['click'].sum()
		budget_spent = Won_Auctions[:ExceedBudget_Index]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:ExceedBudget_Index])
	# We didn't exceed the budget
	else:
		num_clicks = Won_Auctions[:]['click'].sum()
		budget_spent = Won_Auctions[:]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:])
	# Count KPI (# clicks)
	if num_clicks > optimal_num_clicks:
		optimal_basebid = bid
		optimal_num_clicks = num_clicks
	print ('No Budget Limit: ', Won_Auctions['payprice'].sum(), 'Num Clicks: ',  Won_Auctions['click'].sum() , 'Yes Budget Limit: ', budget_spent, "Num Clicks: ", num_clicks , "CTR: ", num_clicks/num_won_auctions, '# won impressions: ', num_won_auctions , "bid: ", bid)
	
	base_bids_array.append(bid)
	num_clicks_array.append(num_clicks)


# Plot a graph of bids vs KPI (number of click)
plt.plot(base_bids_array, num_clicks_array)
plt.xlabel("bid value")
plt.ylabel("KPI (# clicks)")
plt.title("Bid vs KPI (subject to budget constraint)")
plt.show()



optimal_bids = (optimal_basebid/avg_ctr) * predicted_CTR
optimal_bids = (0.117/avg_ctr) * predicted_CTR
Grapher = list(zip(list(predicted_CTR), list(optimal_bids)))
Grapher.sort(key=lambda x: x[1])
x_vals = [i[0] for i in Grapher ]
y_vals = [i[1] for i in Grapher ]
plt.plot(x_vals, y_vals)
plt.show()

#optimal bid is:  0.119558816568 Optimal num clicks:  163
print ("optimal bid is: ", optimal_basebid, "Optimal num clicks: ", optimal_num_clicks)

print ('-----------------------------------')
#########################################################



############### Square Bidding Strategy ###################
print ('------- Square Bidding Strategy -------')

budget = 6250
optimal_basebid = 0
optimal_num_clicks = 0
num_won_auctions = 0

# For plotting
base_bids_array = []
num_clicks_array = []

avg_ctr = 7.375623e-04  # Avg CTR from training set (for all the advertisers)
predicted_CTR = model_xgb.predict_proba(test_encoding_final)[:,1]
# Perform Calibration
predicted_CTR = predicted_CTR/(predicted_CTR+((1-predicted_CTR)/w)) 

# Do a grid search to find optimal base bid which achieves max KPI (# clicks) and depletes the budget completely
for bid in np.linspace(0.05, 0.6, 90):
	df_cv['bid'] = bid * ((predicted_CTR/avg_ctr)**2)
	# You win the auction if bid >= slot price and pay price (2nd highest bidder)
	Won_Auctions = df_cv[ (df_cv['bid'] >= df_cv['slotprice']) & (df_cv['bid'] >= df_cv['payprice']) ]
	exceeding_indexes = np.flatnonzero(  (Won_Auctions['payprice'].cumsum().values) > budget  )
	# We exceeded the budget in the won auctions
	if len(exceeding_indexes) > 0:
		# Gives the first index as soon as the sum of the payprice exceeds the budget
		ExceedBudget_Index = exceeding_indexes[0]
		# Only consider the won auctions up to ExceedBudget_Index
		num_clicks = Won_Auctions[:ExceedBudget_Index]['click'].sum()
		budget_spent = Won_Auctions[:ExceedBudget_Index]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:ExceedBudget_Index])
	# We didn't exceed the budget
	else:
		num_clicks = Won_Auctions[:]['click'].sum()
		budget_spent = Won_Auctions[:]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:])
	# Count KPI (# clicks)
	if num_clicks > optimal_num_clicks:
		optimal_basebid = bid
		optimal_num_clicks = num_clicks
	print ('No Budget Limit: ', Won_Auctions['payprice'].sum(), 'Num Clicks: ',  Won_Auctions['click'].sum() , 'Yes Budget Limit: ', budget_spent, "Num Clicks: ", num_clicks , "CTR: ", num_clicks/num_won_auctions, '# won impressions: ', num_won_auctions , "bid: ", bid)
	
	base_bids_array.append(bid)
	num_clicks_array.append(num_clicks)


# Plot a graph of bids vs KPI (number of click)
plt.plot(base_bids_array, num_clicks_array)
plt.xlabel("bid value")
plt.ylabel("KPI (# clicks)")
plt.title("Bid vs KPI (subject to budget constraint)")
plt.show()


optimal_bids = optimal_basebid * ((predicted_CTR/avg_ctr)**2)
optimal_bids = 0.17359 * ((predicted_CTR/avg_ctr)**2)
Grapher = list(zip(list(predicted_CTR), list(optimal_bids)))
Grapher.sort(key=lambda x: x[1])
x_vals = [i[0] for i in Grapher ]
y_vals = [i[1] for i in Grapher ]
plt.plot(x_vals, y_vals)
plt.show()

#optimal bid is:  0.119558816568 Optimal num clicks:  163
print ("optimal bid is: ", optimal_basebid, "Optimal num clicks: ", optimal_num_clicks)

print ('-----------------------------------')
#########################################################




############### ORTB Bidding Strategy ###################
print ('------- ORTB & My best strategy Bidding Strategy -------')

# My best strategy is to mix ORTB 1 and Linear bidding and the threshold is the intersection of the 2 strategies independently

def ortb1(bid, lambda_ortb, pCTR):
	return np.sqrt( ((bid/lambda_ortb)*pCTR) + np.square(bid)) - bid

def ortb2(bid, lambda_ortb, pCTR):
	return (bid * ( np.cbrt( (pCTR + np.sqrt( ( np.square(bid) * np.square(lambda_ortb) ) + np.square(pCTR)  )) / (bid*lambda_ortb)   )  - np.cbrt( (bid*lambda_ortb) / ( pCTR + np.sqrt( ( np.square(bid) * np.square(lambda_ortb) ) + np.square(pCTR)  ) )   )  ) )

def linear(bid, lambda_ortb, pCTR):
	return (bid/avg_ctr) * pCTR

def squared_bidding(parameter_1, pCTR):
	return parameter_1 * ((pCTR/avg_ctr)**2)


#I tried parameter_1*x^2 +parameter_2 and parameter_1*x^2 + parameter_1*x, both gave 170 on validation set
# parameter_1 * (predicted_CTR/avg_CTR)**2

#(or 0.215 and 0.2175)    square   OR 0.265

budget = 6250 # CNY
optimal_C = 0
lambda_ortb = 5.2e-7  # Value found by gradient descent in the papers
optimal_num_clicks = 0
num_won_auctions = 0

# For plotting
base_bids_array = []
num_clicks_array = []

# predicted_CTR = model_LR.predict_proba(test_encoding_final)[:,1]
predicted_CTR = model_xgb.predict_proba(test_encoding_final)[:,1]
# predicted_CTR = Stacked_Meta_Clf.predict_proba(test_encoding_final)[:,1]

# Perform Calibration
predicted_CTR = predicted_CTR/(predicted_CTR+((1-predicted_CTR)/w))

# Do a grid search to find optimal base bid which achieves max KPI (# clicks) and depletes the budget completely
for bid in np.linspace(5, 10, 100):
# for bid in np.linspace(, 40, 200):
	# df_cv['bid'] = (np.sqrt( ((bid/lambda_ortb)*predicted_CTR) + np.square(bid)) - bid)/1000.0    # ORTB1
	# df_cv['bid'] = (bid * ( np.cbrt( (predicted_CTR + np.sqrt( ( np.square(bid) * np.square(lambda_ortb) ) + np.square(predicted_CTR)  )) / (bid*lambda_ortb)   )  - np.cbrt( (bid*lambda_ortb) / ( predicted_CTR + np.sqrt( ( np.square(bid) * np.square(lambda_ortb) ) + np.square(predicted_CTR)  ) )   )  ) )/1000.0 # ORTB 2 (more competitive impressions)
	# df_cv['bid'] = np.array( [linear(0.119558816568, lambda_ortb, i) if i>0.00055 else ortb1(bid, lambda_ortb, i)/1000.0 for i in predicted_CTR] )   # My best mixed bidding strategy
	df_cv['bid'] = np.array( [squared_bidding(0.17359, i) if i>0.0005 else ortb1(bid, lambda_ortb, i)/1000.0 for i in predicted_CTR] )   # My best mixed bidding strategy
	# You win the auction if bid >= slot price and pay price (2nd highest bidder)
	Won_Auctions = df_cv[ (df_cv['bid'] >= df_cv['slotprice']) & (df_cv['bid'] >= df_cv['payprice']) ]
	exceeding_indexes = np.flatnonzero(  (Won_Auctions['payprice'].cumsum().values) > budget  )
	# We exceeded the budget in the won auctions
	if len(exceeding_indexes) > 0:
		# Gives the first index as soon as the sum of the payprice exceeds the budget
		ExceedBudget_Index = exceeding_indexes[0]
		# Only consider the won auctions up to ExceedBudget_Index
		num_clicks = Won_Auctions[:ExceedBudget_Index]['click'].sum()
		budget_spent = Won_Auctions[:ExceedBudget_Index]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:ExceedBudget_Index])
	# We didn't exceed the budget
	else:
		num_clicks = Won_Auctions[:]['click'].sum()
		budget_spent = Won_Auctions[:]['payprice'].sum()
		num_won_auctions = len(Won_Auctions[:])
	# Count KPI (# clicks)
	if num_clicks > optimal_num_clicks:
		optimal_C = bid
		# lambda_ortb = l
		optimal_num_clicks = num_clicks
	print ('No Budget Limit: ', Won_Auctions['payprice'].sum(), 'Num Clicks: ',  Won_Auctions['click'].sum() , 'Yes Budget Limit: ', budget_spent, "Num Clicks: ", num_clicks , "CTR: ", num_clicks/num_won_auctions, '# won impressions: ', num_won_auctions , "bid: ", bid)
	
	base_bids_array.append(bid)
	num_clicks_array.append(num_clicks)


optimal_bids = (np.sqrt( ((optimal_C/lambda_ortb)*predicted_CTR) + np.square(optimal_C)) - optimal_C)/1000.0
optimal_bids = ((optimal_C * ( np.cbrt( (predicted_CTR + np.sqrt( ( np.square(optimal_C) * np.square(lambda_ortb) ) + np.square(predicted_CTR)  )) / (optimal_C*lambda_ortb)   )  - np.cbrt( (optimal_C*lambda_ortb) / ( predicted_CTR + np.sqrt( ( np.square(optimal_C) * np.square(lambda_ortb) ) + np.square(predicted_CTR)  ) )   )  ) )/1000.0)
optimal_bids = np.array( [linear(0.119558816568, lambda_ortb, i) if i>0.0005 else ortb1(bid, lambda_ortb, i)/1000.0 for i in predicted_CTR] )

optimal_bids = np.array( [linear(0.117, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR] )
optimal_bids = np.array( [ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR] )

optimal_bids = np.array( [linear(0.117, lambda_ortb, i) if i>0.000635 else (ortb1(3.0, lambda_ortb, i-0.000320)/1000.0)+0.0505 if i>0.000330  else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR] )
optimal_bids = np.array( [linear(0.150, lambda_ortb, i) if i>0.001 else linear(0.117, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR  ] )


optimal_bids = np.array( [squared_bidding(0.17359, i) if i>0.7 else linear(0.118, lambda_ortb, i) if i>0.0005 else ortb1(5.1, lambda_ortb, i)/1000.0 for i in predicted_CTR] )

optimal_bids = []
for i in predicted_CTR:
	if i < 0.0005:
		optimal_bids.append( ortb1(5.1, lambda_ortb, i)/1000.0 )  
	elif i < 0.002:
		optimal_bids.append( linear(0.117, lambda_ortb, i) )
	else:
		# optimal_bids.append( linear(0.130, lambda_ortb, i) )
		optimal_bids.append( linear(0.117, lambda_ortb, i) )

optimal_bids = np.array(optimal_bids)


Grapher = list(zip(list(predicted_CTR), list(optimal_bids)))
Grapher.sort(key=lambda x: x[1])
x_vals = [i[0] for i in Grapher ]
y_vals = [i[1] for i in Grapher ]
plt.plot(x_vals, y_vals)
plt.xlabel("Predicted CTR (Calibrated)")
plt.ylabel("Bid Price")
plt.title("Best Strategy Bid vs Calibrated pCTR")
plt.show()

# Plot a graph of bids vs KPI (number of click)
plt.plot(base_bids_array, num_clicks_array)
plt.xlabel("bid value")
plt.ylabel("KPI (# clicks)")
plt.title("Bid vs KPI (subject to budget constraint)")
plt.show()

print ("optimal C is: ", optimal_C, "Optimal num clicks: ", optimal_num_clicks)

print ('-----------------------------------')
#########################################################



print ('--------Predict Test Set Bids-----------------')


# parameter * (prediction/avg_ctr)^2


Reversed_Test_Set = submission_test_encoding_final[::-1]


predicted_CTR = model_xgb.predict_proba(submission_test_encoding_final)[:,1]
predicted_CTR = predicted_CTR/(predicted_CTR+((1-predicted_CTR)/w))


# Optimal hyper-parameters for my mixed (ORTB 1 + linear) bidding strategy which depletes the budget is: 
# For the linear portion, optimal base bid is 0.119558816568
# For the ORTB 1 portion, optimal C is 5.52
optimal_bids = np.array( [linear(0.125, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR] )
optimal_bids = np.array( [linear(0.150, lambda_ortb, i) if i>0.001 else linear(0.117, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR  ] )
optimal_bids = np.array( [linear(0.117, lambda_ortb, i) if i>0.000635 else (ortb1(5.0, lambda_ortb, i-0.000320)/1000.0)+0.0505 if i>0.000330  else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR] )  # Two ORTB1s

test_bids_file = pd.DataFrame(index=range(0,len(predicted_CTR)), columns=['bidid', 'bidprice'])
test_bids_file['bidid'] = np.array( df_test['bidid'] )
test_bids_file['bidprice'] = (optimal_bids * 1000.0)  # Submit the bids in CPM


# Reverse the optimal bids to match the original format
test_bids_file['bidprice'] = ((optimal_bids[::-1]) * 1000.0)  # Submit the bids in CPM


# Save the CSV files
try:
    os.remove('./testing_bidding_price.csv')
except OSError:
    pass


test_bids_file.to_csv(path_or_buf='./testing_bidding_price.csv', index=False)


# For the linear portion, optimal base bid is 0.119558816568
# For the ORTB 1 portion, optimal C is 5.52
# Threshold is 0.0005
{
"ranking": 1, 
"group": "16", 
"result": 
		{"impressions": 141202, 
		"cost": 6209.72200000264, 
		"clicks": 175, 
		"ctr": 0.0012393592158751292, 
		"cpc": 35.4841257143008}, 
"daily submission limit": 3, 
"today tried times": 1, 
"best result": 
		{"impressions": 141202, 
		"cost": 6209.72200000264, 
		"clicks": 175, 
		"ctr": 0.0012393592158751292, 
		"cpc": 35.4841257143008
		}
}



# For the linear portion, optimal base bid is 0.119558816568
# For the ORTB 1 portion, optimal C is 5.55
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 141388,
        "cost": 6218.613000002652,
        "clicks": 175,
        "ctr": 0.0012377288030101564,
        "cpc": 35.53493142858658
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# Reversed
# For the linear portion, optimal base bid is 0.119558816568
# For the ORTB 1 portion, optimal C is 5.61
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 141753,
        "cost": 6236.001000002676,
        "clicks": 175,
        "ctr": 0.0012345417733663484,
        "cpc": 35.63429142858672
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}




# ORTB 1, C=7.94
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 151217,
        "cost": 6249.998000002938,
        "clicks": 163,
        "ctr": 0.0010779211332059226,
        "cpc": 38.343546012287966
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# Try Only Linear  (At the optimal base bid of validation set), 
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 133803,
        "cost": 6127.582000002441,
        "clicks": 173,
        "ctr": 0.001292945599127075,
        "cpc": 35.41954913296209
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# Try Normal Order, increase C = 5.67  to deplete budget on test set
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 142052,
        "cost": 6249.999000002685,
        "clicks": 173,
        "ctr": 0.0012178638808323713,
        "cpc": 36.1271618497265
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# Try Normal Order, increase C = 5.63  to deplete budget on test set
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 141885,
        "cost": 6242.31100000268,
        "clicks": 175,
        "ctr": 0.0012333932410050392,
        "cpc": 35.67034857144389
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# Try Normal Order, increase C = 5.64  to deplete budget on test set
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 141942,
        "cost": 6245.052000002679,
        "clicks": 175,
        "ctr": 0.0012328979442307421,
        "cpc": 35.68601142858674
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}




# Reversed
# My best: C = 5.67
{
    "ranking": 1,
    "group": "16",
    "result": {
        "impressions": 142052,
        "cost": 6249.999000002685,
        "clicks": 173,
        "ctr": 0.0012178638808323713,
        "cpc": 36.1271618497265
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# 175_6197_Mixed_5.52,0.119,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 141097,
        "cost": 6197.141000002631,
        "clicks": 175,
        "ctr": 0.0012402815084658072,
        "cpc": 35.41223428572932
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6173_Mixed_5.52,0.118,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 140895,
        "cost": 6173.241000002642,
        "clicks": 175,
        "ctr": 0.0012420596898399517,
        "cpc": 35.275662857157954
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# 175_6196_Mixed_5.52,0.1175,0.00049
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 141267,
        "cost": 6196.410000002653,
        "clicks": 175,
        "ctr": 0.0012387889599127892,
        "cpc": 35.4080571428723
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6145_Mixed_5.5,0.117,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 140568,
        "cost": 6145.120000002619,
        "clicks": 175,
        "ctr": 0.001244949063798304,
        "cpc": 35.114971428586394
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# 175_6131_Mixed_5.5,0.117,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 140288,
        "cost": 6131.8530000026,
        "clicks": 175,
        "ctr": 0.0012474338503649636,
        "cpc": 35.03916000001486
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6099_Mixed_5.35,0.117,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 139611,
        "cost": 6099.765000002569,
        "clicks": 175,
        "ctr": 0.0012534828917492174,
        "cpc": 34.85580000001468
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6074_Mixed_5.27,0.117,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 139072,
        "cost": 6074.668000002536,
        "clicks": 175,
        "ctr": 0.001258341003221353,
        "cpc": 34.712388571443064
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6039_Mixed_5.15,0.117,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 138325,
        "cost": 6039.703000002507,
        "clicks": 175,
        "ctr": 0.0012651364540032532,
        "cpc": 34.5125885714429
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# 175_6045_Mixed_5.1,0.118,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 138173,
        "cost": 6045.854000002504,
        "clicks": 175,
        "ctr": 0.0012665281929175743,
        "cpc": 34.547737142871455
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 172_5856_Mixed_4.5,0.118,0.0005
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 134015,
        "cost": 5856.888000002346,
        "clicks": 172,
        "ctr": 0.001283438421072268,
        "cpc": 34.05167441861829
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6082_Mixed_5.1,0.118,0.00049
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 138674,
        "cost": 6082.180000002535,
        "clicks": 175,
        "ctr": 0.0012619524928970103,
        "cpc": 34.75531428572877
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6175_Mixed_5.1,0.118,0.00046
{
    "ranking": 8,
    "group": "16",
    "result": {
        "impressions": 139997,
        "cost": 6175.915000002601,
        "clicks": 175,
        "ctr": 0.0012500267862882775,
        "cpc": 35.290942857157724
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6214_Mixed_5.05,0.118,0.00044
{
    "ranking": 9,
    "group": "16",
    "result": {
        "impressions": 140450,
        "cost": 6214.133000002574,
        "clicks": 175,
        "ctr": 0.001245995016019936,
        "cpc": 35.50933142858614
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6064_Mixed_5.0,0.120,0.0005
{
    "ranking": 9,
    "group": "16",
    "result": {
        "impressions": 137936,
        "cost": 6064.811000002492,
        "clicks": 175,
        "ctr": 0.0012687043266442408,
        "cpc": 34.656062857157096
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}

# 172_6249_Mixed_5.0,0.130,0.0005
{
    "ranking": 9,
    "group": "16",
    "result": {
        "impressions": 138720,
        "cost": 6249.998000002583,
        "clicks": 172,
        "ctr": 0.0012399077277970012,
        "cpc": 36.33719767443362
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6116_Mixed_5.0,0.122,0.0005
{
    "ranking": 10,
    "group": "16",
    "result": {
        "impressions": 138363,
        "cost": 6116.8380000025245,
        "clicks": 175,
        "ctr": 0.0012647889970584622,
        "cpc": 34.95336000001443
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}


# 175_6187_Mixed_5.0,0.125,0.0005
{
    "ranking": 10,
    "group": "16",
    "result": {
        "impressions": 138927,
        "cost": 6187.0850000025675,
        "clicks": 175,
        "ctr": 0.0012596543508461278,
        "cpc": 35.3547714285861
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# Mixed levels
{
    "ranking": 10,
    "group": "16",
    "result": {
        "impressions": 140000,
        "cost": 6186.447000002595,
        "clicks": 175,
        "ctr": 0.00125,
        "cpc": 35.351125714300544
    },
    "daily submission limit": 3,
    "today tried times": 3,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# optimal_bids = np.array( [linear(0.130, lambda_ortb, i) if i>0.001 else linear(0.117, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR  ] )
# 1.csv
{
    "ranking": 11,
    "group": "16",
    "result": {
        "impressions": 137660,
        "cost": 6061.646000002497,
        "clicks": 175,
        "ctr": 0.0012712480023245678,
        "cpc": 34.637977142871414
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}




# optimal_bids = np.array( [linear(0.150, lambda_ortb, i) if i>0.001 else linear(0.117, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR  ] )
# 2.csv
{
    "ranking": 11,
    "group": "16",
    "result": {
        "impressions": 138034,
        "cost": 6143.31400000255,
        "clicks": 175,
        "ctr": 0.0012678035846240782,
        "cpc": 35.104651428585996
    },
    "daily submission limit": 3,
    "today tried times": 2,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}



# optimal_bids = np.array( [linear(0.150, lambda_ortb, i) if i>0.001 else linear(0.117, lambda_ortb, i) if i>0.0005 else ortb1(5.0, lambda_ortb, i)/1000.0 for i in predicted_CTR  ] )
# 3.csv
{
    "ranking": 13,
    "group": "16",
    "result": {
        "impressions": 137134,
        "cost": 6249.999000002473,
        "clicks": 156,
        "ctr": 0.0011375734682864936,
        "cpc": 40.064096153862
    },
    "daily submission limit": 3,
    "today tried times": 1,
    "best result": {
        "impressions": 141202,
        "cost": 6209.72200000264,
        "clicks": 175,
        "ctr": 0.0012393592158751292,
        "cpc": 35.4841257143008
    }
}











