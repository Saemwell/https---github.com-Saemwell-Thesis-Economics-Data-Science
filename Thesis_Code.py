# Additional functions for analysing and manipulating data
from audioop import rms
import io
from msilib.schema import Directory
from re import A
import pandas as pd

# Fundamental package for scientific computing with Python
import numpy as np
from datetime import datetime

# Important package for visualization - we use this to plot the market data
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.dates as mdates
import seaborn as sns
import calplot
import seaborn as sb

# To get the right dates for our prediction of tomorrow's stockdata we need the following package
from datetime import datetime, timedelta

# Spearman correlation package
from scipy.stats import spearmanr

#MinMax scale package
from sklearn import preprocessing

#RNN LSTM package
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#keras tuning packages for RNN LSTM
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

#SES package
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from statsmodels.tsa.api import Holt
import statsmodels.api as sm
from sklearn import metrics

#Kfold Cross Validation
from sklearn.model_selection import cross_val_score

#RFECV 
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from pandas.core.common import random_state

#Keras Tuner for RNN LSTM model configuration
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# importing module for Multi Linear Regression
from sklearn.linear_model import LinearRegression
import pyfinance


# Hiding warnings
import warnings
warnings.filterwarnings('ignore')
#%%

#Import data from csv file
On_Chain_metrics = pd.read_csv(r"D:\Users\samue\Documents\Economics - Data Science Track\Thesis\Code\Full_dataset.csv")

On_Chain_metrics.index

#Date as index in a datetime object format

On_Chain_metrics['Date'] = pd.to_datetime(On_Chain_metrics['Date'])
On_Chain_metrics = On_Chain_metrics.set_index('Date')

#%%

#Data inspection of 'Bitcoin's On-Chain dataset retrieved via the Santiment's API 
  #We only look at first five here for minimal output

print('We have',On_Chain_metrics.shape[0],'rows and',On_Chain_metrics.shape[1],'columns in this DataFrame')
On_Chain_metrics.head(5)

#%%
#Checking data types

On_Chain_metrics.dtypes

#Statistical overview of On-Chain metric data
On_Chain_metrics.describe()

correlation_matrix = On_Chain_metrics.drop(['Open ', 'High', 'Low ', 'Dev_Activity', 'Exchange_In_Out_Flow','Age_Destroyed'], axis=1)

#Create new Correlation Matrix: rho
#call the correlation function with method Spearman & round the values to two decimals
rho = correlation_matrix.corr(method="spearman").round(2)

#get the p values
pval = correlation_matrix.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)

#set the p values, *** for less than 0.001, ** for less than 0.01, * for less than 0.05
p = pval.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

#Corr_Data_Pvalue below gives the dataframe with correlation coefficients and p values
Corr_Data_Pvalue = rho.astype(str) + p

Corr_Data_Pvalue

#%%
#Creating a correlation matrix between all the features in the dataset for a heatmap

#Create the matrix based on the spearman correlation method
matrix = np.triu(correlation_matrix.corr(method="spearman"))

#convert to array for the heatmap
Heatmap_Dataset = Corr_Data_Pvalue.to_numpy()

#plot the heatmap
plt.figure(figsize=(30,16))
sns.heatmap(rho, annot = Heatmap_Dataset, fmt='', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask = matrix)

#%%
#Correlation dataframe previously derived from Spearman method duplicated
#because the loop function doesn't work with the Close_Price with Pvalues (asterix) in thesame column in dataframe: Corr_Data_Pvalue,
#rho will be used to create Spearman's correlation table Appended as a column in the dataframe as "Spearman_Verdict"
#https://www.statstutor.ac.uk/resources/uploaded/spearmans.pdf

Spearman_Verdict = []

for value in rho["Close_Price"]:

    if value > 0:
      if value < .20:
        Spearman_Verdict.append("Very Weak") 
      elif value < .40:
        Spearman_Verdict.append("Weak")
      elif value < .60: 
        Spearman_Verdict.append("moderate")
      elif value < .80:
        Spearman_Verdict.append("strong")
      else:
        Spearman_Verdict.append("Very Strong")
    else:
      if value > -.20:
        Spearman_Verdict.append("Very Weak") 
      elif value > -.40:
        Spearman_Verdict.append("Weak")
      elif value > -.60: 
        Spearman_Verdict.append("moderate")
      elif value > -.80:
        Spearman_Verdict.append("strong")
      else:
        Spearman_Verdict.append("Very Strong")

rho["Spearman_Verdict"] = Spearman_Verdict

#Select Columns: Close_Price & Spearman_Verdict
rho =rho[["Close_Price","Spearman_Verdict"]]

#Drop the first 4 rows of the dataframe: Open, High, Low & Close_Price, 
rho = rho.drop(rho.iloc[[0]].index)

#Name index: 'Metrics'
rho.index.name = 'Metrics'

print('We have',rho.shape[0],'rows and',rho.shape[1],'columns in this DataFrame')
rho

#%%

# Add the Close Price to the "rho" selection of indicators to make an array to use in subselecting 
# the correct metrics (indicators) to use for the RNN LSTM

#First use the index: 'Metrics' to make a new column in the dataframe
rho.reset_index('Metrics', inplace=True)

#Now convert the new column into an array 
Metrics = rho['Metrics'].to_numpy()

#Add Close_Price
Metrics = np.append(Metrics, "Close_Price")

# Set as new dataframe
BTC_dataset = On_Chain_metrics[Metrics]
BTC_dataset

#%%

#Creating a dataframe with the Close Price as the target variable for Double Exponential Smoothing

On_Chain_DES = BTC_dataset['Close_Price']
df_train = pd.DataFrame(On_Chain_DES.iloc[:2000], columns=['Close_Price'])
df_test = pd.DataFrame(On_Chain_DES.iloc[2000:], columns=['Close_Price'])

#%%

#Scaling the test and train dataset between 0 and 1 seperately
sct = MinMaxScaler(feature_range= (0, 1))

df_train_sc = sct.fit_transform(df_train)
df_test_sc = sct.transform(df_test)
#%%

#Evaluation metrics

def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Calculate the mean absolute percentage error as a metric for evaluation
    
    Args:
        y_true (float64): Y values for the dependent variable (test part), numpy array of floats 
        y_pred (float64): Predicted values for the dependen variable (test parrt), numpy array of floats
    
    Returns:
        Mean absolute percentage error 
    '''    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def timeseries_evaluation_metrics_func(y_true, y_pred):
    '''
    Calculate the following evaluation metrics:
        - MSE
        - MAE
        - RMSE
        - MAPE
        - R²
    
    Args:
        y_true (float64): Y values for the dependent variable (test part), numpy array of floats 
        y_pred (float64): Predicted values for the dependen variable (test parrt), numpy array of floats
    
    Returns:
        MSE, MAE, RMSE, MAPE and R² 
    '''    
    print('Evaluation metric results: ')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')

#%%


#Double Exponential Smoothing with the optimized version of the algorithm
#Where data is scaled between 0 and 1 for comparison with the RNN LSTM and Rolling Linear Regression

DES = Holt(df_train_sc)
fit_Holt_auto = DES.fit(optimized= True, use_brute = True)

fcst_auto_pred_Holt = fit_Holt_auto.forecast(len(df_test_sc))
timeseries_evaluation_metrics_func(df_test_sc, fcst_auto_pred_Holt)

#Double Exponential Smoothing with the optimized version of the algorithm
#Where data is not scaled, but will be used for plotting the result in the code below
DES = Holt(df_train)
fit_Holt_auto2 = DES.fit(optimized= True, use_brute = True)

fcst_auto_pred_Holt2 = fit_Holt_auto2.forecast(len(df_test))
timeseries_evaluation_metrics_func(df_test, fcst_auto_pred_Holt2)

#%%

#Scaled data result summary of the Double Exponential Smoothing
fit_Holt_auto.summary()

#%%
#plotting the results for DES with the unscaled dataset

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(df_train, label='Train')
plt.plot(df_test, label='Test')
plt.plot(fcst_auto_pred_Holt2, label='Double Exponential Smoothing using optimized=True')
plt.legend(loc='best')
plt.show()


#%%

# Moving on to Recursive Feature Selection
# Define X as all dependent variables(features) to be selected from & target as the independent variable
X_input = BTC_dataset.drop('Close_Price', axis=1)
target = BTC_dataset['Close_Price']

#Chose  a random state seed for reproducibility
rfc = DecisionTreeRegressor(random_state=45)
rfecv = RFECV(estimator=rfc, step=1, cv=5, scoring='r2')

#redifine X as a 
rfecv.fit(X_input, target)

# to see how many features are optimal to produce the best accuracy
print('Optimal number of features: {}'.format(rfecv.n_features_))

#plot the accuracy obtained with every number of features used:

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()

#%%
#which features are considered to be least important and drop them with this snippet:
print(np.where(rfecv.support_ == False)[0])

X_input.drop(X_input.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

#feature_importance
print(rfecv.estimator_.feature_importances_)

dset = pd.DataFrame()
dset['attr'] = X_input.columns
dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)


plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()

dset  
#%%

# Create a an input variable of previous days closing price
Input_Price = BTC_dataset['Close_Price'].shift(1)


Input_Price = pd.DataFrame(Input_Price)

Input_Price.columns = ['Input_Price']

#Adding Input_Price column to On_Chain_Metrics
BTC_dataset["Input_Price"] = Input_Price

BTC_dataset

#%%
'''select those attributes deemed relevant by the RFECV model in the previous code'''

#select those attributes deemed relevant by the RFECV model in the previous code # 'Input_Price',

to_keep = X_input.columns.tolist()
to_keep.append('Input_Price')
to_keep.append('Close_Price')
BTC_dataset2 = BTC_dataset[to_keep]

# Drop NaN values in BTC_dataset
BTC_dataset2 = BTC_dataset2.dropna()

print(BTC_dataset2.head)

#%%

#Performance analytics core chart


sb.pairplot(BTC_dataset2, corner=True)

#%%
correlation_matrix = BTC_dataset2

#Create new Correlation Matrix: rho
#call the correlation function with method Spearman & round the values to two decimals
rho = correlation_matrix.corr(method="spearman").round(2)

#get the p values
pval = correlation_matrix.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)

#set the p values, *** for less than 0.001, ** for less than 0.01, * for less than 0.05
p = pval.applymap(lambda x: ''.join(['*' for t in [0.001,0.01,0.05] if x<=t]))

#Corr_Data_Pvalue below gives the dataframe with correlation coefficients and p values
Corr_Data_Pvalue = rho.astype(str) + p

print(Corr_Data_Pvalue)

#%%
#Creating a correlation matrix between all the features in the dataset for a heatmap

#Create the matrix based on the spearman correlation method
matrix = np.triu(correlation_matrix.corr(method="spearman"))

#convert to array for the heatmap
Heatmap_Dataset = Corr_Data_Pvalue.to_numpy()

#plot the heatmap
plt.figure(figsize=(30,16))
sns.heatmap(rho, annot = Heatmap_Dataset, fmt='', vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', mask = matrix)

#%%
#Correlation dataframe previously derived from Spearman method duplicated
#because the loop function doesn't work with the Close_Price with Pvalues (asterix) in thesame column in dataframe: Corr_Data_Pvalue,
#rho will be used to create Spearman's correlation table Appended as a column in the dataframe as "Spearman_Verdict"
#https://www.statstutor.ac.uk/resources/uploaded/spearmans.pdf

Spearman_Verdict = []

for value in rho["Close_Price"]:

    if value > 0:
      if value < .20:
        Spearman_Verdict.append("Very Weak") 
      elif value < .40:
        Spearman_Verdict.append("Weak")
      elif value < .60: 
        Spearman_Verdict.append("moderate")
      elif value < .80:
        Spearman_Verdict.append("strong")
      else:
        Spearman_Verdict.append("Very Strong")
    else:
      if value > -.20:
        Spearman_Verdict.append("Very Weak") 
      elif value > -.40:
        Spearman_Verdict.append("Weak")
      elif value > -.60: 
        Spearman_Verdict.append("moderate")
      elif value > -.80:
        Spearman_Verdict.append("strong")
      else:
        Spearman_Verdict.append("Very Strong")

rho["Spearman_Verdict"] = Spearman_Verdict

#Select Columns: Close_Price & Spearman_Verdict
rho =rho[["Close_Price","Spearman_Verdict"]]

#Drop the first 4 rows of the dataframe: Open, High, Low & Close_Price, 
#rho = rho.drop(rho.iloc[[0]].index)

#Name index: 'Metrics'
rho.index.name = 'Metrics'

print('We have',rho.shape[0],'rows and',rho.shape[1],'columns in this DataFrame')
rho

#%%

# Plot historic graph

# frist turn dataset into array
BTC_dataset2_np = BTC_dataset2.to_numpy()
print(BTC_dataset2_np)
top_plt = plt.subplot2grid((5,4), (0, 0), rowspan=3, colspan=4)
top_plt.plot(BTC_dataset2.index, BTC_dataset2["Close_Price"])
plt.title('Historical BTC price [29-04-2013 to 15-05-2022]')
bottom_plt = plt.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
bottom_plt.bar(BTC_dataset2.index, BTC_dataset2['Token_Circulation'])
plt.title('BTC Token Circulation', y=-0.60)
plt.gcf().set_size_inches(18,10)

#%%
#Data split into X and Y in preperation for RNN LSTM

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
    # find the end of this pattern
    end_ix = i + n_steps
    # check if we are beyond the dataset
    if end_ix > len(sequences):
      break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)
  # choose 14 days time steps to look back 
n_steps = 7

# convert into input/output
X, y = split_sequences(BTC_dataset2_np, n_steps)

print(X.shape, y.shape)

n_features = X.shape[2]

for i in range(len(X[:1])):
  print(X[i],'y:',y[i])

#Splitting dataset into training, testing and validation to be used for multi linear regression model
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(X, y, test_size=0.20, random_state=42) #0.16 & #0.20

#Minmax scaling -> train dataset
sc_LR = MinMaxScaler(feature_range= (0, 1))
sc1_LR = MinMaxScaler(feature_range= (0, 1))

X_train_scaled_LR = sc_LR.fit_transform(X_train_LR.reshape(-1,1)).reshape(X_train_LR.shape)
y_train_scaled_LR = sc1_LR.fit_transform(y_train_LR.reshape(-1,1)).reshape(y_train_LR.shape)

#Minmax scaling -> test dataset
X_test_scaled_LR = sc_LR.transform(X_test_LR.reshape(-1,1)).reshape(X_test_LR.shape)
y_test_scaled_LR = sc1_LR.transform(y_test_LR.reshape(-1,1)).reshape(y_test_LR.shape)

print(("length of training dataset:"), len(X_train_scaled_LR), (" & "), ("length of testing dataset:"), len(X_test_scaled_LR))
print(X_train_scaled_LR.shape)

print(y_train_scaled_LR.shape)

#%%
#Reshaping train data from 3d to 2d

nsamples, nx, ny = X_train_scaled_LR.shape
d2_train_scaled = X_train_scaled_LR.reshape((nsamples,nx*ny))

#Reshaping test data from 3d to 2d

nsamples, nx, ny = X_test_scaled_LR.shape
d2_test_scaled = X_test_scaled_LR.reshape((nsamples,nx*ny))

#%%
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(d2_train_scaled,y_train_scaled_LR)

#%%
y_prediction =  LR.predict(d2_test_scaled)
y_prediction

print("mean_squared_error is==",mean_squared_error(y_test_scaled_LR,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test_scaled_LR,y_prediction)))

#%%

pred_df=pd.DataFrame({'Actual Value':y_test_scaled_LR,'Predicted Value':y_prediction,'Difference':y_test_scaled_LR-y_prediction})
pred_df['sq_er'] = pred_df['Difference']*pred_df['Difference']
pred_df['BTC_RMSE_LIN'] = np.sqrt(pred_df['sq_er'])
pred_df 

#%%

# Plotting the graph with scatter plot multi linear regression
plt.figure(figsize=(10,10))
plt.scatter(y_test_scaled_LR, y_prediction, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prediction), max(y_test_scaled_LR))
p2 = min(min(y_prediction), min(y_test_scaled_LR))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

#%%

# Plotting the graph with boxplot

fig, ax = plt.subplots(figsize=(12, 7))
# Remove top and right border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# Remove y-axis tick marks
ax.yaxis.set_ticks_position('none')
# Add major gridlines in the y-axis
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
# Set plot title
ax.set_title('Distribution of RMSE for Multi Linear Regression')
# Set labels for the boxplot
dataset = pred_df['BTC_RMSE_LIN']
labels = pred_df['BTC_RMSE_LIN']
ax.boxplot(dataset)
plt.show()


#%%
#Splitting dataset into training, testing and validation to be used for RNN LSTM model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=42) #0.16 & #0.20

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1) # 0.25 x 0.8 = 0.2


"""The reason for using fit and then transform with train data is a) Fit would calculate mean,var etc of train set and
 then try to fit the model to data b) post which transform is going to convert data as per the fitted model.
If you use fit again with test set this is going to add bias to your model.
"""

#Minmax scaling -> train dataset
sc = MinMaxScaler(feature_range= (0, 1))
sc1 = MinMaxScaler(feature_range= (0, 1))

X_train_scaled = sc.fit_transform(X_train.reshape(-1,1)).reshape(X_train.shape)
y_train_scaled = sc1.fit_transform(y_train.reshape(-1,1)).reshape(y_train.shape)

#Minmax scaling 3.0 -> val dataset
X_val_scaled = sc.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
y_val_scaled = sc1.transform(y_val.reshape(-1,1)).reshape(y_val.shape)

#Minmax scaling 3.0 -> test dataset
X_test_scaled = sc.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
y_test_scaled = sc1.transform(y_test.reshape(-1,1)).reshape(y_test.shape)

print(("length of training dataset:"), len(X_train_scaled), (" & "), ("length of testing dataset:"), len(X_test_scaled), ("length of validation dataset:"), len(X_val_scaled))

print(X_train_scaled.shape)

print(y_test_scaled.shape)


#%%
# Keras Tuner for the outlay of the RNN LSTM model
# The neurons range specified in the Keras model is kept small as the data is small and 
# the model is not going to be overfitted.

def build_model(hp):
    model = Sequential()
    #model.add(LSTM(input_shape=(X_train.shape[-2:])))
    model.add(keras.layers.Input(shape=X_train_scaled.shape[-2:]))
    for i in range(hp.Int('n_layers_LSTM', 0, 2)):  # adding variation of layers.
          model.add(layers.LSTM(hp.Int(f'LSTM_{i}_units',
                                 min_value=2,
                                  max_value=50,
                                  step=2),activation = "tanh" , return_sequences = True))
    model.add(layers.LSTM(hp.Int('LSTM_out_units', min_value=2, max_value=50, step=2),activation='tanh', return_sequences = False))
    for i in range(hp.Int('n_layers_Dense', 0, 2)):  # adding variation of layers.
          model.add(keras.layers.Dense(hp.Int(f'Dense_{i}_units',
                                    min_value=2,
                                    max_value=50,
                                    step=2),activation = "tanh" ))
  
    #model.add(Dense(hp.Int('Dense_Layer_neurons',min_value=32,max_value=128,step=32), activation='relu'))
    model.add(keras.layers.Dense(1, activation='tanh'))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[0.0001])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=hp_learning_rate),loss='mean_squared_error', metrics = ['mse', 'mae', 'mape', 'acc'])
    return model

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner = RandomSearch(build_model, objective='val_loss', max_trials=90, executions_per_trial=3, overwrite = True, directory = 'my_dir')
tuner.search(X_train_scaled, y_train_scaled, batch_size = 1, epochs=25, callbacks = [es], validation_data=(X_val_scaled, y_val_scaled))

#%%
best_parameters = tuner.get_best_hyperparameters(1)[0]
print(best_parameters.values)

best_model = tuner.get_best_models()[0]
best_model.summary()

#%%
#Grid search for Hyperparameters optimization

def createLSTMModel(dropout_rate=0.1, optimizer='SGD', learning_rate=0.0001, loss='mean_squared_error', activation='relu'):
  #  #Initialising the RNN
  model = Sequential()
  
  model.add(keras.Input(shape=(X_train.shape[-2:])))

  model.add(LSTM(18, activation = 'tanh', return_sequences = False))
  model.add(Dense(46))
  model.add(Dropout(dropout_rate))
  model.add(Dense(4))
  model.add(Dense(1, activation = 'tanh'))

  #Compile the Model
  model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate), loss = loss, metrics = ['mse', 'mae', 'mape', 'acc'])
  return model
  
def split_train_window_length(splits, X, y):
  out = []
  for i in splits:
    out.append((X[-i:], y[-i:], i))

  return out

def grid_search(model, param_grid, X_train, y_train):
  splits = param_grid.pop('100', None)
  if splits is None:
    splits = [len(X_train)]

  models = []
  # grid search for different input time windows
  for X, y, split in split_train_window_length(splits, X_train, y_train):
    # split train data into cross validation sets, for grid search
    cv = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=3).split(X):
      cv.append((train_idx, test_idx))
    
    model_LSTM = KerasRegressor(build_fn=createLSTMModel)

    GridLSTM = GridLSTM = GridSearchCV(estimator=model_LSTM,
                     param_grid=grid_param_LSTM,
                     scoring='neg_mean_squared_error',
                     cv=3)
    
    grid_res = GridLSTM.fit(X_train_scaled,y_train_scaled, validation_data=(X_val_scaled, y_val_scaled),callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True)])
    models.append((grid_res, split))

  best_model = sorted(models, key= lambda x: x[0].best_score_, reverse=True)[0]

  best_model[0].best_params_[10] = best_model[1]
  
  return best_model[0]

model_LSTM = KerasRegressor(build_fn=createLSTMModel)

#200 regels grid search testen

#'adam' said to be best in CV
grid_param_LSTM = {
    'batch_size': [1],
    'epochs': [24, 25, 26],   
    'optimizer': ['SGD', 'adam'],
    'activation': ['tanh','sigmoid'],
}

#%%
grid_search_output = grid_search(model_LSTM, grid_param_LSTM, X_train_scaled, y_train_scaled)
model = grid_search_output.best_estimator_.model # trained model with best parameters
model_params = grid_search_output.best_params_ # best grid searched parameters
 
#%%

print(model_params)


# %%

# Visualize history
# Plot history: Loss
plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

#%%

model = Sequential()

model.add(keras.Input(shape=(X_train.shape[-2:])))
model.add(LSTM(18, activation = 'tanh', return_sequences = False))
model.add(Dropout(0.1))
model.add(Dense(46))
model.add(Dense(4))
model.add(Dense(1, activation = 'tanh'))


#Compile the Model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001), loss = tf.keras.metrics.mean_squared_error,
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

#Fit data to model
es = EarlyStopping(monitor='val_loss', patience=11, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train_scaled, epochs=25, batch_size=1, validation_data=(X_val_scaled, y_val_scaled), callbacks=[es])
#%%

# Plot training
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    
plot_train_history(history,
'Single Step Training and validation loss')


#%%
#Prediction values from model
prediction = model.predict(X_test_scaled)

#%%
#Plotting the actual and prediction value
plt.plot(y_test_scaled, color='blue', label='Actual')
plt.plot(prediction, color='red', label='Prediction')
plt.legend()
plt.show()

#%%

#Plotting the actual and prediction value
plt.figure(figsize=(10,10))
plt.scatter(y_test_scaled, prediction, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(prediction), max(y_test_scaled))
p2 = min(min(prediction), min(y_test_scaled))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


#%%

# Create a dataframe with RMSLE values
Predicted_values = pd.DataFrame(prediction[:528])
Predicted_values.rename(columns={0: 'predicted'}, inplace=True)
Actual_values = pd.DataFrame(y_test_scaled[:])
Actual_values.rename(columns={0: 'Actual'}, inplace=True)

All_values = pd.concat([Actual_values, Predicted_values], axis=1)
All_values ["RMSE"] = All_values.apply(lambda x: mean_squared_error(All_values.loc[:x.name, 'Actual'].tolist(), All_values.loc[:x.name, 'predicted'].tolist(), squared=False), axis = 1)
All_values 

#%%

#Boxplot of RMSE for RNN LSTM

fig, ax = plt.subplots(figsize=(12, 7))
# Remove top and right border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# Remove y-axis tick marks
ax.yaxis.set_ticks_position('none')
# Add major gridlines in the y-axis
ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
# Set plot title
ax.set_title('Distribution of RMSE for RNN LSTM')
# Set names as labels for the boxplot
dataset = All_values["RMSE"]
labels = All_values
ax.boxplot(dataset)
plt.show()

#%%
#K-Fold Cross Validation
# Define X as all dependent variables(features) to be selected from & target as the independent variable
n_splits = 10

#BTC_dataset2.values

scaler = MinMaxScaler(feature_range=(0, 1))
BTC_dataset2_scaled = scaler.fit_transform(BTC_dataset2)

X_CV = BTC_dataset2_scaled[:-1, 0:-1]
Y_CV = BTC_dataset2_scaled[:-1, -1]

#transform data

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X_CV, y=None, groups=None):
        n_samples = len(X_CV)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


#%%

def createLSTMModel():
                 
  #Initialising the RNN
  model = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
  model.add(keras.Input(shape=(X_train.shape[-2:])))
  
  model.add(LSTM(18, activation = 'tanh', return_sequences = False))
  model.add(Dropout(0.1))
  model.add(Dense(46))
  model.add(Dense(4))
  model.add(Dense(1, activation = 'tanh'))
  
  #Compile the Model
  model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0007), loss = tf.keras.metrics.mean_squared_error,
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

  return model

model_LSTM = KerasRegressor(build_fn=createLSTMModel)
BTSCV = BlockingTimeSeriesSplit(n_splits) 

rmse_block = np.sqrt(-cross_val_score(model_LSTM, X_train_scaled, y_train_scaled, cv=BTSCV, scoring='neg_mean_squared_error'))

print('The root mean squared errors for each of the 10 splits are:', rmse_block)
print('The mean RMSE for the folds is: ', (sum(rmse_block)/10))


#%%

df_results = pd.DataFrame(rmse_block)
df_results = df_results.rename(columns={0:'BTC_RMSE_RNN'})
df_results.to_csv('BITCOIN_ON_CHAIN_LSTM.csv')
#%%

