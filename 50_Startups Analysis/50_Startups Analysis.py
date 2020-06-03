"""
Created on Fri May 15 10:59:11 2020
@author: DESHMUKH
NEURAL NETWORK 
"""
# pip install keras
# pip install tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense # Activation,Layer,Lambda

# ===============================================================================================
# Business Problem - Build a Neural Network Model for 50_startups data to predict profit.
# ===============================================================================================

startups = pd.read_csv('50_Startups.csv')
startups.head()
startups.isnull().sum()
startups.info()
startups.columns = "RandD","admin","marketing","state","profit"
startups = startups.replace(" ","_",regex = True)

# Summary
startups.describe()

# Histogram
startups.hist(grid = False)

# Boxplot
startups.boxplot(patch_artist = True, grid = False, notch = True)   

# Normal Quantile-Quantile plot
stats.probplot(startups.RandD,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.admin,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.marketing,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(startups.profit,dist = 'norm',plot = pylab) # Normal distribution

# Pairplot
sns.pairplot(startups, corner = True, diag_kind = "kde")

# Heat map and Correlation Coifficient 
sns.heatmap(startups.corr(), annot = True, cmap = 'Blues')

###################################### - Data Preprocessing - ######################################

# Nomalization of data (as data contain binary value)
#startups.iloc[:,0:3] = normalize(startups.iloc[:,0:3])

# Converting Dummy variable,Removing old columns and adding new dummy column in datafram.
dummy = pd.get_dummies(startups.state,drop_first = True)
startups = startups.drop(['state'],axis = 1)
startups = pd.concat([dummy,startups],axis = 1)

##################################### - Splitting data - ###########################################

# Splitting in X and y
X = startups.iloc[:,0:5]
y = startups.iloc[:,5]

# Splitting in Train and Test 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

######################################## - Fitting Model- ##########################################

# Preparing a function to define the structure ANN network.
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["mse"])
    return (model)

# Giving inputs to the model 5 = input dimension, 100 = neurons/nodes, 1 = output dimension
first_model = prep_model([5,100,1])
first_model.fit(np.array(X_train),np.array(y_train),epochs=200)

# Predication train data
pred_train = first_model.predict(np.array(X_train))
pred_train = pd.Series([i[0] for i in pred_train])

# RMSE train data
rmse_train = np.sqrt(np.mean((pred_train-y_train)**2)) # 61847

# Accuracy train data
np.corrcoef(pred_train,y_train) # 0.96

# Visualising train data
plt.plot(pred_train,y_train,"bo")

# Predicting on test data
pred_test = first_model.predict(np.array(X_test))
pred_test = pd.Series([i[0] for i in pred_test])

# RMSE test data
rmse_test = np.sqrt(np.mean((pred_test-y_test)**2)) # 74017

# Accuracy test data
np.corrcoef(pred_test,y_test) #0.98

# Visualising test data
plt.plot(pred_test,y_test,"ro")

                         # ---------------------------------------------------- #


