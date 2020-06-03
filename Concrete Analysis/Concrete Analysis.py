"""
Created on Fri May 15 14:20:08 2020
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
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense # Activation,Layer,Lambda
pd.set_option('display.max_columns',None)

# ===============================================================================================
# Business Problem - Prepare a model for strength of concrete data using Neural Networks
# ===============================================================================================

concrete = pd.read_csv('concrete.csv')
concrete.head()
concrete.isnull().sum()
concrete.info()

# Summary
concrete.describe()

# Histogram
concrete.hist(grid = False)

# Boxplot
concrete.boxplot(patch_artist = True, grid = False, notch = True)   

# Normal Quantile-Quantile plot
stats.probplot(concrete.cement,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(concrete.slag,dist = 'norm',plot = pylab) 
stats.probplot(concrete.ash,dist = 'norm',plot = pylab)
stats.probplot(concrete.water,dist = 'norm',plot = pylab) 
stats.probplot(concrete.superplastic,dist = 'norm',plot = pylab) 
stats.probplot(concrete.coarseagg,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(concrete.fineagg,dist = 'norm',plot = pylab) # Normal distribution
stats.probplot(concrete.age,dist = 'norm',plot = pylab) 
stats.probplot(concrete.strength,dist = 'norm',plot = pylab) # Normal distribution

# Pairplot
sns.pairplot(concrete, corner = True, diag_kind = "kde")

# Heat map and Correlation Coifficient 
sns.heatmap(concrete.corr(), annot = True, cmap = 'Greens')

##################################### - Splitting data - ############################################

# Splitting in X and y
X = concrete.iloc[:,0:8]
y = concrete.iloc[:,8]

# Splitting in Train and Test 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)

######################################## - Fitting Model - ##########################################

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

#  Giving inputs to the model 8 = input dimension, 100 = neurons/nodes, 1 = output dimension
first_model = prep_model([8,100,1])
first_model.fit(np.array(X_train),np.array(y_train),epochs=172)

# Predication train data
pred_train = first_model.predict(np.array(X_train))
pred_train = pd.Series([i[0] for i in pred_train])

# RMSE train data
rmse_train = np.sqrt(np.mean((pred_train-y_train)**2)) # 23

# Accuracy train data
np.corrcoef(pred_train,y_train) # 0.96

# Visualising train data
plt.plot(pred_train,y_train,"bo")

# Predicting on test data
pred_test = first_model.predict(np.array(X_test))
pred_test = pd.Series([i[0] for i in pred_test])

# RMSE test data
rmse_test = np.sqrt(np.mean((pred_test-y_test)**2)) # 23

# Accuracy test data
np.corrcoef(pred_test,y_test) #0.95

# Visualising test data
plt.plot(pred_test,y_test,"go")

                         # ---------------------------------------------------- #
