import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def base_model():
    # create model here
    model = Sequential()
    model.add(Dense(20, input_dim=3, activation='relu'))
    model.add(Dropout(rate=0.10))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    #compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model

seed = 42
df = pd.read_csv("nn_data.csv")
df = df.sample(frac=0.5)
Y = df.iloc[:,3]
X = df.iloc[:,0:3]
# print(X)
# print(Y)
#
# print(df.isna().sum())                  # check if anything is missing
# df = df.dropna()                       # drop any null values in data
# print(X.describe())                     # statistical summary of the variables
# print(df.groupby(Y).size())             # check for class imbalance

# Normalize features within range 0 to 1
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)
X = pd.DataFrame(X)

# One Hot Encoding
Y = pd.get_dummies(Y)

# Convert df to array values
X = X.values
Y = Y.values

# Create Keras Classifier and use predefined base_model
estimator = KerasClassifier(build_fn = base_model, epochs = 30, batch_size = 30, verbose = 1, validation_split=0.3)

# KFold Cross Validation
kfold = KFold(n_splits = 6, shuffle = True, random_state = seed)

# Object to describe the result
results = cross_val_score(estimator, X, Y, cv = kfold)

# Results
print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
