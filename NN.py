import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

seed = 42
df = pd.read_csv("nn_data.csv")
df = df.sample(frac=0.5)

# pairplots
sns.pairplot(df.sample(frac=0.5), hue='PID')
plt.savefig("pp_data.png")

#print(df)                               # first look at the data
Y = df.iloc[:,3]
X = df.iloc[:,0:3]
# print(X.shape)

#print(df.isna().sum())                  # check if anything is missing
#df = df.dropna()                       # drop any null values in data
#print(X.describe())                     # statistical summary of the variables
print(df.groupby(Y).size())             # check for class imbalance

# Normalize features within range 0 to 1
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)
X = pd.DataFrame(X)

# Convert df to array values
X = X.values
Y = Y.values

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed, shuffle=True)

# One Hot Encoding
y_train = pd.get_dummies(y_train)
y_test_t = pd.get_dummies(y_test)

# Build Neural Network
model = Sequential()
model.add(Dense(20, input_dim=3, activation='relu'))
model.add(Dropout(rate=0.10))
model.add(Dense(10, activation='relu'))
# model.add(Dropout(rate=0.10))
model.add(Dense(4, activation='softmax'))

#compile model here
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.001), metrics=['accuracy'])

#compile model here
history = model.fit(x_train, y_train, validation_data = (x_test, y_test_t), batch_size=30, epochs=30)
print(history.history.keys)

# Model Summary
summary = model.summary()
print("Model Summary")
print(summary)

# PLOTS !!!
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("NN_acc.png")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("NN_loss.png")

# Evaluate Model Performance
results = model.evaluate(x_test, y_test_t)
print("RESULTS")
print(results)

# All about Confusion Matrix
print(y_test)
# Generate predictions
y_pred = np.argmax(model.predict(x_test),axis=-1)+2
print(y_pred)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("CM_NN.png")

# Predicted Outputs
dic = {"P" : x_test.transpose()[0], "TPC" : x_test.transpose()[1], "TOF": x_test.transpose()[2], "PID" : y_pred}
df = pd.DataFrame(data=dic)
print(df)

sns.pairplot(df, hue='PID')
plt.savefig("pp_NN.png")

plt.show()
