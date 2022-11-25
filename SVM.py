import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

seed = 42
df = pd.read_csv("nn_data.csv")
df = df.sample(frac=0.5)
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=seed, shuffle=False)

# Support Vector
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=1, random_state=seed).fit(x_train,y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, random_state=seed).fit(x_train, y_train)

poly_pred = poly.predict(x_test)
rbf_pred = rbf.predict(x_test)
poly_acc = accuracy_score(y_test, poly_pred)
rbf_acc = accuracy_score(y_test, rbf_pred)
print("Polynomial Kernel", poly_acc)
print("RBF Kernel", rbf_acc)

ConfusionMatrixDisplay.from_predictions(y_test, poly_pred)
plt.title("Polynomial Predictions")
plt.savefig("CM_Poly_SVM.png")
ConfusionMatrixDisplay.from_predictions(y_test, rbf_pred)
plt.title("RBF Predictions")
plt.savefig("CM_RBF_SVM.png")

# Predicted Outputs
dic = {"P" : x_test.transpose()[0], "TPC" : x_test.transpose()[1], "TOF": x_test.transpose()[2], "PID" : poly_pred}
df = pd.DataFrame(data=dic)
print(df)

sns.pairplot(df, hue='PID')
plt.savefig("pp_svm.png")
plt.show()
