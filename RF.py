import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

clf = RandomForestClassifier(n_estimators=100)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Results on Test Data", acc)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.savefig("CM_RF.png")

# Predicted Outputs
dic = {"P" : x_test.transpose()[0], "TPC" : x_test.transpose()[1], "TOF": x_test.transpose()[2], "PID" : y_pred}
df = pd.DataFrame(data=dic)
print(df)

sns.pairplot(df, hue='PID')
plt.savefig("pp_rf.png")
plt.show()
