
# # Import all the necessary library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import seaborn as sns
import pandas as pd
from sklearn import metrics as met
import warnings

warnings.filterwarnings(action="ignore")

df = pd.read_csv('chronic_kidney_disease_full.csv')
data = df
print("\n\nSample Ckd dataset head(5) :- \n\n", data.head(5))
# ******  Print the shape of file  **************
print("\n\nShape of the ckd dataset  data.shape = ", end="")
print(data.shape)
print("\nckd data description : \n")
print(data.describe())

# DAta Mapping
data['class'] = data['class'].map({'ckd': 1, 'notckd': 0})
data['htn'] = data['htn'].map({'yes': 1, 'no': 0})
data['dm'] = data['dm'].map({'yes': 1, 'no': 0})
data['cad'] = data['cad'].map({'yes': 1, 'no': 0})
data['appet'] = data['appet'].map({'good': 1, 'poor': 0})
data['ane'] = data['ane'].map({'yes': 1, 'no': 0})
data['pe'] = data['pe'].map({'yes': 1, 'no': 0})
data['ba'] = data['ba'].map({'present': 1, 'notpresent': 0})
data['pcc'] = data['pcc'].map({'present': 1, 'notpresent': 0})
data['pc'] = data['pc'].map({'abnormal': 1, 'normal': 0})
data['rbc'] = data['rbc'].map({'abnormal': 1, 'normal': 0})

print(data['class'].value_counts())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
print(data.shape[0])
print(data.shape[0], data.dropna().shape[0])
data.dropna(inplace=True)
print(data.shape)
print(data.info())

# ************** Plotting the histogram of for no. of the ckd or not ckd diagnosis ********
plt.hist(data['class'])
plt.title('class(ckd=1 , notckd=0)')
plt.show()

#Plotting Correletion matrix
plt.figure(figsize=(19, 19))
plt.title('CKD Attributes Correlation')
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')  # looking for strong correlations with "class" row
plt.show()


# Density Plot diagram
data.plot(kind='density', subplots=True, layout=(5,5), sharex=False, )
plt.show()
def plot_cm(y_true, y_pred):
    cm = met.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()
print("***********Logistic Algorithm************")
# Logistic Algorithm
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
X = data.iloc[:, :-1]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logreg.fit(X_train, y_train)

test_pred = logreg.predict(X_test)
train_pred = logreg.predict(X_train)
from sklearn.metrics import accuracy_score, confusion_matrix

print('Train Accuracy: ', accuracy_score(y_train, train_pred))
print('Test Accuracy: ', accuracy_score(y_test, test_pred))

pd.DataFrame(logreg.coef_, columns=X.columns)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')
plot_cm(y_test, test_pred)

print("***********SVM***************")
from sklearn.svm import SVC
svc = SVC()
X = data.iloc[:, :-1]
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
svc.fit(X_train, y_train)
test_pred = svc.predict(X_test)
train_pred = svc.predict(X_train)
from sklearn.metrics import accuracy_score, confusion_matrix

print('Train Accuracy: ', accuracy_score(y_train, train_pred))
print('Test Accuracy: ', accuracy_score(y_test, test_pred))


tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')
plot_cm(y_test, test_pred)
print("*******K-Nearest Neighbors Classifier********")
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
X = data.iloc[:, :-1]
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
knn.fit(X_train, y_train)
test_pred = knn.predict(X_test)
train_pred = knn.predict(X_train)
from sklearn.metrics import accuracy_score, confusion_matrix

print('Train Accuracy: ', accuracy_score(y_train, train_pred))
print('Test Accuracy: ', accuracy_score(y_test, test_pred))


tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

print(f'True Neg: {tn}')
print(f'False Pos: {fp}')
print(f'False Neg: {fn}')
print(f'True Pos: {tp}')
plot_cm(y_test, test_pred)