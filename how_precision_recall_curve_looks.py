from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

nn_clf = KNeighborsClassifier(n_neighbors=100)

X_data, y_data = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.2)

nn_clf.fit(X_train, y_train)
y_pred = nn_clf.predict_proba(X_test)

precision, recall, thresh = precision_recall_curve(y_test, y_pred[:,1])
plt.plot(recall, precision, label="knn")

lr_clf = LogisticRegression()

X_data, y_data = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.2)

lr_clf.fit(X_train, y_train)
y_pred = lr_clf.predict_proba(X_test)

precision, recall, thresh = precision_recall_curve(y_test, y_pred[:,1])
plt.plot(recall, precision, label="logistic")


svc = SVC(probability=True)

X_data, y_data = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.2)

svc.fit(X_train, y_train)
y_pred = svc.predict_proba(X_test)

precision, recall, thresh = precision_recall_curve(y_test, y_pred[:,1])
plt.plot(recall, precision, label="svc")

plt.legend()
plt.show()

