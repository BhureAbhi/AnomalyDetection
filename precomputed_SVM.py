import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

K=np.genfromtxt("/media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED1/Graph/K_X.csv",delimiter=",")
K1=np.genfromtxt("/media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED1/Graph/K_X1.csv",delimiter=",")
y=np.genfromtxt("/media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED1/Graph/K_Y.csv",delimiter=",")
y1=np.genfromtxt("/media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED1/Graph/K_Y1.csv",delimiter=",")

clf = svm.SVC(kernel='precomputed')

print(len(y1))
print(len(y))
clf.fit(K, y)
svm.SVC(C=8.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='precomputed', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.01, verbose=False)

y_pred=clf.predict(K)
print(accuracy_score(y,y_pred))

y1_pred=clf.predict(K1)
print(accuracy_score(y1,y1_pred))
