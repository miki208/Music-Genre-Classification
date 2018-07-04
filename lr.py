from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils import shuffle

import pandas as pd

def read_split():
	data = pd.read_csv('data.csv', index_col = 0)

	X = data.drop(['genre'], axis = 1)
	y = data[['genre', 'type']]

	X_train = X[X['type'] == 'train'].drop(['type'], axis = 1)
	X_test = X[X['type'] == 'test'].drop(['type'], axis = 1)

	y_train = y[y['type'] == 'train'].drop(['type'], axis = 1)
	y_test = y[y['type'] == 'test'].drop(['type'], axis = 1)

	X_train, y_train = shuffle(X_train, y_train, random_state=42)

	#data normalization
	scaler = StandardScaler(copy=False)
	scaler.fit_transform(X_train)
	scaler.transform(X_test)

	return (X_train, X_test, y_train['genre'], y_test['genre'])

X_train, X_test, y_train, y_test = read_split()

#model selection ({'C': 0.01})
params = {
    'C' : [10**x for x in range(-5, 6)]
}

clf = GridSearchCV(LogisticRegression(), params, cv = 5, n_jobs = 4)
clf.fit(X_train, y_train)

print(clf.best_params_)

#train algorithm
model = LogisticRegression(C = clf.best_params_['C'])
model.fit(X_train, y_train)

#model evaluation on the test data (score = 0.500625)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#model evaluation using cross validation (score = 0.5345)
model = LogisticRegression(C = clf.best_params_['C'])

data = pd.read_csv('data.csv', index_col = 0)

X = data.drop(['genre', 'type'], axis = 1)
y = data['genre']

scaler = StandardScaler(copy=False)
scaler.fit_transform(X)

print(cross_val_score(model, X, y, cv = 5, n_jobs = 4).mean())
