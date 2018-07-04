from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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

#model selection ({'n_neighbors': 10, 'p': 1})
params = {
    'n_neighbors' : [5 * x for x in range(1, 6)],
	'p' : [1, 2]
}

clf = GridSearchCV(KNeighborsClassifier(weights = 'distance'), params, cv = 5, n_jobs = 4)
clf.fit(X_train, y_train)

print(clf.best_params_)

#train algorithm
model = KNeighborsClassifier(n_neighbors = clf.best_params_['n_neighbors'], weights = 'distance', p = clf.best_params_['p'])
model.fit(X_train, y_train)

#model evaluation on the test data (score = 0.4025)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#model evaluation using cross validation (score = 0.425625)
model = KNeighborsClassifier(n_neighbors = clf.best_params_['n_neighbors'], weights = 'distance', p = clf.best_params_['p'])

data = pd.read_csv('data.csv', index_col = 0)

X = data.drop(['genre', 'type'], axis = 1)
y = data['genre']

scaler = StandardScaler(copy=False)
scaler.fit_transform(X)

print(cross_val_score(model, X, y, cv = 5, n_jobs = 4).mean())
