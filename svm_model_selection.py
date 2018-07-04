import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

model = SVC()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('svc', model)])

n_comps = [50, 100, 150, 200, 518]

estimator = GridSearchCV(pipe, dict(pca__n_components=n_comps), n_jobs = 4)
estimator.fit(X_train, y_train)

print(estimator.cv_results_) #mean_test_score: [0.46453125, 0.52046875, 0.55328125, 0.5715625 , 0.59421875]
print(estimator.best_params_ ) #n_components: 518 (no pca transformation needed)
print(estimator.best_score_) #0.59421875
