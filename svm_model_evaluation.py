from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

data = pd.read_csv('data.csv', index_col = 0)

X = data.drop(['genre', 'type'], axis = 1)
y = data['genre']

scaler = StandardScaler(copy=False)
scaler.fit_transform(X)

model = SVC()

print(cross_val_score(model, X, y, cv = 3, n_jobs = 4).mean()) #0.5346611581641523
