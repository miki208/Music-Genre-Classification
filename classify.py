from sklearn.externals import joblib
from features import compute_features #postojeci kod koji je koriscen za izdvajanje feature-a, uz male izmene (utils.py takodje)

import pandas as pd

from sys import argv

def extract_features(filename):
	return joblib.load('scalers/scaler.pkl').transform(compute_features(filename).values.reshape(1, -1))

if len(argv) != 3:
	print('Usage: python classify.py {file} {model}')
	exit()

#extract features
instance = extract_features(argv[1])

#load model
model = joblib.load(argv[2])
prediction = model.predict(instance)

#print result
print('Instance {} is classified as {}.'.format(argv[1], prediction[0]))
