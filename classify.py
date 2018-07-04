from sklearn.externals import joblib

import pandas as pd

def extract_features(filename, model_name):
	pass

if len(argv) != 3:
	print('Usage: python classify.py {file} {model}')
	exit()

#extract features
instance = extract_features(argv[1], argv[2])

#load model
model = joblib.load(argv[2])
prediction = model.predict(instance)

#print result
print('Instance {} is classified as {}.'.format(argv[1], prediction[0]))
