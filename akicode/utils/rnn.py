import pdb
import numpy as np

np.random.seed(3)

from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Activation, Masking
from sklearn.metrics import brier_score_loss
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adadelta
from sklearn.metrics import roc_auc_score
from keras.metrics import binary_accuracy, categorical_accuracy
from tensor2_sklearn.preprocessing import Imputer, StandardScaler

def model(num_steps=None, num_feat=None, l1hu=10, l2hu=10, dropout=0.1):
	model = Sequential()
	#model.add(Masking(mask_value=np.NAN, input_shape=(num_steps, num_feat)))
	model.add(LSTM(l1hu, input_shape=(num_steps, num_feat), 
		return_sequences=True))
	model.add(Dropout(dropout))
	model.add(LSTM(l2hu))
	model.add(Dense(2, activation='softmax'))
	#model.add(Activation('softmax'))
	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', binary_accuracy, categorical_accuracy])
	return model

#m = model(num_steps=3, num_feat=1)
#m.fit(X,y, epochs=10)
#m.evaluate(X,y)
#print "roc auc", roc_auc_score(y, m.predict(X))


if __name__ == "__main__":
	X, y = datasets.make_classification(n_samples=1000,
        	n_features=3, n_informative=2,
        	n_redundant=0, weights=[0.5,0.5], random_state=7)
	
	X = X.reshape(1000,3,1)
	y = np.array((y,1-y)).T #must be 2d
	num_steps = 3
	num_feat = 1
	
	clf = KerasClassifier(build_fn=model, num_steps=num_steps, num_feat=num_feat, verbose=0, epochs=10)
	
	clf.fit(X,y)
	#pdb.set_trace()
	print "roc auc", roc_auc_score(y, clf.predict_proba(X))
	print "brier", brier_score_loss(y[:,0], clf.predict_proba(X)[:,0])
	#pdb.set_trace()
	param_grid = {
    	'clf__l1hu': [10, 20, 50],
    	'clf__l2hu': [10, 20, 50],
    	'clf__dropout':[0.25, 0.75]
	}
	
	pipeline = Pipeline([('im',Imputer()), ('sc', StandardScaler()), ('clf', clf)])
	pipeline.fit(X, y)
        print "roc auc", roc_auc_score(y, pipeline.predict_proba(X))
        print "brier", brier_score_loss(y[:,0], pipeline.predict_proba(X)[:,0])
	pdb.set_trace()
	grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=1, scoring='roc_auc')
	grid_result = grid.fit(X, y)
	pdb.set_trace()
	
