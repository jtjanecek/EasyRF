import os
try:
	os.mkdir('figures')
except:
	pass

import numpy as np
import random
from collections import defaultdict
from copy import deepcopy
import pandas as pd

import subprocess
import os, sys
# For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Feature Selection
from sklearn.feature_selection import SelectKBest

# Data processing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

# Cross validation
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# ROC Curve
from sklearn.metrics import roc_curve, auc, classification_report

# PCA analysis
from sklearn.decomposition import PCA

from scipy.stats import sem

# Precision/recall binary scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score
#font = {'size'   : 1}
mpl.rcParams['figure.dpi'] = 150
#matplotlib.rc('font', **font)
mpl.rcParams.update({'font.size': 5})


import argparse
parser = argparse.ArgumentParser(description='Run the RF')
parser.add_argument('--id', help='The name of this dataset', required=True)
parser.add_argument('--data', help='The input CSV data file', required=True)
parser.add_argument('--outcome', help='The outcome variable', required=True)
parser.add_argument('--cv', help='Number of cross validation iterations', required=True, type=int)
parser.add_argument('--ntrees', help='Number of trees for the RF', required=True, type=int)
cli_args = parser.parse_args()


#################################### Read the data in
print("Reading CSV file ...")
data_file = cli_args.data
data = pd.read_csv(data_file)
##################################### Feature Importance

def run(X, y, labels, the_label, cv_splits):
	print("Running feature important ...")

	feature_importance_map = defaultdict(list)
	model = RandomForestClassifier(n_estimators=500)#, oob_score=True)#, random_state=random_seed + i)
	cv = StratifiedKFold(n_splits=cv_splits)


	for i in range(50):
		print(f" Calculating feature importance for model {i} ...")
		model = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=random_seed + i)
		model.fit(X,y)
		feature_importance = model.feature_importances_
		#make importances relative to max importance
		feature_importance = 100.0 * (feature_importance / feature_importance.max())
		sorted_idx = np.argsort(feature_importance)
		
		feature_importance = feature_importance[sorted_idx]
		feature_labels = np.array(labels)[sorted_idx]
		for i in range(len(feature_labels)):
			feature_importance_map[feature_labels[i]].append(feature_importance[i])

	feats = [[label, np.mean(imps), np.std(imps)] for label, imps in feature_importance_map.items()]
	feats = sorted(feats, key=lambda x: x[1], reverse=True)
	'''
	print(feature_importance_map)
	print("=============")
	for label, avg, err in feats:
		print(label, avg, err)
		'''
	feats = [feats[i] for i in range(min(10,len(feats)))]

	feature_importance = np.array([x[1] for x in feats])
	feature_labels = np.array([x[0] for x in feats])
	feature_sems = np.array([x[2] for x in feats])

	fig, ax = plt.subplots()
	y_pos = np.arange(len(feature_labels))

	ax.barh(y_pos, feature_importance.astype(float), xerr=feature_sems, tick_label=feature_labels,align='center')
	#ax.set_yticks(y_pos)
	#ax.set_yticklabels(feature_labels)
	ax.invert_yaxis()
	ax.set_xlabel('Relative Importance')
	ax.set_title('Gini impurity importance - N={}'.format(len(X)))
	plt.savefig('figures/{}_Feature_importance.png'.format(the_label))
	#plt.show()


	cv = StratifiedKFold(n_splits=cv_splits)
	from numpy import interp
	classifier = model

	tprs = []
	aucs = []
	from sklearn.metrics import recall_score
	spcs = [] # Precision
	sens = [] # Recall
	mean_fpr = np.linspace(0, 1, 100)

	plt.figure(figsize=(8, 6), dpi=120)

	print("Calculating ROC across CVs ...")

	cv = StratifiedKFold(n_splits=cv_splits,random_state=random_seed + i,shuffle=True)
	for train, test in cv.split(X,y):
		print(" Running CV ...")
		classifier = RandomForestClassifier(n_estimators=500, random_state=random_seed + i)
		classifier.fit(X[train], y[train])
		probas_ = classifier.predict_proba(X[test])
		predicted = classifier.predict(X[test])
		#print(predicted)
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
		roc_auc = auc(fpr, tpr)
		report = classification_report(y[test], predicted, output_dict=True)
		spcs.append(report['0']['recall'])
		sens.append(report['1']['recall'])
		aucs.append(roc_auc)		
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0

	sens = np.array(sens).flatten()
	spcs = np.array(spcs).flatten()
	# Chance
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
		 label='Chance', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	mean_auc = np.mean(aucs)
	sem_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
		 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, sem_auc),
		 lw=2, alpha=.8)

		#std_tpr = np.std(tprs, axis=0) / np.sqrt(len(tprs))

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
				 label=r'$\pm$ SD')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate', fontsize=22)
	plt.ylabel('True Positive Rate', fontsize=22)
	plt.title('ROC - N={}'.format(len(X)), fontsize=22)
	plt.legend(loc="lower right", prop={'size': 8})
	plt.tick_params(axis='both', which='major', labelsize=16)
	#plt.tight_layout()
	#plt.show()
	plt.savefig('figures/{}_roc.png'.format(the_label))

	print(" ==================================== RESULTS ===============================")
	print(the_label)
	print('SPCS (mean and std): ', np.mean(spcs), np.std(spcs))
	print('SENS (mean and std): ', np.mean(sens), np.std(sens))
	print("AUCS (mean and std): ", np.mean(aucs), np.std(aucs))
	print("All AUCs calculated in CVs:")
	for a in aucs:
		print(a)


cv_splits = 8
random_seed = 123

y = data[cli_args.outcome].values
data = data.drop([cli_args.outcome], axis=1)

labels = list(data.columns)
X = (data.values).astype(float)
run(X, y, labels, cli_args.id, cli_args.cv)


