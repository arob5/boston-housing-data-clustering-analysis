#
# feature_importance.py
# Attempting to determing relative feature importance in the cluster structure
#
# Last Modified: 3/14/2018
# Modified By: Andrew Roberts
#


import run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def feature_label_corr(X, labels, features):
	for j in range(X.shape[1]):
		print(features[j], ": ", np.corrcoef(X[:, j], labels)[0, 1])

def output_df(X, features, labels):
	X = np.concatenate((X, labels.reshape(labels.shape[0], 1)), 1)
	df = pd.DataFrame(X)
	cols = list(features)
	cols.append("label")
	df.columns = cols	

	df.to_csv("df.csv", index=False)
	 	
def main():
	X, X_norm, features = run.fetch_data()
	nclusters = 2

	kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X_norm)
	feature_label_corr(X_norm, kmeans.labels_, features)

	output_df(X_norm, features, kmeans.labels_)

main()
