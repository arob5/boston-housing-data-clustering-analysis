#
# pca_analysis.py
# Using PCA in conjunction with Kmeans on Boston Housing Dataset
#
# Last Modified: 3/15/2018
# Modified By: Andrew Roberts
#

import run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def pca(X, norm=False):
	if not norm:
		mu = X.mean(0)
		X = X - mu

	n = X.shape[0]
	cov_mat = (X.T @ X) / n
	w, v = np.linalg.eig(cov_mat)
	v = v[:, np.argsort(w)]

	scores = X @ v
	X_2 = scores[:, -2:] #@ v[:, -2:].T
	
	return X_2

def show_2D_clusters(X, c, axis1=0, axis2=1):
	"""
	Visualize the different clusters using color encoding.
	:param X: An n-by-d numpy array representing n points in R^d
	:param c: A list (or numpy array) of n elements. The ith entry, c[i], must
	be an integer representing the index of the cluster that point i belongs
	to.
	"""
	plt.figure()
	plt.scatter(X[:, axis1], X[:, axis2], c=c)
	plt.title("KMeans on 2D PCA Projection")
	plt.xlabel("PC 1")
	plt.ylabel("PC 2")

def main():
	X, X_norm, features = run.fetch_data()
	X_norm = np.delete(X_norm, [3, 8], 1)	
	X = np.delete(X, [3, 8], 1)	
	features = np.delete(features, [3, 8], 0)
		
	nclusters = 2
	kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X_norm)
	X_2 = pca(X_norm, True)
	show_2D_clusters(X_2, kmeans.labels_)

	plt.show()	
	
main()
