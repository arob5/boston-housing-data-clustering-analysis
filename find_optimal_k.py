#
# find_optimal_k.py
# Purpose: Find optimal number of clusters for Boston housing dataset
#
# Last Modified: 3/15/2018
# Modified By: Andrew Roberts
#

import run
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow_method(X, k_min, k_max):
	sse = []

	for k in range(k_min, k_max):
		kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
		sse.append(kmeans.inertia_)

	plt.plot(np.arange(k_min, k_max), sse, marker="o")	
	plt.title("SSE as a function of k")
	plt.xlabel("k")
	plt.ylabel("SSE")
	plt.xticks(np.arange(k_min, k_max))

def silhouette_method(X, k_min, k_max):
	s_scores = []

	for k in range(k_min, k_max):	
		kmeans = KMeans(n_clusters=k, random_state=0)
		cluster_labels = kmeans.fit_predict(X)

		silhouette_avg = silhouette_score(X, cluster_labels)
		s_scores.append(silhouette_avg)

	plt.plot(np.arange(k_min, k_max), s_scores, marker="o")
	plt.title("Average silhouette scores as a function of k")
	plt.xlabel("k")
	plt.ylabel("Average Silhouette Coefficient")
	plt.xticks(np.arange(k_min, k_max))
	
def main():
	X, X_norm, features = run.fetch_data()
	X_norm = np.delete(X_norm, [3, 8], 1)

	elbow_method(X_norm, 1, 8)
	silhouette_method(X_norm, 2, 8)
	plt.show()
	
main()
