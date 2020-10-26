import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets import load_boston
from sklearn.cluster import KMeans
import pandas as pd


def z_normalize(X):
    """
    Compute z-normalized data matrix.
    :param X: An n-by-d numpy array representing n points in R^d
    :return: An n-by-d numpy array representing n points in R^d that have been
    z-normalized.
    """
    mu = X.mean(0)
    sg = X.std(0)

    X = (X - mu) / sg
    return X

def show_2D_clusters(X, c, plot, name, axis1, axis2, xlab, ylab):
	"""
	Visualize the different clusters using color encoding.
	:param X: An n-by-d numpy array representing n points in R^d
	:param c: A list (or numpy array) of n elements. The ith entry, c[i], must
	be an integer representing the index of the cluster that point i belongs
	to.
	"""
	'''
	plt.scatter(X[:, axis1], X[:, axis2], c=c)
	plt.title("Normalized clustered data projected onto axes 2 and 4")
	plt.xlabel("INDUS")
	plt.ylabel("RM")
	'''
	
	plot.scatter(X[:, axis1], X[:, axis2], c=c)
	plot.set_title(name)
	plot.set_xlabel(xlab)
	plot.set_ylabel(ylab)
	#plot.xlabel("INDUS")
	#plot.ylabel("CHAS")
	#plt.title(plot_title)
	#fig_title = plot_title + ".png"
	#plt.savefig("axes 2 4 normalized no cat.png")
	#plt.show(block=False)  # this allows you to have multiple plots open

def fetch_data():
	databunch = load_boston()
	X = databunch.data
	X_normalized = z_normalize(X)
	feature_names = databunch.feature_names

	return X, X_normalized, feature_names

if __name__ == "__main__":
	databunch = load_boston()
	X = databunch.data
	feature_names = databunch.feature_names
	
	X = np.delete(X, [3, 8], 1)
	feature_names = np.delete(feature_names, [3, 8], 0)
	print(feature_names)

	fig = plt.figure()
	a = fig.add_subplot(1, 2, 1)

	# Do kmeans
	X_norm = z_normalize(X)
	nclusters = 5
	kmeans = KMeans(n_clusters=nclusters).fit(X_norm)
	show_2D_clusters(X, kmeans.labels_, a, "Axes 5 x 7", 5, 7, feature_names[5], feature_names[7])

	a = fig.add_subplot(1, 2, 2)
	#kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X_norm)
	show_2D_clusters(X, kmeans.labels_, a, "Axes 7 and 9", 7, 9, feature_names[7], feature_names[9])


	plt.suptitle("Projection onto axes chosen by eye - normalized")
	plt.savefig("Projection onto axes chosen by eye norm k5.png")
	# Keep plots open
	plt.show()
