from time import time
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
from sklearn_extra.cluster import KMedoids
#import skfuzzy as fuzz
from fcmeans import FCM
#import hdbscan

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import make_scorer

class ClusteringAlgorithms:
   
    def __init__(self, coords_train, X_train_scaled):
        """
        Initialize the ClusteringAlgorithms class.

        Args:
            coordinates_train (numpy.ndarray): Coordinates data for training.
            X_train_scaled (numpy.ndarray): Scaled features data for training.
        """
        self.coord_train    = coords_train
        self.X_train_scaled = X_train_scaled

    def apply_affinity_propagation(self):
        """
        Apply Affinity Propagation clustering to the input data.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        affinity_propagation = AffinityPropagation()
        return affinity_propagation.fit_predict(self.X)

    def apply_agglomerative(self, n_clusters):
        """
        Apply Agglomerative Clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        return agglomerative.fit_predict(self.X)

    def apply_birch(self, n_clusters):
        """
        Apply Birch clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        birch = Birch(n_clusters=n_clusters)
        return birch.fit_predict(self.X)

    def apply_optics(self):
        """
        Apply OPTICS clustering to the input data.

        Args:

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        optics = OPTICS()
        return optics.fit_predict(self.X)

    def apply_dbscan(self, eps, min_samples):
        """
        Apply DBSCAN clustering to the input data.

        Args:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(self.X)

    def apply_gaussian_mixture(self, n_components):
        """
        Apply Gaussian Mixture Model clustering to the input data.

        Args:
            n_components (int): The number of Gaussian mixture components.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        gmm = GaussianMixture(n_components=n_components)
        return gmm.fit_predict(self.X)

    def apply_kmeans(self, n_clusters):
        """
        Apply K-Means clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        t0 = time()
        labels = kmeans.fit_predict(self.X_train_scaled)
        t1 = time()
        print(" [Info]: Fitting time: ",t1-t0, "s")
        return labels
    
    def apply_fkmeans(self, n_clusters):
        """
        Apply Fuzzy K-Means clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        X_fcm = self.X_train_scaled.to_numpy()
        fcm = FCM(n_clusters=n_clusters)
        t0 = time()
        fcm.fit(X_fcm)
        labels = fcm.predict(X_fcm)
        t1 = time()
        centers = fcm.centers
        print(" [Info]: Fitting time: ",t1-t0, "s")
        return labels, centers

    def apply_mean_shift(self):
        """
        Apply Mean Shift clustering to the input data.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        mean_shift = MeanShift()
        return mean_shift.fit_predict(self.X)

    def apply_mini_batch_kmeans(self, n_clusters):
        """
        Apply Mini-Batch K-Means clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        mini_batch_kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        return mini_batch_kmeans.fit_predict(self.X)

    def apply_optics(self):
        """
        Apply OPTICS clustering to the input data.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        optics = OPTICS()
        return optics.fit_predict(self.X)

    def apply_k_medoids(self, n_clusters):
        """
        Apply K-Medoids clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        k_medoids = KMedoids(n_clusters=n_clusters)
        return k_medoids.fit_predict(self.X)

    def apply_feature_agglomeration(self, n_clusters):
        """
        Apply Feature Agglomeration clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        feature_agglomeration = FeatureAgglomeration(n_clusters=n_clusters)
        return feature_agglomeration.fit_predict(self.X)
    
    def apply_spectral(self, n_clusters):
        """
        Apply Spectral clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10)
        return spectral.fit_predict(self.X)


    def apply_spectral_biclustering(self, n_clusters):
        """
        Apply Spectral Biclustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        spectral_biclustering = SpectralBiclustering(n_clusters=n_clusters)
        return spectral_biclustering.fit_predict(self.X)

    def apply_spectral_coclustering(self, n_clusters):
        """
        Apply Spectral Co-clustering to the input data.

        Args:
            n_clusters (int): The number of clusters.

        Returns:
            numpy.ndarray: Cluster labels for the input data.
        """
        spectral_coclustering = SpectralCoclustering(n_clusters=n_clusters)
        return spectral_coclustering.fit_predict(self.X)


    def grid_search_kmeans(self, param_grid):
        # Create a scoring dictionary
        #scoring = {
        #    'silhouette_score': custom_silhouette_score,
        #    'davies_bouldin_score': custom_davies_bouldin_score,
        #    'calinski_harabasz_score': custom_calinski_harabasz_score
        #}
        kmeans = KMeans()
 
        # Define parameter grids for grid search
        #kmeans_param_grid = {'n_clusters':[2, 3, 4, 5, 6, 7, 8, 9, 10],
        #                     'algorithm':['lloyd', 'elkan', 'auto', 'full'],
        #                     'n_init':['auto'],
        #                     'init':['k-means++', 'random'],
        #                     'max_iter':[100, 200, 300]
        #                   }

        # Define the silhouette scorer
        #silhouette_scorer = make_scorer(silhouette_score)
        davies_bouldin_scorer = make_scorer(davies_bouldin_score) 
        calinski_harabasz_scorer = make_scorer(calinski_harabasz_score)

        # Best K-Means parameters: {'algorithm': 'lloyd', 'init': 'k-means++', 'max_iter': 100, 'n_clusters': 2, 'n_init': 'auto'}
        # Create GridSearchCV instance with the silhouette scorer
        #grid_search = GridSearchCV(kmeans, param_grid, cv=5, scoring=silhouette_scorer)
        grid_search = GridSearchCV(kmeans, param_grid, cv=5, scoring=davies_bouldin_scorer)
        #grid_search = GridSearchCV(kmeans, param_grid, cv=5, scoring=calinski_harabasz_scorer)
        
        # Fit the GridSearchCV to the data
        grid_search.fit(self.X_train_scaled)
        
        # Get the best number of clusters from the grid search
        best_n_clusters = grid_search.best_params_['n_clusters']
        
        print(f"The best number of clusters: {best_n_clusters}")
        
        #grid_search = GridSearchCV(estimator=kmeans,
        #                   param_grid=param_grid,
        #                   scoring=scoring,
        #                   refit='silhouette_score',  # Choose the index to optimize
        #                   cv=5,
        #                   n_jobs=-1)
        #grid_search.fit(self.X)
        return grid_search.best_estimator_, grid_search.best_params_
    
    def grid_search_kmeans2(self, param_grid):
        print("")
        print("[Info]: Starting GridSearchCV")
        # Fit K-Means clustering
        kmeans = KMeans()
        grid_search = GridSearchCV(kmeans, param_grid, cv=5)
        cluster_labels = grid_search.fit(self.X_train_scaled)
        
        # Compute the silhouette score
        silhouette_avg = silhouette_score(self.X_train_scaled, cluster_labels)
        
        # Compute the silhouette scores for each data point
        sample_silhouette_values = silhouette_samples(self.X_train_scaled, cluster_labels)
        
        # Create a plot to visualize the silhouette scores
        fig, ax = plt.subplots()
        y_lower = 10  # Adjust this value for spacing between clusters
        
        for i in range(len(np.unique(cluster_labels))):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / len(np.unique(cluster_labels)))
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for the next plot
            y_lower = y_upper + 10  # Adjust this value for spacing between clusters
        
        # Set labels and title
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        ax.set_title("Silhouette Plot for K-Means Clustering")
        
        # The vertical line indicates the average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        
        plt.show()

    def grid_search_dbscan(self, param_grid):
        dbscan = DBSCAN()
        grid_search = GridSearchCV(estimator=dbscan, param_grid=param_grid, cv=5)
        grid_search.fit(self.X)
        return grid_search.best_estimator_, grid_search.best_params_

    def grid_search_agglomerative(self, param_grid):
        agglomerative = AgglomerativeClustering()
        grid_search = GridSearchCV(estimator=agglomerative, param_grid=param_grid, cv=5)
        grid_search.fit(self.X)
        return grid_search.best_estimator_, grid_search.best_params_

    def custom_silhouette_score(estimator, X):
        labels = estimator.labels_
        return silhouette_score(X, labels)

    def custom_davies_bouldin_score(estimator, X):
        labels = estimator.labels_
        return -davies_bouldin_score(X, labels)
    
    def custom_calinski_harabasz_score(estimator, X):
        labels = estimator.labels_
        return calinski_harabasz_score(X, labels)
    
