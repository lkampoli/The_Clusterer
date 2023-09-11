import pandas as pd
from src.clustering_algorithms import ClusteringAlgorithms
from src.data_preprocessing import DataPreprocessing
from src.postprocessing import Postprocessing

# Define parameter grids for grid search
kmeans_param_grid = {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10], 
                     'algorithm':{'lloyd', 'elkan', 'auto', 'full'},
                     'n_init': {'auto'},
                     'init': ['k-means++', 'random'],
                     'max_iter': [100, 200, 300]
                    }
#dbscan_param_grid = {'eps': [0.1, 0.2, 0.3, 0.4], 'min_samples': [2, 3, 4, 5]}
#agg_param_grid = {'n_clusters': [2, 3, 4, 5]}

# Load your dataset (adjust the file path as needed)
data = pd.read_csv('data/database.csv')
X = data.values

# Data preprocessing
data_preprocessor = DataPreprocessing(X)
X_std = data_preprocessor.standardize_data()

# Instantiate ClusteringAlgorithms class
clustering = ClusteringAlgorithms(X_std)

# Apply clustering algorithms
kmeans_clusters = clustering.apply_kmeans(n_clusters=4)
#dbscan_clusters = clustering.apply_dbscan(eps=0.5, min_samples=5)
#agg_clusters = clustering.apply_agglomerative(n_clusters=4)

# Instantiate Postprocessing class
postprocessor = Postprocessing(X_std)

# Visualize clusters
postprocessor.visualize_clusters(X_std, kmeans_clusters, 'K-Means Clustering')
#postprocessor.visualize_clusters(X_std, dbscan_clusters, 'DBSCAN Clustering')
#postprocessor.visualize_clusters(X_std, agg_clusters, 'Agglomerative Clustering')

# Calculate and print Silhouette Scores
kmeans_score = postprocessor.calculate_silhouette_score(kmeans_clusters)
#dbscan_score = postprocessor.calculate_silhouette_score(dbscan_clusters)
#agg_score = postprocessor.calculate_silhouette_score(agg_clusters)

print(f"K-Means Silhouette Score: {kmeans_score}")
#print(f"DBSCAN Silhouette Score: {dbscan_score}")
#print(f"Agglomerative Silhouette Score: {agg_score}")

# Plot Silhouette Plot for a specific clustering result (e.g., K-Means)
postprocessor.plot_silhouette(X_std, kmeans_clusters)

# Call select_k_with_silhouette to choose the number of clusters
best_num_clusters = clustering.select_k_with_silhouette(max_clusters=10)
print(f"The optimal number of clusters selected: {best_num_clusters}")
postprocessor.plot_silhouette_analysis(X_std, num_clusters=best_num_clusters)

# Compute and plot the percentage of variance explained
postprocessor.plot_variance_explained(X_std, max_clusters=10, fig_settings={'figsize': (8, 4)})
postprocessor.plot_wcss(X_std, max_clusters=10, fig_settings={'figsize': (8, 4)})
postprocessor.plot_wcss_and_variance_explained(X_std, max_clusters=10, fig_settings={'figsize': (8, 4)})

# Perform grid search for hyperparameter tuning (if needed)
# best_kmeans, kmeans_best_params = clustering.grid_search_kmeans(kmeans_param_grid)
# best_dbscan, dbscan_best_params = clustering.grid_search_dbscan(dbscan_param_grid)
# best_agg, agg_best_params = clustering.grid_search_agglomerative(agg_param_grid)

# Perform grid search for K-Means
best_kmeans, kmeans_best_params = clustering.grid_search_kmeans(kmeans_param_grid)
print("Best K-Means parameters:", kmeans_best_params)

# Perform grid search for DBSCAN
#best_dbscan, dbscan_best_params = clustering.grid_search_dbscan(dbscan_param_grid)
#print("Best DBSCAN parameters:", dbscan_best_params)

# Perform grid search for Agglomerative Clustering
#best_agg, agg_best_params = clustering.grid_search_agglomerative(agg_param_grid)
#print("Best Agglomerative Clustering parameters:", agg_best_params)

