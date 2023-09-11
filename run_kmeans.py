import os
import pandas as pd
from src.clustering_algorithms import ClusteringAlgorithms
from src.preprocessing import DataPreprocessing
from src.postprocessing import Postprocessing
#from src.ClusteringValidationIndices import ClusteringValidationIndices

# Create a 'results' directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Create an instance of the DataPreprocessing class without split
data_preprocessor  = DataPreprocessing('data/database.csv', perform_split=False, scaler_type='standard')

# Preprocess the data
coords_train, X_train_scaled, coords_test, X_test_scaled = data_preprocessor.preprocess_data()

# Instantiate ClusteringAlgorithms class
clustering = ClusteringAlgorithms(coords_train, X_train_scaled)

# Apply clustering algorithm
kmeans_clusters = clustering.apply_kmeans(n_clusters=4)

# Perform grid search for hyperparameter tuning for K-Means
# Define parameter grids for grid search
kmeans_param_grid = {'n_clusters':[2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'algorithm':['lloyd', 'elkan'],
                     'n_init':['auto'],
                     'init':['k-means++', 'random'],
                     'max_iter':[100, 200, 300]
                    }
#best_kmeans, kmeans_best_params = clustering.grid_search_kmeans(kmeans_param_grid)
#print("Best K-Means parameters:", kmeans_best_params)
#clustering.grid_search_kmeans2(kmeans_param_grid)

# Instantiate Postprocessing class
postprocessor = Postprocessing(X_train_scaled, coords_train)

# Visualize clusters
postprocessor.visualize_clusters(X_train_scaled, coords_train, kmeans_clusters, 'K-Means Clustering', fig_settings={'figsize': (30,15)})

postprocessor.visualize_elbow(X_train_scaled, coords_train, kmeans_clusters, 'K-Means Clustering', fig_settings={'figsize': (30,15)})
# Calculate and print Silhouette Scores
#kmeans_score = postprocessor.calculate_silhouette_score(kmeans_clusters)
#print(f" K-Means Silhouette Score: {kmeans_score} ")

# Plot Silhouette Plot for a specific clustering result (e.g., K-Means)
#postprocessor.plot_silhouette(X_train_scaled, kmeans_clusters, fig_settings={'figsize': (30,15)})
postprocessor.plot_gap_statistic(X_train_scaled, coords_train, fig_settings={'figsize': (30,15)})
postprocessor.plot_wcss(X_train_scaled, fig_settings={'figsize': (30,15)})
postprocessor.plot_wcss_and_variance_explained(X_train_scaled, fig_settings={'figsize': (30,15)})
postprocessor.plot_variance_explained(X_train_scaled, fig_settings={'figsize': (30,15)})

# Call select_k_with_silhouette to choose the number of clusters
##best_num_clusters = postprocessor.select_k_with_silhouette(max_clusters=10)
##print(f"The optimal number of clusters selected: {best_num_clusters}")
##postprocessor.plot_silhouette_analysis(X_std, num_clusters=best_num_clusters)
postprocessor.plot_kmeans_silhouette_analysis(X_train_scaled)
