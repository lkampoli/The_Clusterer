import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

from gap_statistic import OptimalK

class Postprocessing:

    def __init__(self, X, XYZ):
        """
        Initialize the Postprocessing class.

        Args:
            X (numpy.ndarray): The input data for postprocessing.
        """
        self.X = X
        self.XYZ = XYZ

    def calculate_silhouette_score(self, y_pred):
        """
        Calculate the silhouette score for a given clustering result.

        Args:
            y_pred (numpy.ndarray): Cluster labels for data points.

        Returns:
            float: Silhouette score.
        """
        print("")
        print(" [Info]: Calculate Silhouette score ")
        return silhouette_score(self.X, y_pred)

    def calculate_davies_bouldin(self, y_pred):
        """
        Calculate the Davies-Bouldin score for a given clustering result.

        Args:
            y_pred (numpy.ndarray): Cluster labels for data points.

        Returns:
            float: Davies-Bouldin score.
        """
        print("")
        print(" [Info]: Calculate Davies-Bouldin score ")
        return davies_bouldin_score(self.X, y_pred)

    def calculate_calinski_harabasz(self, y_pred):
        """
        Calculate the Calinski-Harabasz score for a given clustering result.

        Args:
            y_pred (numpy.ndarray): Cluster labels for data points.

        Returns:
            float: Calinski-Harabasz score.
        """
        print("")
        print(" [Info]: Calculate Calinski-Harabasz score ")
        return calinski_harabasz_score(self.X, y_pred)

    def visualize_clusters(self, X, coords, cluster_labels, title, save_eps=True, fig_settings=None):
        """
        Visualize clusters using a scatter plot.

        Args:
            X (numpy.ndarray): Input data.
            coords (numpy.ndarray): physical coordinates of the computational domain.
            cluster_labels (numpy.ndarray): Cluster labels.
            title (str): Title for the plot.
            save_eps (bool): Whether to save the figure in EPS format (default is False).
            fig_settings (dict): Dictionary of figure settings (optional).

        Returns:
            None
        """
        print("")
        print(" [Info]: Visualize clusters ")
        if fig_settings is None:
            fig_settings = {} # Default settings if not provided

        plt.figure(**fig_settings) # Use provided figure settings
        unique_labels = np.unique(cluster_labels)

        plt.scatter(self.XYZ['Cx'], self.XYZ['Cy'], c=cluster_labels, cmap='viridis')
        plt.title(title, fontsize=25)
        #if save_eps:
        #    plt.savefig(f'{title}.eps', format='eps', bbox_inches='tight') # Save as EPS format
        plt.xlim(0,9)
        plt.ylim(0,3)
        plt.xlabel('x/h', fontsize=25)
        plt.ylabel('y/h', fontsize=25)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.tight_layout()
        plt.savefig(f'./results/{title}.eps', format='eps', bbox_inches='tight')
        plt.show()
        plt.close()

#        for label in unique_labels:
#            cluster_points = X[cluster_labels == label]
#            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
#        plt.title(title)
#        plt.xlabel('Feature 1')
#        plt.ylabel('Feature 2')
#        plt.legend()
#        if save_eps:
#            plt.savefig(f'{title}_feature_space.eps', format='eps', bbox_inches='tight')
#        plt.show()
#        plt.close()

    def plot_silhouette(self, X, cluster_labels, fig_settings=None):
        """
        Plot the Silhouette Plot for clustering results.

        Args:
            X (numpy.ndarray): Input data.
            cluster_labels (numpy.ndarray): Cluster labels.

        Returns:
            None
        """
        print("")
        print(" [Info]: Plot Silhouette ")
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        print(" [Info]: Visualize clusters ")
        if fig_settings is None:
            fig_settings = {} # Default settings if not provided

        fig, ax = plt.subplots(**fig_settings)
        y_lower = 10
        for i in range(len(set(cluster_labels))):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / len(set(cluster_labels)))
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        ax.xaxis.label.set_size(25)
        ax.yaxis.label.set_size(25)
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticks([]) # Clear the y-axis labels
        ax.set_title("Silhouette Plot")
        plt.tight_layout()
        plt.savefig("./results/Silhouette.eps")
        plt.show()
        plt.close()

    # https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29
    # https://stats.stackexchange.com/questions/95290/how-should-i-interpret-gap-statistic
    def plot_gap_statistic(self, X, coords, max_clusters=10, n_bootstrap=5, fig_settings=None):
        """
        Plot the Gap Statistic for a range of cluster numbers.

        Args:
            X (numpy.ndarray): Input data.
            max_clusters (int): The maximum number of clusters to consider.
            n_bootstrap (int): The number of bootstrap samples for the reference distribution.

        Returns:
            None
        """
        print("")
        print(" [Info]: Plot Gap statistic ")
        gap_values = []  # List to store Gap Statistic values
        std_devs = []    # List to store standard deviations

        for k in range(1, max_clusters + 1):
            # Fit K-Means for the current k
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)

            # Compute within-cluster dispersion (Wk)
            cluster_dispersion = []
            for cluster_id in range(k):
                cluster_points = X[kmeans.labels_ == cluster_id]
                centroid = kmeans.cluster_centers_[cluster_id]
                cluster_dispersion.append(np.sum((cluster_points - centroid) ** 2))

            Wk = np.sum(cluster_dispersion)

            # Generate random data for the reference distribution
            Bk = []
            for _ in range(n_bootstrap):
                random_data = np.random.rand(X.shape[0], X.shape[1])
                random_kmeans = KMeans(n_clusters=k, random_state=42)
                random_kmeans.fit(random_data)
                random_dispersion = []
                for cluster_id in range(k):
                    cluster_points = random_data[random_kmeans.labels_ == cluster_id]
                    centroid = random_kmeans.cluster_centers_[cluster_id]
                    random_dispersion.append(np.sum((cluster_points - centroid) ** 2))
                Bk.append(np.sum(random_dispersion))

            Bk = np.array(Bk)
            gap = np.mean(np.log(Bk)) - np.log(Wk)
            std_dev = np.std(np.log(Bk)) * np.sqrt(1 + 1 / n_bootstrap)
            gap_values.append(gap)
            std_devs.append(std_dev)

        # Plot Gap Statistic
        if fig_settings is None:
            fig_settings = {} # Default settings if not provided

        plt.figure(**fig_settings)
        plt.plot(range(1, max_clusters + 1), gap_values, marker='o', color='b', label='Gap Statistic')
        plt.errorbar(range(1, max_clusters + 1), gap_values, yerr=std_devs, fmt='-o', color='b', alpha=0.7, label='Gap Statistic ± 1 Std. Dev.')
        plt.xlabel('Number of Clusters (k)', fontsize=25)
        plt.ylabel('Gap Statistic', fontsize=25)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.title('Gap Statistic vs. Number of Clusters', fontsize=25)
        plt.legend(fontsize=25)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./results/Gap.eps")
        plt.show()
        plt.close()

#        #########################################################################       
#        optimalK = OptimalK(parallel_backend='rust')
#        print(optimalK)
#        n_clusters = optimalK(X, cluster_array=np.arange(1, 15))
#        print('Optimal clusters: ', n_clusters)
#        print(optimalK.gap_df.head())
#        plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
#        plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
#                    optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
#        plt.grid(True)
#        plt.xlabel('Cluster Count')
#        plt.ylabel('Gap Value')
#        plt.title('Gap Values by Cluster Count')
#        plt.show()
#
#        # Now that we have the optimal clusters, n, we build our own KMeans model...
#        km = KMeans(n_clusters)
#        km.fit(X)
#        
#        df = pd.DataFrame(coords, columns=['Cx','Cy'])
#        df['label'] = km.labels_
#        
#        colors = plt.cm.Spectral(np.linspace(0, 1, len(df.label.unique())))
#        
#        for color, label in zip(colors, df.label.unique()):
#        
#            tempdf = df[df.label == label]
#            plt.scatter(tempdf.x, tempdf.y, c=color)
#        
#        plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], c='r', s=500, alpha=0.7, )
#        plt.grid(True)
#        plt.show()
#        #########################################################################       
    
    def plot_wcss(self, X, max_clusters=10, fig_settings=None):
        """
        Compute and plot the Within-Cluster Sum of Squares (WCSS) for a range of cluster numbers.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            max_clusters (int): The maximum number of clusters to consider.
            fig_settings (dict): Dictionary of figure settings (optional).

        Returns:
            None
        """
        print("")
        print(" [Info]: Plot WCSS")
        if fig_settings is None:
            fig_settings = {}  # Default settings if not provided

        # Convert X to NumPy array if it's a DataFrame
        #if isinstance(X, pd.DataFrame):
        #    X = X.values

        wcss = []  # List to store WCSS values

        for k in range(1, max_clusters + 1):
            # Fit K-Means for the current k
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)

            # Compute WCSS and append to the list
            wcss.append(kmeans.inertia_)

        # Plot WCSS
        plt.figure(**fig_settings)  # Use provided figure settings
        plt.plot(range(1, max_clusters + 1), wcss, marker='o', color='b')
        plt.title('Within-Cluster Sum of Squares (WCSS) vs. Number of Clusters', fontsize=25)
        plt.xlabel('Number of Clusters (k)', fontsize=25)
        plt.ylabel('WCSS', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=25)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./results/WCSS.eps")
        plt.show()
        plt.close()

    def plot_wcss_and_variance_explained(self, X, max_clusters=10, fig_settings=None):
        """
        Compute and plot WCSS and the percentage of variance explained for a range of cluster numbers.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            max_clusters (int): The maximum number of clusters to consider.
            fig_settings (dict): Dictionary of figure settings (optional).

        Returns:
            None
        """
        print("")
        print(" [Info]: Plot WCSS and % explained variance")
        if fig_settings is None:
            fig_settings = {}  # Default settings if not provided

        # Convert X to NumPy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        wcss = []  # List to store WCSS values
        variance_explained = []  # List to store percentage of variance explained

        for k in range(1, max_clusters + 1):
            # Fit K-Means for the current k
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)

            # Compute WCSS and append to the list
            wcss.append(kmeans.inertia_)

            # Compute the percentage of variance explained
            total_variance = np.sum((X - X.mean(axis=0)) ** 2)
            explained_variance = total_variance - kmeans.inertia_
            percent_explained = (explained_variance / total_variance) * 100
            variance_explained.append(percent_explained)

        # Plot WCSS and percentage of variance explained
        plt.figure(**fig_settings)  # Use provided figure settings

        # Plot WCSS
        plt.subplot(2, 1, 1)
        plt.plot(range(1, max_clusters + 1), wcss, marker='o', color='b')
        plt.title('Within-Cluster Sum of Squares (WCSS) vs. Number of Clusters', fontsize=25)
        plt.xlabel('Number of Clusters (k)', fontsize=25)
        plt.ylabel('WCSS', fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.grid(True)

        # Plot percentage of variance explained
        plt.subplot(2, 1, 2)
        plt.plot(range(1, max_clusters + 1), variance_explained, marker='o', color='r')
        plt.title('Percentage of Variance Explained vs. Number of Clusters', fontsize=25)
        plt.xlabel('Number of Clusters (k)', fontsize=25)
        plt.ylabel('Percentage of Variance Explained (%)', fontsize=25)
        plt.grid(True)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.savefig("./results/WCSS_perc_exp_var.eps")
        plt.show()
        plt.close()

    def plot_variance_explained(self, X, max_clusters=10, fig_settings=None):
        """
        Compute and plot the percentage of variance explained for a range of cluster numbers.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            max_clusters (int): The maximum number of clusters to consider.
            fig_settings (dict): Dictionary of figure settings (optional).

        Returns:
            None
        """
        print("")
        print(" [Info]: Plot % explained variance")
        if fig_settings is None:
            fig_settings = {}  # Default settings if not provided

        variance_explained = []  # List to store percentage of variance explained

        for k in range(1, max_clusters + 1):
            # Fit K-Means for the current k
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)

            # Compute the percentage of variance explained
            total_variance = np.sum((X - X.mean(axis=0)) ** 2)
            explained_variance = total_variance - kmeans.inertia_
            percent_explained = (explained_variance / total_variance) * 100
            variance_explained.append(percent_explained)

        # Plot percentage of variance explained
        plt.figure(**fig_settings)  # Use provided figure settings
        plt.plot(range(1, max_clusters + 1), variance_explained, marker='o', color='r')
        plt.title('Percentage of Variance Explained vs. Number of Clusters', fontsize=25)
        plt.xlabel('Number of Clusters (k)', fontsize=25)
        plt.ylabel('Percentage of Variance Explained (%)', fontsize=25)
        plt.grid(True)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.savefig("./results/perc_exp_var.eps")
        plt.show()
        plt.close()

    def select_k_with_silhouette(X, max_clusters=10):
        """
        Select the number of clusters using silhouette analysis with K-Means clustering.
    
        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            max_clusters (int): The maximum number of clusters to consider.
    
        Returns:
            int: The optimal number of clusters selected based on silhouette analysis.
        """
        # Convert X to NumPy array if it's a DataFrame
        #if isinstance(X, pd.DataFrame):
        #    X = X.values
    
        best_num_clusters = 2  # Initialize with a minimum of 2 clusters
        best_silhouette_score = -1  # Initialize with a low value
    
        for num_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
    
            silhouette_avg = silhouette_score(X, cluster_labels)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_num_clusters = num_clusters

        return best_num_clusters

    def plot_silhouette_analysis(X, coords, num_clusters):
        """
        Plot silhouette analysis for K-Means clustering with the selected number of clusters.
    
        Args:
            X (numpy.ndarray or pandas.DataFrame): Input data.
            num_clusters (int): The number of clusters to use in K-Means.
    
        Returns:
            None
        """
        print("")
        print(" [Info]: Plot Silhouette Analysis ")
    
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
    
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        plt.figure(figsize=(8, 6))
        plt.title(f"Silhouette Analysis for {num_clusters} Clusters")
        plt.xlabel("Silhouette Coefficient Values")
        plt.ylabel("Cluster Label")
    
        y_lower = 10
        for i in range(num_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = plt.cm.nipy_spectral(float(i) / num_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for the next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.xlim([-0.1, 1])
        plt.ylim([0, len(X) + (num_clusters + 1) * 10])
        plt.show()

    def plot_kmeans_silhouette_analysis(self, X):
        print("")
        print(" [Info]: Plot Silhouette Analysis ")
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Convert X to NumPy array if it's a DataFrame
        #if isinstance(X, pd.DataFrame):
        #    X = X.values

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
        
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
            cluster_labels = clusterer.fit_predict(X)
        
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )
        
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        
                ith_cluster_silhouette_values.sort()
        
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
        
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
        
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
        
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
        
            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )
        
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )
        
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
        
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
        
            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )
        
        plt.tight_layout()
        plt.savefig("./results/Silhouette_analysis_cluster_"+str(n_clusters)+".eps")
        plt.show()
        plt.close()

    def visualize_elbow(self, X, coords, cluster_labels, title, save_eps=True, fig_settings=None):
        print("")
        print(" [Info]: Plot Elbow ")
        cs = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(X)
            cs.append(kmeans.inertia_)
        plt.plot(range(1, 11), cs)
        plt.title('The Elbow Method', fontsize=25)
        plt.xlabel('Number of clusters', fontsize=25)
        plt.ylabel('CS', fontsize=25)
        plt.grid()
        plt.tight_layout()
        plt.savefig("./results/elbow.eps")
        plt.show()
        plt.close()

#        # plot FCM result
#        f, axes = plt.subplots(1, 2, figsize=(11,5))
#        axes[0].scatter(X[:,0], X[:,1], alpha=.1)
#        axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
#        axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
#        plt.savefig('images/basic-clustering-output.jpg')
#        plt.show()
