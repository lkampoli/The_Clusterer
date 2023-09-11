from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class ClusteringValidationIndices:
    def __init__(self, data):
        """
        Initialize the ClusteringValidationIndices class.

        Args:
            data (numpy.ndarray or pandas.DataFrame): Input data for clustering validation.
        """
        self.data = data

    def silhouette_score(self, labels):
        """
        Calculate the Silhouette Score for clustering validation.

        Args:
            labels (numpy.ndarray): Cluster labels for the data.

        Returns:
            float: Silhouette Score.
        """
        return silhouette_score(self.data, labels)

    def davies_bouldin_score(self, labels):
        """
        Calculate the Davies-Bouldin Score for clustering validation.

        Args:
            labels (numpy.ndarray): Cluster labels for the data.

        Returns:
            float: Davies-Bouldin Score.
        """
        return davies_bouldin_score(self.data, labels)

    def calinski_harabasz_score(self, labels):
        """
        Calculate the Calinski-Harabasz Score for clustering validation.

        Args:
            labels (numpy.ndarray): Cluster labels for the data.

        Returns:
            float: Calinski-Harabasz Score.
        """
        return calinski_harabasz_score(self.data, labels)

