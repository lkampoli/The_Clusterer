import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

class DataPreprocessing:

    def __init__(self, dataset_path, test_size=0.2, random_state=42, perform_split=True, scaler_type='standard'):
        """
        Initialize the DataPreprocessing class.

        Args:
            dataset_path (str): Path to the dataset file in CSV format.
            test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
            random_state (int): Random seed for reproducibility (default is 42).
            perform_split (bool): Whether to perform train-test split (default is True).
            scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust'; default is 'standard').
        """
        self.dataset_path  = dataset_path
        self.test_size     = test_size
        self.random_state  = random_state
        self.perform_split = perform_split
        self.scaler_type   = scaler_type

    def preprocess_data(self):
        """
        Load the dataset, optionally apply train-test split, scale features, and return coordinates and features.

        Returns:
            tuple: A tuple containing coordinates and scaled features arrays.
        """
        # Load the dataset
        data = pd.read_csv(self.dataset_path)
        print("")
        print(" [Info]: full dataset ")
        print(data)

        # Separate coordinates and features
        coordinates = data[['Cx', 'Cy', 'Cz']]
        features    = data.drop(['Cx','Cy','Cz','abs_DELTA_U','DELTA_U','abs_DELTA_V','DELTA_V','abs_DELTA_Rxx','abs_DELTA_Rxy','abs_DELTA_Ryy','abs_DELTA_K','DELTA_K'], axis=1)

        if self.perform_split:
            # Split the data into training and testing sets
            X_train, X_test, coords_train, coords_test = train_test_split(
                features, coordinates, test_size=self.test_size, random_state=self.random_state
            )
        else:
            # If not performing split, use the entire dataset for training
            X_train      = features
            X_test       = None
            coords_train = coordinates
            coords_test  = None

        print("")
        print(" [Info]: train dataset after train_test_split ")
        print(X_train)

        # Choose the scaler based on the specified scaler_type
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaler_type. Use 'standard', 'minmax', or 'robust'.")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        print("")
        print(f" [Info]: train datset after '{self.scaler_type}' scaling ")
        print(X_train_scaled)

        if self.perform_split:
            # Scale the test set features if the split was performed
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = None
        print("")
        print(f" [Info]: test datset after '{self.scaler_type}' scaling ",X_test_scaled)

        return coords_train, X_train_scaled, coords_test, X_test_scaled

