import statistics  # Importing the statistics module for computing mean values
import pandas as pd  # Importing the pandas module for data manipulation
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # Importing specific classes from sklearn.preprocessing

from utils import get_specific_chunked_data_as_list, plot_double_bar  # Importing functions from a custom module
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier class from sklearn.ensemble
from xgboost import XGBClassifier  # Importing the XGBClassifier class from xgboost
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score  # Importing specific functions from sklearn.metrics
from tqdm import tqdm  # Importing the tqdm module for progress bar visualization

metrics = {}  # Initializing a dictionary to store the metrics

rf = RandomForestClassifier(n_estimators=40, max_depth=20, random_state=0, min_samples_leaf=2,
                            min_samples_split=2, bootstrap=False)  # Initializing a RandomForestClassifier
xgb = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.2, min_child_weight=5, subsample=0.8)  # Initializing an XGBClassifier

# Looping over the sensors and their corresponding inverses
for sensor, inverse in tqdm(zip(['objects', 'location', 'body'], ['subject', 'without location', 'room']), desc="sensor", position=0, total=3):
    objects_df, rest_df, labels = get_specific_chunked_data_as_list(sensor)  # Calling a function to get specific data based on the sensor

    sensor_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}  # Initializing a dictionary to store the metrics for the sensor data
    inverse_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}  # Initializing a dictionary to store the metrics for the inverse data

    sens_met = {"accuracy": [], "f1": [], "kappa": []}  # Initializing a dictionary to store the metrics for the sensor data (per cross-validation fold)
    inv_met = {"accuracy": [], "f1": [], "kappa": []}  # Initializing a dictionary to store the metrics for the inverse data (per cross-validation fold)

    # Looping over the sensor data and inverse data
    for data, met in tqdm(zip([objects_df, rest_df], [sens_met, inv_met]), desc="data", position=1, leave=False, total=2):
        # Looping over the indices and data in the current fold
        for i, (X_test, y_test) in enumerate(zip(data, labels)):
            X_train = pd.concat([df for j, df in enumerate(data) if i != j])
            y_train = [element for j, sublist in enumerate(labels) if i != j for element in sublist]

            encoder = LabelEncoder().fit(y_train)  # Encoding the labels using LabelEncoder

            f_model = rf.fit(X_train, encoder.transform(y_train))  # Fitting the random forest model with the training data
            y_pred = f_model.predict(X_test)  # Predicting the labels for the test data
            y_test = encoder.transform(y_test)  # Encoding the true test labels

            # Computing and storing accuracy, f1 score, and kappa for the current fold
            met["accuracy"].append(accuracy_score(y_test, y_pred))
            met["f1"].append(f1_score(y_test, y_pred, average='micro'))
            met["kappa"].append(cohen_kappa_score(y_test, y_pred))

    # Computing the mean of the metrics across all cross-validation folds for the sensor and inverse data
    for k, v in sens_met.items():
        sensor_metrics[k] = statistics.mean(v)
    for k, v in inv_met.items():
        inverse_metrics[k] = statistics.mean(v)

    # Creating a double bar plot to compare the metrics between the sensor and inverse data
    plot_double_bar("XGBoost on data from only one type of sensor", "Metrics", "Values",
                    ['Accuracy', 'F1', 'Kappa'], [v for k, v in sensor_metrics.items()],
                    [v for k, v in inverse_metrics.items()], sensor, inverse)
