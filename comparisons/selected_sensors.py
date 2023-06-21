import statistics
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils import get_specific_chunked_data_as_list, plot_double_bar
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm

metrics = {}

# Initializing RandomForestClassifier with specific parameters
rf = RandomForestClassifier(n_estimators=40, max_depth=20, random_state=0, min_samples_leaf=2,
                            min_samples_split=2, bootstrap=False)

# Initializing XGBClassifier with specific parameters
xgb = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.2, min_child_weight=5, subsample=0.8)

# Looping through sensors: ['objects', 'location', 'body']
for sensor, inverse in tqdm(zip(['objects', 'location', 'body'], ['subject', 'without location', 'room']),
                            desc="sensor", position=0, total=3):

    # Retrieving specific chunked data as lists
    objects_df, rest_df, labels = get_specific_chunked_data_as_list(sensor)

    # Flattening the labels list
    y = [element for sublist in labels for element in sublist]

    # Printing the length of columns in objects_df[0]
    print("length is ", len(objects_df[0].columns))

    # Printing the length of columns in rest_df[0]
    print("length is ", len(rest_df[0].columns))

    # Selecting features from objects_df using RandomForestClassifier
    sfm = SelectFromModel(estimator=rf).fit(pd.concat([df for df in objects_df]), y)
    objects_df = [pd.DataFrame(sfm.transform(df)) for df in objects_df]

    # Selecting features from rest_df using RandomForestClassifier
    sfm = SelectFromModel(estimator=rf).fit(pd.concat([df for df in rest_df]), y)
    rest_df = [pd.DataFrame(sfm.transform(df)) for df in rest_df]

    # Printing the length of columns in objects_df[0] after feature selection
    print("length is ", len(objects_df[0].columns))

    # Printing the length of columns in rest_df[0] after feature selection
    print("length is ", len(rest_df[0].columns))

    # Initializing dictionaries to store metrics
    sensor_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}
    inverse_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}

    # Looping through data (objects_df, rest_df) and metrics (sens_met, inv_met)
    for data, met in tqdm(zip([objects_df, rest_df], [sens_met, inv_met]), desc="data", position=1, leave=False, total=2):
        for i, (X_test, y_test) in enumerate(zip(data, labels)):
            X_train = pd.concat([df for j, df in enumerate(data) if i != j])
            y_train = [element for j, sublist in enumerate(labels) if i != j for element in sublist]

            # Encoding the target labels
            encoder = LabelEncoder().fit(y_train)

            # Fitting RandomForestClassifier to the training data
            f_model = rf.fit(X_train, encoder.transform(y_train))

            # Predicting labels for the test data
            y_pred = f_model.predict(X_test)

            # Transforming the target labels of the test data
            y_test = encoder.transform(y_test)

            # Appending accuracy, F1 score, and Cohen's kappa score to respective lists
            met["accuracy"].append(accuracy_score(y_test, y_pred))
            met["f1"].append(f1_score(y_test, y_pred, average='micro'))
            met["kappa"].append(cohen_kappa_score(y_test, y_pred))

    # Calculating the mean of metrics for the sensor
    for k, v in sens_met.items():
        sensor_metrics[k] = statistics.mean(v)

    # Calculating the mean of metrics for the inverse sensor
    for k, v in inv_met.items():
        inverse_metrics[k] = statistics.mean(v)

    # Plotting the double bar graph with sensor-specific metrics
    plot_double_bar("Random Forest with feature selection on data from only one type of sensor", "Metrics", "Values",
                    ['Accuracy', 'F1', 'Kappa'], [v for k, v in sensor_metrics.items()],
                    [v for k, v in inverse_metrics.items()], sensor, inverse)
