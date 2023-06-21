import statistics  # Importing the statistics module for computing mean values
import pandas as pd  # Importing the pandas module for data manipulation
from sklearn.feature_selection import SelectFromModel  # Importing the SelectFromModel class from sklearn.feature_selection
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # Importing specific classes from sklearn.preprocessing

from utils import get_chunked_drill_vs_adl_as_list, plot_double_bar  # Importing functions from a custom module
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier class from sklearn.ensemble
from xgboost import XGBClassifier  # Importing the XGBClassifier class from xgboost
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score  # Importing specific functions from sklearn.metrics
from tqdm import tqdm  # Importing the tqdm module for progress bar visualization

drill_df, drill_labels, adl_df, adl_labels = get_chunked_drill_vs_adl_as_list()  # Loading the drill and ADL data

drill_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}  # Initializing a dictionary to store the metrics for drill data
adl_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}  # Initializing a dictionary to store the metrics for ADL data

rf = RandomForestClassifier(n_estimators=40, max_depth=20, random_state=0, min_samples_leaf=2,
                            min_samples_split=2, bootstrap=False)  # Initializing a RandomForestClassifier
xgb20 = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2, min_child_weight=5,
                      subsample=0.8)  # Initializing an XGBClassifier with 20 estimators
xgb80 = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.2, min_child_weight=5,
                      subsample=0.8)  # Initializing an XGBClassifier with 80 estimators

y = [element for sublist in (drill_labels, adl_labels) for lab_list in sublist for element in
     lab_list]  # Combining the labels for drill and ADL data
encoder = LabelEncoder().fit(y)  # Encoding the combined labels using LabelEncoder

# Looping over the models and their names
for model, model_name in tqdm(zip([rf, xgb20, xgb80], ["Random Forest", "XGBoost (n=20)", "XGBoost (n=80)"])):
    sfm = SelectFromModel(estimator=model).fit(
        pd.concat([df for df_list in (drill_df, adl_df) for df in df_list]),
        encoder.transform(y))  # Fitting SelectFromModel on the concatenated drill and ADL data
    selected_drill_df = [pd.DataFrame(sfm.transform(df)) for df in
                         drill_df]  # Transforming the drill data with the selected features
    selected_adl_df = [pd.DataFrame(sfm.transform(df)) for df in adl_df]  # Transforming the ADL data with the selected features

    drill_cv_met = {"accuracy": [], "f1": [],
                    "kappa": []}  # Initializing a dictionary to store the metrics for drill data (per cross-validation fold)
    adl_cv_met = {"accuracy": [], "f1": [],
                  "kappa": []}  # Initializing a dictionary to store the metrics for ADL data (per cross-validation fold)

    # Looping over the data, labels, and indices for both drill and ADL data
    for i, (X_test_drill, y_test_drill, X_test_adl, y_test_adl) in enumerate(
            zip(selected_drill_df, drill_labels, selected_adl_df, adl_labels)):
        X_train_drill = pd.concat(
            [df for j, df in enumerate(selected_drill_df) if i != j])  # Combining all the drill data except for the current fold
        y_train_drill = [element for j, sublist in enumerate(drill_labels) if i != j for element in
                         sublist]  # Combining all the drill labels except for the current fold

        X_train_adl = pd.concat(
            [df for j, df in enumerate(selected_adl_df) if i != j])  # Combining all the ADL data except for the current fold
        y_train_adl = [element for j, sublist in enumerate(adl_labels) if i != j for element in
                       sublist]  # Combining all the ADL labels except for the current fold

        drill_model = model.fit(X_train_drill, encoder.transform(y_train_drill))  # Fitting the model on the drill data
        y_pred_drill = drill_model.predict(X_test_drill)  # Predicting the labels for the drill test data

        y_test_drill = encoder.transform(y_test_drill)  # Encoding the drill test labels
        drill_cv_met["accuracy"].append(
            accuracy_score(y_test_drill, y_pred_drill))  # Computing accuracy and appending to the metrics dictionary
        drill_cv_met["f1"].append(
            f1_score(y_test_drill, y_pred_drill, average='micro'))  # Computing F1 score and appending to the metrics dictionary
        drill_cv_met["kappa"].append(
            cohen_kappa_score(y_test_drill, y_pred_drill))  # Computing kappa score and appending to the metrics dictionary

        adl_model = model.fit(X_train_adl, encoder.transform(y_train_adl))  # Fitting the model on the ADL data
        y_pred_adl = adl_model.predict(X_test_adl)  # Predicting the labels for the ADL test data

        y_test_adl = encoder.transform(y_test_adl)  # Encoding the ADL test labels
        adl_cv_met["accuracy"].append(
            accuracy_score(y_test_adl, y_pred_adl))  # Computing accuracy and appending to the metrics dictionary
        adl_cv_met["f1"].append(
            f1_score(y_test_adl, y_pred_adl, average='micro'))  # Computing F1 score and appending to the metrics dictionary
        adl_cv_met["kappa"].append(
            cohen_kappa_score(y_test_adl, y_pred_adl))  # Computing kappa score and appending to the metrics dictionary

    for k, v in drill_cv_met.items():
        drill_metrics[k] = statistics.mean(
            v)  # Computing the mean value of each metric across all cross-validation folds for drill data
    for k, v in adl_cv_met.items():
        adl_metrics[k] = statistics.mean(
            v)  # Computing the mean value of each metric across all cross-validation folds for ADL data

    plot_double_bar("Performance of " + model_name + " on only drill and only ADL data", "Metrics", "Values",
                    ['Accuracy', 'F1', 'Kappa'], [v for k, v in drill_metrics.items()], [v for k, v in adl_metrics.items()],
                    'Drill', 'ADL') # Plotting the performance comparison between drill and ADL data for the current model
