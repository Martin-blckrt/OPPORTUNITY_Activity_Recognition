import statistics  # Importing the statistics module for computing mean values
import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for visualizations
import numpy as np  # Importing the numpy module for numerical operations
import pandas as pd  # Importing the pandas module for data manipulation
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # Importing specific classes from sklearn.preprocessing

from utils import get_chunked_drill_vs_adl_as_list  # Importing a function from a custom module
from sklearn.ensemble import RandomForestClassifier  # Importing the RandomForestClassifier class from sklearn.ensemble
from xgboost import XGBClassifier  # Importing the XGBClassifier class from xgboost
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score  # Importing specific functions from sklearn.metrics
from tqdm import tqdm  # Importing the tqdm module for progress bar visualization

# Calling the 'get_chunked_drill_vs_adl_as_list' function and assigning the returned values to variables
drill_df, drill_labels, adl_df, adl_labels = get_chunked_drill_vs_adl_as_list()

# Initializing dictionaries to store the metrics
drill_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}
adl_metrics = {"accuracy": 0, "f1": 0, "kappa": 0}

# Initializing three classification models with specific parameters
rf = RandomForestClassifier(n_estimators=40, max_depth=20, random_state=0, min_samples_leaf=2,
                            min_samples_split=2, bootstrap=False)
xgb20 = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2, min_child_weight=5, subsample=0.8)
xgb80 = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.2, min_child_weight=5, subsample=0.8)

# Looping over the models and their corresponding names
for model, model_name in tqdm(zip([rf, xgb20, xgb80], ["Random Forest", "XGBoost (n=20)", "XGBoost (n=80)"])):
    # Initializing dictionaries to store the metrics for each cross-validation fold
    drill_cv_met = {"accuracy": [], "f1": [], "kappa": []}
    adl_cv_met = {"accuracy": [], "f1": [], "kappa": []}

    # Looping over the indices and data in the drill and adl datasets
    for i, (X_test_drill, y_test_drill, X_test_adl, y_test_adl) in enumerate(zip(drill_df, drill_labels, adl_df, adl_labels)):
        # Creating the training set for the drill data
        X_train_drill = pd.concat([df for j, df in enumerate(drill_df) if i != j])
        y_train_drill = [element for j, sublist in enumerate(drill_labels) if i != j for element in sublist]

        # Encoding the drill labels using LabelEncoder
        encoder_drill = LabelEncoder().fit(y_train_drill)

        # Creating the training set for the adl data
        X_train_adl = pd.concat([df for j, df in enumerate(adl_df) if i != j])
        y_train_adl = [element for j, sublist in enumerate(adl_labels) if i != j for element in sublist]

        # Fitting the drill model with the training data
        drill_model = model.fit(X_train_drill, encoder_drill.transform(y_train_drill))
        # Predicting the labels for the drill test data
        y_pred_drill = drill_model.predict(X_test_drill)

        # Computing and storing accuracy, f1 score, and kappa for the drill model
        drill_cv_met["accuracy"].append(accuracy_score(encoder_drill.transform(y_test_drill), y_pred_drill))
        drill_cv_met["f1"].append(f1_score(encoder_drill.transform(y_test_drill), y_pred_drill, average='micro'))
        drill_cv_met["kappa"].append(cohen_kappa_score(encoder_drill.transform(y_test_drill), y_pred_drill))

        # Encoding the adl labels using LabelEncoder
        encoder_adl = LabelEncoder().fit(y_train_adl)

        # Fitting the adl model with the training data
        adl_model = model.fit(X_train_adl, encoder_adl.transform(y_train_adl))
        # Predicting the labels for the adl test data
        y_pred_adl = adl_model.predict(X_test_adl)

        # Computing and storing accuracy, f1 score, and kappa for the adl model
        adl_cv_met["accuracy"].append(accuracy_score(encoder_adl.transform(y_test_adl), y_pred_adl))
        adl_cv_met["f1"].append(f1_score(encoder_adl.transform(y_test_adl), y_pred_adl, average='micro'))
        adl_cv_met["kappa"].append(cohen_kappa_score(encoder_adl.transform(y_test_adl), y_pred_adl))

    # Computing the mean of the metrics across all cross-validation folds for drill and adl data
    for k, v in drill_cv_met.items():
        drill_metrics[k] = statistics.mean(v)
    for k, v in adl_cv_met.items():
        adl_metrics[k] = statistics.mean(v)

    # Creating bar plots to compare the metrics between drill and adl data
    X = ['Accuracy', 'F1', 'Kappa']
    X_axis = np.arange(len(X))
    drill_met = [v for k, v in drill_metrics.items()]
    adl_met = [v for k, v in adl_metrics.items()]

    plt.bar(X_axis - 0.2, drill_met, 0.4, label='Drill')  # Bar plot for drill metrics
    plt.bar(X_axis + 0.2, adl_met, 0.4, label='ADL')  # Bar plot for adl metrics

    plt.xticks(X_axis, X)  # Setting the x-axis labels
    plt.title("Performance of " + model_name + " on only drill and only ADL data")  # Setting the plot title
    plt.xlabel("Metrics")  # Setting the x-axis label
    plt.ylabel("Values")  # Setting the y-axis label
    plt.legend()  # Adding a legend
    plt.show()  # Displaying the plot
