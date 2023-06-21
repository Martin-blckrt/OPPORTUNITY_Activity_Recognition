import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils import get_chunked_data_as_list, leave_one_out_crossval
import statistics

# Get chunked data and labels as lists
df_list, labels_list = get_chunked_data_as_list()

# Initialize RandomForestClassifier and XGBClassifier
rf = RandomForestClassifier(n_estimators=40, max_depth=20, random_state=0, min_samples_leaf=2, min_samples_split=2,
                            bootstrap=False)
xgb = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2, min_child_weight=5, subsample=0.8)

# Encode labels
enc = LabelEncoder().fit([element for sublist in labels_list for element in sublist])

# Select features using SelectFromModel
sfm = SelectFromModel(estimator=xgb).fit(pd.concat([df for df in df_list]), enc.transform([element for sublist in labels_list for element in sublist]))
selected_df_list = [pd.DataFrame(sfm.transform(df)) for df in df_list]

# Perform leave-one-out cross-validation on all features
all_met = leave_one_out_crossval(xgb, df_list, labels_list, ["accuracy", "f1", "kappa"])

# Perform leave-one-out cross-validation on selected features
selected_met = leave_one_out_crossval(xgb, selected_df_list, labels_list, ["accuracy", "f1", "kappa"])

# Compute mean performance metrics for all and selected features
X = ['Accuracy', 'F1', 'Kappa']
X_axis = np.arange(len(X))
all_means = [statistics.mean(v) for k, v in all_met.items()]
selected_means = [statistics.mean(v) for k, v in selected_met.items()]

# Plot the performance comparison
plt.bar(X_axis - 0.2, all_means, 0.4, label='All')
plt.bar(X_axis + 0.2, selected_means, 0.4, label='Selected')

plt.xticks(X_axis, X)
plt.title("Performance of XGBoost using a certain amount of features")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.legend()
plt.show()
