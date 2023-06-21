from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils import get_chunked_data_as_list

# Get chunked data and labels as lists
df_list, labels_list = get_chunked_data_as_list()

# Concatenate dataframes
X = pd.concat([df for df in df_list])

# Flatten labels
y = [element for sublist in labels_list for element in sublist]

# Encode labels
enc = LabelEncoder().fit(y)

# Initialize Random Forest and XGBoost classifiers
rf = RandomForestClassifier(n_estimators=40, max_depth=20, random_state=0, min_samples_leaf=2, min_samples_split=2,
                            bootstrap=False)
xgb = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2, min_child_weight=5, subsample=0.8)

# Fit classifiers on the data
rf = rf.fit(X, enc.transform(y))
xgb = xgb.fit(X, enc.transform(y))

# Get feature importances
rf_imp = rf.feature_importances_
xgb_imp = xgb.feature_importances_

# Sort feature importances in descending order
rf_imp = pd.Series(rf_imp, index=df_list[0].columns).sort_values(ascending=False)
xgb_imp = pd.Series(xgb_imp, index=df_list[0].columns).sort_values(ascending=False)

# Select top 100 features from each classifier
top_rf_ft = rf_imp[:100]
top_xgb_ft = xgb_imp[:100]

# Count the occurrence of each feature across the top features of both classifiers
counts = Counter(list(top_rf_ft.keys()) + list(top_xgb_ft.keys()))

# Print the number of identical features
print("There are", sum(value == 2 for value in counts.values()), "identical features")
