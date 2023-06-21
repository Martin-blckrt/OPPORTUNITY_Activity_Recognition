import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder
from utils import get_chunked_data_as_list

plt.rcParams["figure.figsize"] = (20, 25)

# Get chunked data and labels as lists
df_list, labels_list = get_chunked_data_as_list()

# Concatenate dataframes
X = pd.concat([df for j, df in enumerate(df_list)])

# Flatten labels
y = [element for sublist in labels_list for element in sublist]

# Encode labels
encoder = LabelEncoder().fit(y)
y = encoder.transform(y)

# Fit Ridge regression model
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)

# Get feature importances (absolute coefficients)
importance = np.abs(ridge.coef_)

# Get feature names
feature_names = np.array(X.columns)

# Plot feature importances
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

# Set threshold values for low and high importances
LOW_THRESHOLD = 0.01
HIGH_THRESHOLD = 4

# Initialize lists for low and high importances and their corresponding feature names
low_imp, low_col = [], []
high_imp, high_col = [], []

# Iterate over importances and feature names
for imp, col in zip(importance, feature_names):
    if imp < LOW_THRESHOLD:
        low_imp.append(imp)
        low_col.append(col)
    elif imp > HIGH_THRESHOLD:
        high_imp.append(imp)
        high_col.append(col)

# Plot lowest feature importances
plt.bar(height=low_imp, x=low_col)
plt.xticks(rotation='vertical')
plt.title("Lowest feature importances via coefficients")
plt.show()

# Plot highest feature importances
plt.bar(height=high_imp, x=high_col)
plt.xticks(rotation='vertical')
plt.title("Highest feature importances via coefficients")
plt.show()
