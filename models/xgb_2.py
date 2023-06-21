import statistics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import get_chunked_data_as_list
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Get the chunked data as a list of dataframes and labels
df_list, labels_list = get_chunked_data_as_list()

# Calculate the number of classes
num_classes = len(set(element for sublist in labels_list for element in sublist))

# Initialize a list to store accuracy rates
acc_rate = []

# Iterate over different max depths
for n in tqdm([20, 30, 50, 80]):
    # Initialize an XGBoost classifier with specified parameters
    xgb = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2)

    # Initialize a list to store accuracy values for cross-validation
    cv_acc = []

    # Iterate over each chunk of data and corresponding labels
    for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
        # Prepare training data by concatenating all other chunks except the current one
        X_train = pd.concat([df for j, df in enumerate(df_list) if i != j])
        y_train = [element for j, sublist in enumerate(labels_list) if i != j for element in sublist]

        # Encode labels using LabelEncoder
        encoder = LabelEncoder().fit(y_train)

        # Fit the XGBoost classifier on the training data
        model = xgb.fit(X_train, encoder.transform(y_train))

        # Predict labels for the test data
        y_pred = model.predict(X_test)

        # Calculate and store accuracy score
        cv_acc.append(accuracy_score(encoder.transform(y_test), y_pred))

    # Calculate the mean accuracy rate for the current max depth
    acc_rate.append(statistics.mean(cv_acc))

# Plot the accuracy rates for different max depths
plt.figure(figsize=(10, 6))
plt.plot([20, 30, 50, 80], acc_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Evolution of accuracy with respect to the max depth')
plt.xlabel('XGB max depth')
plt.ylabel('Accuracy')
plt.show()
