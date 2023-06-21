import statistics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import get_chunked_data_as_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

df_list, labels_list = get_chunked_data_as_list()

# Concatenate all dataframes into a single big_df
big_df = pd.concat([df for df in df_list], axis=0)

acc_rate = []

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(big_df)
df_list = [pd.DataFrame(scaler.transform(df), columns=df.columns) for df in df_list]

# Loop over different values of n_estimators
for n in tqdm([10, 20, 40, 60, 80]):
    # Create a RandomForestClassifier with specified parameters
    rf = RandomForestClassifier(n_estimators=n, max_depth=20, random_state=0, min_samples_leaf=2,
                                min_samples_split=2, bootstrap=False)
    cv_acc = []
    # Loop over the data and labels
    for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
        X_train = pd.concat([df for j, df in enumerate(df_list) if i != j])
        y_train = [element for j, sublist in enumerate(labels_list) if i != j for element in sublist]

        # Fit the Random Forest model
        model = rf.fit(X_train, y_train)
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        # Compute accuracy and append to cv_acc
        cv_acc.append(accuracy_score(y_test, y_pred))

    # Compute the mean accuracy for the current n_estimators value and append to acc_rate
    acc_rate.append(statistics.mean(cv_acc))

# Plot the accuracy rates against the number of estimators
plt.figure(figsize=(10, 6))
plt.plot([10, 20, 40, 60, 80], acc_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Evolution of accuracy with respect to the number of estimators')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()
