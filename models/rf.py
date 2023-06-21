import statistics
import matplotlib.pyplot as plt
import pandas as pd
from utils import get_chunked_data_as_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

df_list, labels_list = get_chunked_data_as_list()

acc_rate = []

# Loop over different values of max_depth
for n in tqdm([10, 20, 30, 40, 50]):
    # Create a RandomForestClassifier with specified max_depth
    rf = RandomForestClassifier(max_depth=n, random_state=0)
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

    # Compute the mean accuracy for the current max_depth and append to acc_rate
    acc_rate.append(statistics.mean(cv_acc))

# Plot the accuracy rates against max_depth
plt.figure(figsize=(10, 6))
plt.plot([10, 20, 30, 40, 50], acc_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Evolution of accuracy with respect to depth')
plt.xlabel('Forest depth')
plt.ylabel('Accuracy')
plt.show()
