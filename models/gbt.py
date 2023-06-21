import statistics
import matplotlib.pyplot as plt
import pandas as pd
from utils import get_chunked_data_as_list
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

df_list, labels_list = get_chunked_data_as_list()

acc_rate = []

# Loop over different numbers of estimators
for n in tqdm([10, 20, 30]):
    gbt = GradientBoostingClassifier(n_estimators=n, learning_rate=1.0, max_depth=1, random_state=0)
    cv_acc = []

    # Cross-validation loop
    for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
        X_train = pd.concat([df for j, df in enumerate(df_list) if i != j])
        y_train = [element for j, sublist in enumerate(labels_list) if i != j for element in sublist]

        # Fit the Gradient Boosting Classifier
        model = gbt.fit(X_train, y_train)

        # Predict labels for the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy and collect results
        cv_acc.append(accuracy_score(y_test, y_pred))

    # Compute mean accuracy for the current number of estimators
    acc_rate.append(statistics.mean(cv_acc))

# Plot the accuracy evolution
plt.figure(figsize=(10, 6))
plt.plot([10, 20, 30], acc_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Evolution of accuracy with respect to number of estimators')
plt.xlabel('Number of GBT estimators')
plt.ylabel('Accuracy')
plt.show()
