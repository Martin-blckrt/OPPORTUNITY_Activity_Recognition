import statistics
from utils import get_chunked_data_as_list
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import trange

df_list, labels_list = get_chunked_data_as_list()

error_rate = []

# Loop over different values of K
for n in trange(1, 30):
    knn = KNeighborsClassifier(n_neighbors=n)
    cv_acc = []

    # Cross-validation loop
    for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
        X_train = pd.concat([df for j, df in enumerate(df_list) if i != j])
        y_train = [element for j, sublist in enumerate(labels_list) if i != j for element in sublist]

        # Fit the K Nearest Neighbors classifier
        model = knn.fit(X_train, y_train)

        # Predict labels for the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy and collect results
        cv_acc.append(accuracy_score(y_test, y_pred))

    # Compute error rate for the current K value
    error_rate.append(1 - statistics.mean(cv_acc))

# Plot the error rate evolution
plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Evolution of Error rate with respect to K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
