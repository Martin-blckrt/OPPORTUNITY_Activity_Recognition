import statistics
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sktime.classification.kernel_based import RocketClassifier
from utils import get_sequenced_data
from tqdm import tqdm

# Get the sequenced data
df_list, labels_list = get_sequenced_data()

# Initialize a list to store accuracy values
acc = []

# Initialize a dictionary to store metric values
metrics = {"accuracy": [], "f1": [], "kappa": []}

# Initialize RocketClassifier with specified parameters
rocket = RocketClassifier(num_kernels=2000)

# Iterate over each sequence and corresponding labels
for i, (X_test, y_test) in enumerate(tqdm(zip(df_list, labels_list), total=len(df_list))):
    # Prepare training data by concatenating all other sequences except the current one
    X_train = np.asarray(np.concatenate([df for j, df in enumerate(df_list) if i != j])).astype('float32')
    y_train = np.array([element for j, sublist in enumerate(labels_list) if i != j for element in sublist])

    # Prepare test data
    X_test = np.asarray(X_test).astype('float32')
    y_test = np.array(y_test)

    # Fit the RocketClassifier on the training data
    net = rocket.fit(X_train, y_train)

    # Predict labels for the test data
    y_pred = net.predict(X_test)

    # Calculate and store accuracy, F1 score, and Cohen's kappa score
    metrics["accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["f1"].append(f1_score(y_test, y_pred, average='micro'))
    metrics["kappa"].append(cohen_kappa_score(y_test, y_pred))

# Print the classifier name and the average values of each metric
print("\n----- " + str(rocket) + " -----")
for k, v in metrics.items():
    print(k + " is : " + str(statistics.mean(v)))
