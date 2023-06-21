import statistics
import numpy as np
from sktime.classification.deep_learning.cnn import CNNClassifier
from utils import get_sequenced_data

df_list, labels_list = get_sequenced_data()

acc = []
CNN = CNNClassifier(n_epochs=20)

# Loop over the data and labels
for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
    X_train = np.asarray(np.concatenate([df for j, df in enumerate(df_list) if i != j])).astype('float32')
    y_train = np.array([element for j, sublist in enumerate(labels_list) if i != j for element in sublist])

    X_test = np.asarray(X_test).astype('float32')
    y_test = np.array(y_test)

    # Create and fit the CNN model
    cnn = CNN.fit(X_train, y_train)

    # Evaluate the model on the test set and store the accuracy
    acc.append(cnn.score(X_test, y_test))

# Compute and print the average accuracy
print("Model average accuracy is: " + str(statistics.mean(acc)))
