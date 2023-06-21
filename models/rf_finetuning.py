import statistics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import get_chunked_data_as_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

df_list, labels_list = get_chunked_data_as_list()

big_df = pd.concat([df for df in df_list], axis=0)

acc_rate = 0
scaler = MinMaxScaler()
scaler.fit(big_df)
df_list = [pd.DataFrame(scaler.transform(df), columns=df.columns) for df in df_list]

# Nested loops to iterate over different hyperparameter values
for max_ft in tqdm(['auto', 'sqrt']):
    for min_leaf in tqdm([1, 2, 4]):
        for min_split in tqdm([2, 5, 10]):
            for bs in tqdm([True, False]):
                # Create a RandomForestClassifier with specified hyperparameters
                rf = RandomForestClassifier(max_depth=20, max_features=max_ft, min_samples_leaf=min_leaf,
                                            min_samples_split=min_split, bootstrap=bs, random_state=0)
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

                # Check if the current configuration has a higher accuracy than the previous best configuration
                if statistics.mean(cv_acc) > acc_rate:
                    # Update the best configuration and accuracy
                    acc_rate = statistics.mean(cv_acc)
                    best_config = {
                        "max_features": max_ft,
                        "min_samples_leaf": min_leaf,
                        "min_samples_split": min_split,
                        "bootstrap": bs
                    }

# Print the best configuration and its accuracy
print("Best configuration is:")
print(best_config)
print("Accuracy:", acc_rate)
