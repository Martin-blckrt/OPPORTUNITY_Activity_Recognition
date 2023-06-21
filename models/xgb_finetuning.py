import statistics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from utils import get_chunked_data_as_list
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Get the chunked data as a list of dataframes and labels
df_list, labels_list = get_chunked_data_as_list()

# Concatenate all dataframes into a single big dataframe
big_df = pd.concat([df for df in df_list], axis=0)

# Initialize the accuracy rate variable
acc_rate = 0

# Scale the feature values using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(big_df)
df_list = [pd.DataFrame(scaler.transform(df), columns=df.columns) for df in df_list]

# Iterate over different values for min_child_weight and subsample
for min_child_weight in tqdm(range(5, 8)):
    for subsample in [i / 10. for i in tqdm(range(8, 11))]:
        # Initialize an XGBoost classifier with specified parameters
        xgb = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.2, min_child_weight=min_child_weight,
                            subsample=subsample)

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

        # Check if the current configuration has a higher accuracy rate
        if statistics.mean(cv_acc) > acc_rate:
            # Update the highest accuracy rate and print the best configuration
            acc_rate = statistics.mean(cv_acc)
            print("Best config is : ")
            print("min_child_weight : ", min_child_weight)
            print("subsample : ", subsample)

# Print the overall accuracy rate
print("accuracy of ", acc_rate)
