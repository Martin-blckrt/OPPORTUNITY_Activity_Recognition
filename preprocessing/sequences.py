import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../treated_data/encoded_data.csv')
# Read the CSV file into a DataFrame named 'df'

df = df.drop(df.columns[0], axis=1)
# Drop the first column from the 'df' DataFrame

X_train = df[df['File'] == 'S1-Drill.dat']
# Select rows from 'df' where the 'File' column is 'S1-Drill.dat' as training data

X_test = df[df['File'] == 'S4-Drill.dat']
# Select rows from 'df' where the 'File' column is 'S4-Drill.dat' as test data

num_classes = len(df['ML_Both_Arms'].unique())
# Count the number of unique classes in the 'ML_Both_Arms' column

X_train = X_train.drop(['File'], axis=1)
# Drop the 'File' column from the training data DataFrame

X_test = X_test.drop(['File'], axis=1)
# Drop the 'File' column from the test data DataFrame

cols_to_scale = [i for i in X_train.columns if i not in ['MILLISEC', 'ML_Both_Arms']]
# Create a list of column names to scale, excluding 'MILLISEC' and 'ML_Both_Arms'

scaler = MinMaxScaler(feature_range=(-1, 1))
# Create a MinMaxScaler object with a feature range of (-1, 1)

scaler = scaler.fit(X_train[cols_to_scale])
# Fit the scaler to the training data columns to compute the scaling parameters

X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
# Apply the scaler transformation to the training data columns

X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
# Apply the same scaler transformation to the test data columns

def create_sequences(input_data: pd.DataFrame, target_column, sequence_length):
    # Function to create sequences for time series data

    sequences = []
    data_size = len(input_data)

    for i in range(data_size - sequence_length):
        sequence = input_data[i:i + sequence_length]
        label_position = i + sequence_length
        label = input_data.iloc[label_position][target_column]

        sequences.append((sequence, label))
        # Append a tuple of (sequence, label) to the 'sequences' list

    return sequences

# Iterate over the columns to calculate rolling mean, max, and min values
for col in cols_to_scale:
    X_train[str(col) + 'rolling_mean'] = X_train[col].rolling(window=30).mean()
    # Calculate the rolling mean with a window size of 30 for each column

    X_train[str(col) + 'rolling_max'] = X_train[col].rolling(window=30).max()
    # Calculate the rolling maximum with a window size of 30 for each column

    X_train[str(col) + 'rolling_min'] = X_train[col].rolling(window=30).min()
    # Calculate the rolling minimum with a window size of 30 for each column

SEQ_LEN = 100
# Define the sequence length for creating sequences

train_sequences = create_sequences(X_train, "ML_Both_Arms", sequence_length=SEQ_LEN)
# Create training sequences with the specified sequence length and target column

test_sequences = create_sequences(X_test, "ML_Both_Arms", sequence_length=SEQ_LEN)
# Create test sequences with the specified sequence length and target column
