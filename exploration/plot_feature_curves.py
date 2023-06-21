import matplotlib.pyplot as plt
import pandas as pd

# Read the train and test data from CSV files
train_df = pd.read_csv('../treated_data/train.csv')
test_df = pd.read_csv('../treated_data/test.csv')

# Concatenate the train and test dataframes
df = pd.concat([train_df, test_df])

# Drop the first column from the concatenated dataframe
df.drop(columns=df.columns[0], axis=1, inplace=True)

# Group the columns based on their names
groups = {}
for col in df.columns[1:-7]:
    if col.split(' ')[0] in ['Accelerometer', 'InertialMeasurementUnit']:
        name = col.rsplit(' ', 1)[0]
    else:
        name = col.rsplit(' ', 2)[0]
    if name == 'REED':
        name = 'REED SWITCH'
    if name in groups:
        groups[name].append(col)
    else:
        groups[name] = [col]

# Plot the groups of columns
for g in groups:
    if len(groups[g]) > 10:
        df[groups[g]].plot(subplots=True, legend=False, title=groups[g], figsize=(15, 25))
    elif len(groups[g]) > 5:
        df[groups[g]].plot(subplots=True, legend=False, title=groups[g], figsize=(15, 20))
    else:
        df[groups[g]].plot(subplots=True, legend=False, title=groups[g], figsize=(15, 15))
    plt.savefig('./feature_plots/' + g + '.jpg')
