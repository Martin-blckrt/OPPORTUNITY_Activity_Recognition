import matplotlib.pyplot as plt
import pandas as pd

# Read the train and test data from CSV files
train_df = pd.read_csv('../treated_data/train.csv')
test_df = pd.read_csv('../treated_data/test.csv')

# Concatenate the train and test dataframes
df = pd.concat([train_df, test_df])

# Drop the first column from the concatenated dataframe
df.drop(columns=df.columns[0], axis=1, inplace=True)

# Initialize a dictionary to store the remapped labels
remap = {}

# Read the label_legend.txt file and populate the remap dictionary
with open(r"C://Users/marti/PycharmProjects/Sujets_Speciaux/OpportunityUCIDataset/dataset/label_legend.txt") as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        if index in [0, 1]:
            continue
        remap[int(line.split('-')[0])] = line.split('-')[-1].strip(), line.split('-')[1].strip()

# Uncomment the following lines to remap the labels in the dataframe
"""
for k in remap:
    df[remap[k][1]] = df[remap[k][1]].replace(k, remap[k][0])
    df[remap[k][1]] = df[remap[k][1]].replace(0, "None")
"""

# Plot histograms and line plots for the last 7 columns of the dataframe
for col in df.columns[-7:]:
    # Plot histogram
    df[col].hist(grid=False, figsize=(15, 15))
    plt.title(col)
    plt.show()

    # Plot line plot
    df[col].plot.line(title=col, figsize=(15, 15))
    plt.show()
