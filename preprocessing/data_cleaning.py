import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

col_names = []
with open(r"C://Users/marti/PycharmProjects/Sujets_Speciaux/OpportunityUCIDataset/dataset/column_names.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.split(':')[0] == 'Column':
            if int(line.split(' ')[1]) == 1 or int(line.split(' ')[1]) >= 244:
                col_names.append(line.split(' ')[2])  # Collect column names from the file

            else:
                name = line.split(' ', maxsplit=2)[-1].split(';')[0]
                if name in col_names:
                    name = name.replace("X", "Y")
                    if name in col_names:
                        name = name.replace("Y", "Z")
                    col_names.append(name)  # Append modified column names if duplicates exist
                else:
                    col_names.append(line.split(' ', maxsplit=2)[-1].split(';')[0])  # Append column names

string_targets = {}
with open(r"C://Users/marti/PycharmProjects/Sujets_Speciaux/OpportunityUCIDataset/dataset/label_legend.txt") as f:
    lines = f.readlines()
    for index, line in enumerate(lines):
        if index in [0, 1]:
            continue
        string_targets[int(line.split('-')[0])] = line.split('-')[-1].strip(), line.split('-')[1].strip()
        # Collect string targets from the file

folder_path = r"C://Users/marti/PycharmProjects/Sujets_Speciaux/OpportunityUCIDataset/dataset/"
dat_files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
# Get a list of .dat files in the specified folder path

unused_labels = ['Locomotion', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object']
label = ['ML_Both_Arms']

df = pd.DataFrame()

for file in dat_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_table(file_path, sep="\s+", names=col_names)
    # Read data from each file into a DataFrame

    for col in data.columns:
        if col.split(' ')[0] == 'REED' or col in unused_labels:
            data = data.drop(col, axis=1)  # Drop unwanted columns
        elif col == 'ML_Both_Arms':
            for k in string_targets:
                data['ML_Both_Arms'] = data['ML_Both_Arms'].replace(k, string_targets[k][0])
            data['ML_Both_Arms'] = data['ML_Both_Arms'].replace(0, "No action")
            # Replace values in the 'ML_Both_Arms' column with string targets

    data = data.fillna(method='ffill')  # Forward-fill missing values
    data = data.fillna(0)  # Fill remaining missing values with 0
    data['File'] = file  # Add a 'File' column with the file name
    df = pd.concat([df, data])  # Concatenate current data with the main DataFrame

encoder = LabelEncoder()
encoder.fit(df['ML_Both_Arms'])
print(list(encoder.classes_))  # Print the classes learned by the LabelEncoder

# df['ML_Both_Arms'] = encoder.transform(df['ML_Both_Arms'])
# Uncomment the above line to transform the 'ML_Both_Arms' column with the LabelEncoder

df.to_csv("../treated_data/data.csv")  # Save the DataFrame to a CSV file
