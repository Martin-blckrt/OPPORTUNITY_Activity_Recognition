import matplotlib.pyplot as plt
import pandas as pd

# Read the column names from the file
col_names = []
with open(r"C://Users/marti/PycharmProjects/Sujets_Speciaux/OpportunityUCIDataset/dataset/column_names.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.split(':')[0] == 'Column':
            if int(line.split(' ')[1]) == 1 or int(line.split(' ')[1]) >= 244:
                col_names.append(line.split(' ')[2])
            else:
                name = line.split(' ', maxsplit=2)[-1].split(';')[0]
                if name in col_names:
                    name = name.replace("X", "Y")
                    if name in col_names:
                        name = name.replace("Y", "Z")
                    col_names.append(name)
                else:
                    col_names.append(line.split(' ', maxsplit=2)[-1].split(';')[0])

# Set the folder path
folder_path = r"C://Users/marti/PycharmProjects/Sujets_Speciaux/OpportunityUCIDataset/dataset/"

# Read the data files for different subjects
df_s1 = pd.read_table(folder_path + 'S1-Drill.dat', sep="\s+", names=col_names)
df_s2 = pd.read_table(folder_path + 'S2-Drill.dat', sep="\s+", names=col_names)
df_s3 = pd.read_table(folder_path + 'S3-Drill.dat', sep="\s+", names=col_names)
df_s4 = pd.read_table(folder_path + 'S4-Drill.dat', sep="\s+", names=col_names)

# Create a list of dataframes for each subject
dfs = [df_s1, df_s2, df_s3, df_s4]

# Define unused labels and the kept label
unused_labels = ['Locomotion', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object', 'LL_Right_Arm', 'LL_Right_Arm_Object']
kept_label = ['ML_Both_Arms']

# Preprocess the dataframes for each subject
for i, df in enumerate(dfs):
    for col in df.columns:
        if col.split(' ')[0] == 'REED' or col in unused_labels:
            dfs[i] = dfs[i].drop(col, axis=1)
    dfs[i] = dfs[i].fillna(method='ffill')
    dfs[i] = dfs[i].fillna(0)

subs = ["S1", "S2", "S3", "S4"]
colors = ['red', 'blue', 'green', 'orange']

# Plot line plots and kernel density estimate plots for each column
for col in dfs[0].columns[1:]:
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(50, 15))

    for i, df in enumerate(dfs):
        # Rename the column as per the subject
        dfs[i][subs[i]] = dfs[i][col]

        # Plot line plot
        dfs[i].plot(x="MILLISEC", y=[subs[i]], color=colors[i], ls="--", ax=axs[0])
        axs[0].set_title('Line Plot')

        # Plot kernel density estimate plot
        dfs[i].plot(x="MILLISEC", y=[subs[i]], color=colors[i], kind="kde", ax=axs[1])
        axs[1].set_title('Kernel Density Estimate Plot')

    fig.suptitle(col + " for the different subject drill runs")
    plt.savefig('./exploration/Line_and_KDE_Drills/' + col + '.jpg')
    plt.close()

print("done")
