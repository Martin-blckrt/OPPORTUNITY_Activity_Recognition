import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Function to retrieve chunked data as a list of dataframes and labels
def get_chunked_data_as_list(encoded_labels=False):
    df = pd.read_csv('../treated_data/chunked_data.csv')
    df = df.drop(df.columns[0], axis=1)

    # Initialize dataframes and label lists
    df_s1, df_s2, df_s3, df_s4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    labels_s1, labels_s2, labels_s3, labels_s4 = [], [], [], []

    # Iterate over unique files
    for f in df['File'].unique():
        data = df[df['File'] == f]
        labels = data['ML_Both_Arms']
        data = data.drop(['ML_Both_Arms', 'File'], axis=1)

        # Split data based on file name prefixes
        if f.split('-')[0] == 'S1':
            df_s1 = pd.concat([df_s1, data])
            labels_s1.extend(labels)
        elif f.split('-')[0] == 'S2':
            df_s2 = pd.concat([df_s2, data])
            labels_s2.extend(labels)
        elif f.split('-')[0] == 'S3':
            df_s3 = pd.concat([df_s3, data])
            labels_s3.extend(labels)
        elif f.split('-')[0] == 'S4':
            df_s4 = pd.concat([df_s4, data])
            labels_s4.extend(labels)

    # Combine dataframes and labels into lists
    df_list = [df_s1, df_s2, df_s3, df_s4]
    big_df = pd.concat([df for df in df_list], axis=0)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(big_df)
    df_list = [pd.DataFrame(scaler.transform(df), columns=df.columns) for df in df_list]

    return df_list, [labels_s1, labels_s2, labels_s3, labels_s4]


def get_chunked_drill_vs_adl_as_list():
    df = pd.read_csv('../treated_data/chunked_data.csv')
    df = df.drop(df.columns[0], axis=1)

    subject_list = ['S1', 'S2', 'S3', 'S4']

    drill_dfs = {subject: pd.DataFrame() for subject in subject_list}
    adl_dfs = {subject: pd.DataFrame() for subject in subject_list}

    drill_labels = {subject: [] for subject in subject_list}
    adl_labels = {subject: [] for subject in subject_list}

    for f in df['File'].unique():
        data = df[df['File'] == f]
        labels = data['ML_Both_Arms']
        data = data.drop(['ML_Both_Arms', 'File'], axis=1)

        f_name = f.split('-')
        run_type = f_name[1].split('.')[0]
        subject = f_name[0]

        (drill_dfs if run_type == 'Drill' else adl_dfs)[subject] = pd.concat(
            [(drill_dfs if run_type == 'Drill' else adl_dfs)[subject], data])
        (drill_labels if run_type == 'Drill' else adl_labels)[subject].extend(labels)

    # Construct the lists of dataframes and labels
    drill_dataframes = [drill_dfs[subject] for subject in subject_list]
    drill_labels = [drill_labels[subject] for subject in subject_list]
    adl_dataframes = [adl_dfs[subject] for subject in subject_list]
    adl_labels = [adl_labels[subject] for subject in subject_list]

    # Return the lists of dataframes and labels
    return drill_dataframes, drill_labels, adl_dataframes, adl_labels


def leave_one_out_crossval(model, df_list, labels_list, metrics, encoded=True):
    met = {"accuracy": [], "f1": [], "kappa": []}
    for i, (X_test, y_test) in enumerate(tqdm(zip(df_list, labels_list))):
        X_train = pd.concat([df for j, df in enumerate(df_list) if i != j])
        y_train = [element for j, sublist in enumerate(labels_list) if i != j for element in sublist]

        if encoded:
            encoder = LabelEncoder().fit(y_train)
            model = model.fit(X_train, encoder.transform(y_train))
            y_pred = model.predict(X_test)
            y_test = encoder.transform(y_test)
        else:
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        met["accuracy"].append(accuracy_score(y_test, y_pred))
        met["f1"].append(f1_score(y_test, y_pred, average='micro'))
        met["kappa"].append(cohen_kappa_score(y_test, y_pred))

    print("\n----- " + str(model) + " -----")
    for k, v in met.items():
        if k in metrics:
            print(k + " is : " + str(sum(v) / len(v)))

    return met


def get_specific_chunked_data_as_list(sub="None"):
    if sub not in ['objects', 'location', 'body']:
        print("Dataset sub not supported")

    df = pd.read_csv('../treated_data/chunked_data.csv')
    df = df.drop(df.columns[0], axis=1)

    sub_cols = []
    inv_cols = []

    for col in df.columns[:-2]:
        sens = col.split(' ')[1]
        if sub == 'objects':
            if sens in ['UPPERDRAWER', 'MIDDLEDRAWER', 'LOWERDRAWER', 'DOOR1', 'DOOR2', 'LAZYCHAIR', 'FRIDGE', 'DISHWASHER',
                        'PLATE', 'SPOON', 'CUP', 'KNIFE1', 'GLASS', 'KNIFE2', 'SUGAR', 'BREAD', 'SALAMI', 'MILK', 'WATER',
                        'CHEESE']:
                sub_cols.append(col)
            else:
                inv_cols.append(col)
        elif sub == 'location':
            if sens in ['TAG1', 'TAG2', 'TAG3', 'TAG4']:
                sub_cols.append(col)
            else:
                inv_cols.append(col)
        elif sub == 'body':
            if sens in ['UPPERDRAWER', 'MIDDLEDRAWER', 'LOWERDRAWER', 'DOOR1', 'DOOR2', 'LAZYCHAIR', 'FRIDGE', 'DISHWASHER',
                        'PLATE', 'SPOON', 'CUP', 'KNIFE1', 'GLASS', 'KNIFE2', 'SUGAR', 'BREAD', 'SALAMI', 'MILK', 'WATER',
                        'CHEESE', 'TAG1', 'TAG2', 'TAG3', 'TAG4']:
                inv_cols.append(col)
            else:
                sub_cols.append(col)

    subject_list = ['S1', 'S2', 'S3', 'S4']

    sub_dfs = {subject: pd.DataFrame() for subject in subject_list}
    inv_dfs = {subject: pd.DataFrame() for subject in subject_list}

    labels_list = {subject: [] for subject in subject_list}

    for f in df['File'].unique():
        data = df[df['File'] == f]
        labels = data['ML_Both_Arms']
        data = data.drop(['ML_Both_Arms', 'File'], axis=1)
        subject = f.split('-')[0]

        sub_dfs[subject] = pd.concat([sub_dfs[subject], data[sub_cols]])
        inv_dfs[subject] = pd.concat([inv_dfs[subject], data[inv_cols]])

        labels_list[subject].extend(labels)

    # Construct the lists of dataframes and labels
    sub_dataframes = [sub_dfs[subject] for subject in subject_list]
    inv_dataframes = [inv_dfs[subject] for subject in subject_list]
    labels_list = [labels_list[subject] for subject in subject_list]

    # Return the lists of dataframes and labels
    return sub_dataframes, inv_dataframes, labels_list


def get_sequenced_data():
    df = pd.read_csv('../treated_data/data.csv')
    df = df.drop(df.columns[0], axis=1)

    df_s1 = np.empty((0, 30, 229), dtype=object)
    df_s2 = np.empty((0, 30, 229), dtype=object)
    df_s3 = np.empty((0, 30, 229), dtype=object)
    df_s4 = np.empty((0, 30, 229), dtype=object)
    labels_s1, labels_s2, labels_s3, labels_s4 = [], [], [], []

    for f in df['File'].unique():
        data = df[df['File'] == f]
        labels = data['ML_Both_Arms']
        data = data.drop(['ML_Both_Arms', 'File', 'MILLISEC'], axis=1)

        num_seconds = int(len(data) / 30)
        actual_count = num_seconds

        for i in range(num_seconds):
            start_idx = i * 30
            end_idx = start_idx + 30
            if len(np.unique(labels.values[start_idx:end_idx])) != 1:
                actual_count -= 1

        data_3d = np.zeros((actual_count, 30, data.shape[1]))
        # data_3d = np.zeros((num_seconds, data.shape[1], 30))
        labels1d = np.empty(actual_count, dtype=object)

        index = 0
        for i in range(num_seconds):
            start_idx = i * 30
            end_idx = start_idx + 30
            if len(np.unique(labels.values[start_idx:end_idx])) != 1:
                continue
            data_3d[index] = data[start_idx:end_idx]
            labels1d[index] = np.unique(labels.values[start_idx:end_idx])[0]
            index += 1

        if f.split('-')[0] == 'S1':
            df_s1 = np.concatenate((df_s1, data_3d))
            labels_s1.extend(labels1d)
        elif f.split('-')[0] == 'S2':
            df_s2 = np.concatenate((df_s2, data_3d))
            labels_s2.extend(labels1d)
        elif f.split('-')[0] == 'S3':
            df_s3 = np.concatenate((df_s3, data_3d))
            labels_s3.extend(labels1d)
        elif f.split('-')[0] == 'S4':
            df_s4 = np.concatenate((df_s4, data_3d))
            labels_s4.extend(labels1d)

    return [df_s1, df_s2, df_s3, df_s4], [labels_s1, labels_s2, labels_s3, labels_s4]


def plot_double_bar(title, xlab, ylab, X, y1, y2, y1lab, y2lab):
    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, y1, 0.4, label=y1lab)
    for i in range(len(X)):
        plt.text(i - 0.2, y1[i] / 2, round(y1[i], 2), ha='center')

    plt.bar(X_axis + 0.2, y2, 0.4, label=y2lab)
    for i in range(len(X)):
        plt.text(i + 0.2, y2[i] / 2, round(y2[i], 2), ha='center')

    plt.xticks(X_axis, X)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()


def plot_line(list, X, title, X_label, Y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(X, list, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.show()


def plot_conf_mat(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()
