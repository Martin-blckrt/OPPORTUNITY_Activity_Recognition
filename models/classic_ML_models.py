import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Read the treated data from the CSV file
df = pd.read_csv('../treated_data/chunked_data.csv')
df = df.drop(df.columns[0], axis=1)

# Separate data and labels for each subject
df_s1, df_s2, df_s3, df_s4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
labels_s1, labels_s2, labels_s3, labels_s4 = [], [], [], []

for f in df['File'].unique():
    data = df[df['File'] == f]
    labels = data['ML_Both_Arms']
    data = data.drop(['ML_Both_Arms', 'File'], axis=1)

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

# Create lists of dataframes and labels for each subject
df_list = [df_s1, df_s2, df_s3, df_s4]
labels_list = [labels_s1, labels_s2, labels_s3, labels_s4]

# Define the metrics dictionary to store evaluation results
metrics = {"accuracy": [], "f1": [], "kappa": []}

# Define the models to evaluate
dt = DecisionTreeClassifier(random_state=0)
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(max_depth=3, random_state=0)
gbt = GradientBoostingClassifier(n_estimators=3, learning_rate=1.0, max_depth=1, random_state=0)

models = [dt, nb, knn, rf, gbt]

# Perform model evaluation
for model in models:
    for i, (X_test, y_test) in enumerate(zip(df_list, labels_list)):
        X_train = pd.concat([df for j, df in enumerate(df_list) if i != j])
        y_train = [element for j, sublist in enumerate(labels_list) if i != j for element in sublist]

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred, average='micro'))
        metrics["kappa"].append(cohen_kappa_score(y_test, y_pred))

    print("\n----- " + str(model) + " -----")
    for k, v in metrics.items():
        print(k + " is : " + str(sum(v) / len(v)))
