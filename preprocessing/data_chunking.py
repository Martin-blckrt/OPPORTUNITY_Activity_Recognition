import pandas as pd
from tqdm import trange


def extract_statistics(df, feature_cols, num_chunks):
    # Create an empty dataframe to store the extracted statistics
    extracted_df = pd.DataFrame()

    extra_count = 0

    for i in trange(num_chunks):
        # Extract a chunk of 30 rows from the original dataframe
        chunk = df.iloc[i * 30: (i + 1) * 30]

        # Check if there is only one unique label in the chunk
        if len(chunk['ML_Both_Arms'].unique()) != 1:
            extra_count += 1
            continue  # Skip the chunk if there are multiple labels

        # Extract the label from the chunk
        label = chunk['ML_Both_Arms'].unique()[0]

        chunk_stats = pd.DataFrame()

        # Calculate the desired statistics for the selected feature columns
        for col in feature_cols:
            chunk_col = pd.DataFrame({f"{col}_mean": chunk[col].mean(),
                                      f"{col}_max": chunk[col].max(),
                                      f"{col}_min": chunk[col].min(),
                                      f"{col}_std": chunk[col].std(),
                                      f"{col}_median": chunk[col].median(),
                                      f"{col}_skew": chunk[col].skew(),
                                      f"{col}_kurt": chunk[col].kurt()}, index=[0])

            chunk_stats = pd.concat([chunk_stats, chunk_col], axis=1)

        # Add the label and file to the chunk statistics dataframe
        add = pd.DataFrame({"ML_Both_Arms": label, "File": chunk['File'].unique()[0]}, index=[0])
        chunk_stats = pd.concat([chunk_stats, add], axis=1)

        # Append the chunk statistics to the extracted dataframe
        extracted_df = pd.concat([extracted_df, chunk_stats], axis=0)

    return extracted_df


# Read the CSV file into a pandas dataframe
df = pd.read_csv('../treated_data/data.csv')
df = df.drop([df.columns[0], df.columns[1]], axis=1)

# Extract the feature columns
features = [i for i in df.columns if i not in ['MILLISEC', 'ML_Both_Arms', 'File']]

# Create an empty dataframe to store the chunked data
chunked_data = pd.DataFrame()

# Iterate over each unique file in the dataframe
for f in df['File'].unique():
    # Select data for the current file
    data = df[df['File'] == f]

    # Extract statistics for the data and concatenate it to the chunked data
    data_stats = extract_statistics(data, features, len(data) // 30)

    chunked_data = pd.concat([chunked_data, data_stats], axis=0)

# Save the chunked data to a CSV file
chunked_data.to_csv("../treated_data/chunked_data.csv")
