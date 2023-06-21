import pandas as pd

df = pd.read_csv('../treated_data/data.csv')
# Read the CSV file into a DataFrame named 'df'

df = df.drop(df.columns[0], axis=1)
# Drop the first column from the 'df' DataFrame

chunk_df = pd.read_csv('../treated_data/chunked_data.csv')
# Read another CSV file into a DataFrame named 'chunk_df'

chunk_df = chunk_df.drop(chunk_df.columns[0], axis=1)
# Drop the first column from the 'chunk_df' DataFrame

print("length of original dataset : ", len(df))
# Print the length (number of rows) of the 'df' DataFrame

print("length of original dataset/30 : ", len(df)/30)
# Print the length of the 'df' DataFrame divided by 30

print("length of chunky dataset : ", len(chunk_df))
# Print the length (number of rows) of the 'chunk_df' DataFrame

print("% kept : ", (len(chunk_df)/(len(df)/30))*100)
# Calculate and print the percentage of rows kept in 'chunk_df' compared to 'df'

print("% loss : ", (100 - (len(chunk_df)/(len(df)/30))*100))
# Calculate and print the percentage of rows lost from 'df' to create 'chunk_df'
