import os
import pandas as pd

# Specify the folder containing the .csv files
folder_path = 'accuracy_reports_std'
output_file = 'combined_accuracy_std_11_02.csv'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty dictionary to store data
data = {}

# Process each file
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure the file has at least two columns
    if df.shape[1] < 2:
        print(f"Skipping file {file}: less than 2 columns.")
        continue

    # Extract graph names and accuracy values
    graph_names = df.iloc[:, 0]
    accuracy_values = df.iloc[:, 1]

    # Use the graph names as the keys for the dictionary
    if not data:
        data['Graph Names'] = graph_names

    # Add accuracy values under the column named after the file
    data[file] = accuracy_values

# Convert the dictionary to a DataFrame
combined_df = pd.DataFrame(data)

# Save the combined DataFrame to a new CSV file
output_path = os.path.join(folder_path, output_file)
combined_df.to_csv(output_path, index=False)

print(f"Combined data has been saved to {output_path}")
