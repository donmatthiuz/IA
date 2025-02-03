import pandas as pd

# Replace the path with your actual file path
csv_file_path = 'task3\\high_diamond_ranked_10min.csv'

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the DataFrame
print(df.columns)



