import pandas as pd

# Load the CSV files
file1 = "cm_jabama.csv"
file2 = "comments_selenium.csv"

# Read the CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the dataframes using the lowercase common columns
merged_df = pd.merge(df1, df2, on=["page", "location", "room_type", "comment"], how="outer")

# Save the combined data to a new CSV file
merged_df.to_csv("merged_comments.csv", index=False)

print("CSV files combined successfully into merged_comments.csv!")
