import pandas as pd

# Load your data
file_path = 'cleaned_cms.csv'  # Update this with your actual file path
df = pd.read_csv(file_path)

# Standardize the room_type column by stripping whitespace and replacing similar characters
df['room_type'] = df['room_type'].str.strip()  # Remove leading/trailing whitespace
df['room_type'] = df['room_type'].replace({
    'ي': 'ی',   # Replace Arabic "ye" with Persian "ye"
    'ك': 'ک',   # Replace Arabic "kaf" with Persian "kaf"
    '\u200c': '' # Remove any zero-width non-joiners if present
}, regex=True)

# Define the mapping for room type grouping
room_type_mapping = {
    'آپارتمان_سوئیت': ['سوئیت', 'آپارتمان', 'سوییت'],
    'خانه': ['ویلا', 'خانه', 'روستایی'],
    'اقامتگاه_بومگردی': ['بوم گردی', 'اقامتگاه', 'اقامتگاه سنتی', 'مجتمع اقامتگاهی', 'مهمانسرا', 'مسافرخانه', 'کاروانسرا'],
    'هتل': ['هاستل', 'هتل'],
    'کلبه': ['کلبه']
}

# Create a reverse mapping for easy lookup
reverse_mapping = {}
for group, types in room_type_mapping.items():
    for t in types:
        reverse_mapping[t] = group

# Map the room types to the new categories directly in the room_type column
df['room_type'] = df['room_type'].map(reverse_mapping)

# Save the updated DataFrame to a new CSV file
output_file_path = 'cms_grouped.csv'
df.to_csv(output_file_path, index=False)

# Display the counts for each grouped room type
group_counts = df['room_type'].value_counts()
print("Grouped room types and their counts:\n")
print(group_counts)
