# Move to Main Folder for run

import pandas as pd

df = pd.read_csv("audio_dataset.csv")

# drop audio_age_range and audio_gen
df = df.drop(['audio_age_range', 'audio_gen'], axis=1)


def calculate_severity(domain, hate, sentiment):
    domain_severity = {'Political': 3, 'Religious': 3, 'Sexual': 2, 'Entertainment': 1, 'Education': 1, 'Sports': 1}
    # age_severity = {0: 1, 1: 3, 2: 2, 3: 0}
    # gender_severity = {0: 2, 1: 1}
    hate_severity = {1: 5, 0: 0}
    sentiment_severity = {-1: 3, 0: 1, 1: 0}

    severity = domain_severity.get(domain, 0)
    # severity *= age_severity.get(age_group, 0)
    # severity *= gender_severity.get(gender, 0)
    severity += hate_severity.get(hate, 0)
    severity += sentiment_severity.get(sentiment, 0)

    return severity


# # Example usage:
# domain = 'Political'
# hate = 1
# sentiment = 1

# Calculate severity for each row
df['Severity'] = df.apply(lambda row: calculate_severity(row['classes'], row['hate'], row['sentiment_word']), axis=1)

# Normalize severity values to a range of 1 to 5
min_severity = df['Severity'].min()
max_severity = df['Severity'].max()
if min_severity == 1 and max_severity == 6:
    # for df['Severity'] = 6 replace with 5
    df['Normalized Severity'] = df['Severity'].replace(6, 5)


#
# df['Normalized Severity'] = df['Severity'].apply(
#     lambda x: round(((x - min_severity) / ((max_severity - min_severity) + 1)) * 4) + 1)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('datasets/audio_dataset_severity.csv', index=False)
