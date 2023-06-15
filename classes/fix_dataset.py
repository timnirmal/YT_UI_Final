import pandas as pd

# show all columns
pd.set_option('display.max_columns', None)

df = pd.read_csv("classes_dataset_processed.csv")


# remove á ñá á
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'á', '')
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'ñá', '')
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'ñ', '')

# remove if only have “ and whitespace
df = df[~df['filtered_sentence'].str.match(r'^\s*"\s*$')]

# remove ” ‘ ’ “ –
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'”', '')
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'‘', '')
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'’', '')
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'“', '')
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'–', '')


# if filtered_sentences column is empty remove raw
df['filtered_sentence'].fillna(df['text'], inplace=True)

# remove leading and trailing spaces
df['filtered_sentence'] = df['filtered_sentence'].str.strip()

# remove null
df = df.dropna()


# save csv
df.to_csv("classes_dataset_processed.csv", index=False)

# remove Education from classes column
df = df[~df['class'].str.match(r'^Education$')]

# remove emojis
df['filtered_sentence'] = df['filtered_sentence'].str.replace(r'[^\w\s]', '')

# remove leading and trailing spaces
df['filtered_sentence'] = df['filtered_sentence'].str.strip()

# remove null
df = df.dropna()

# save
df.to_csv("classes_dataset_final.csv", index=False)
