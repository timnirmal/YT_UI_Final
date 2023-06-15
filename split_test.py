import pandas as pd

# show all columns
pd.set_option('display.max_columns', None)

df = pd.read_csv("runs/run_1/recognized_processed_classes_features_age_gen.csv")

num_of_chunks = len(df)//10

for i in range(num_of_chunks+1):
    start = i*10
    end = start + 10
    chunk_df = df[start:end]

    # join word in word column
    joined_text = " ".join(chunk_df['word'].tolist())

    # replace text column with joined text
    chunk_df['text'].replace(chunk_df['text'].tolist(), joined_text, inplace=True)

# save the df
df.to_csv("runs/run_1/recognized_processed_classes_features_age_gen_chunked.csv", index=False)
