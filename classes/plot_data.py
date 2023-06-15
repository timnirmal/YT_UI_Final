import pandas as pd
from matplotlib import pyplot as plt

# show all columns
pd.set_option('display.max_columns', None)

df = pd.read_csv("classes_dataset_cleaned.csv")

# plot data distribution
df['class'].value_counts().plot(kind='bar')
plt.show()
