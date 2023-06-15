import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('classes_dataset.csv')  # Replace with your actual file path

# label encoder
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

labels = data['class'].tolist()

# get dummy variables
encoded_labels = pd.get_dummies(labels).values
print(encoded_labels.shape)

