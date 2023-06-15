import pandas as pd
import seaborn as sb

sb.set_style("whitegrid", {'axes.grid': False})
pd.set_option('display.max_columns', None)

# save model
import pickle


def voice_age_predict_df(df):
    # load model
    with open('models/voice/age_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # load scaler
    with open('models/voice/age_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # load label encoder
    with open('models/voice/age_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    features = ['mean', 'skew', 'kurtosis', 'median', 'mode', 'std', 'low', 'peak', 'q25', 'q75', 'iqr']

    # count Nan
    print(df.isnull().sum())

    # print how labels are encoded
    print(le.classes_)
    print(le.transform(le.classes_))

    # Standardize features by removing the mean and scaling to unit variance
    X = scaler.transform(df.loc[:, features])

    # predict
    y_pred = model.predict(X)

    # # inverse transform
    # y_pred = le.inverse_transform(y_pred)

    # add prediction to df
    df['audio_age_range'] = y_pred

    return df
