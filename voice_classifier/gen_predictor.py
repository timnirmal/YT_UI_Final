import pandas as pd
import seaborn as sb

sb.set_style("whitegrid", {'axes.grid': False})
pd.set_option('display.max_columns', None)
show_plot = False

# save model
import pickle


def voice_gen_predict_df(df):
    # load model
    with open('models/voice/gen_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # load scaler
    with open('models/voice/gen_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    features = ['mean', 'skew', 'kurtosis', 'median', 'mode', 'std', 'low', 'peak', 'q25', 'q75', 'iqr']

    # scale
    X = scaler.transform(df.loc[:, features])

    # predict
    y_pred = model.predict(X)

    # add prediction to df
    df['audio_gen'] = y_pred

    return df
