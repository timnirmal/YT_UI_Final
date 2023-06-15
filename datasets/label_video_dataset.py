import pandas as pd

df = pd.read_csv("video_dataset.csv")


def assign_severity_score(row):
    emotion = row['emotion']
    gender = row['gen']
    age_group = row['age_classes']

    if emotion == 'Angry':
        if gender == 'M':
            if age_group == 0:
                return 5
            elif age_group == 1:
                return 4
            elif age_group == 2:
                return 4
            elif age_group == 3:
                return 3
        elif gender == 'F':
            if age_group == 0:
                return 5
            elif age_group == 1:
                return 4
            elif age_group == 2:
                return 4
            elif age_group == 3:
                return 4

    elif emotion == 'Fear':
        if gender == 'M':
            if age_group == 0:
                return 4
            elif age_group == 1:
                return 3
            elif age_group == 2:
                return 3
            elif age_group == 3:
                return 2
        elif gender == 'F':
            if age_group == 0:
                return 4
            elif age_group == 1:
                return 3
            elif age_group == 2:
                return 3
            elif age_group == 3:
                return 4

    elif emotion == 'Happy':
        return 1

    elif emotion == 'Neutral':
        if gender == 'M' and age_group == 3:
            return 1
        elif gender == 'F' and age_group == 3:
            return 1
        else:
            return 1

    elif emotion == 'Sad':
        if gender == 'M' and age_group == 3:
            return 2
        elif gender == 'F' and age_group == 3:
            return 3
        else:
            return 2

    elif emotion == 'Surprise':
        return 1

    # Default severity score if no specific rule matches
    return 1


df['severity_score'] = df.apply(assign_severity_score, axis=1)
df.to_csv("video_dataset.csv", index=False)