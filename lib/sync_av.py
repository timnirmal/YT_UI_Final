import pandas as pd


def sync_audio_and_video(audio_df, video_df):
    # new df
    new_df = pd.DataFrame(columns=["word", "a_start_time", "a_end_time", "a_start_time_floor", "a_end_time_ceil",
                                   "speaker", "text", "confidence", "processed_text", "classes", "audio_age_range",
                                   "audio_gen",
                                   "hate", "sentiment_word",
                                   "frame", "v_start_time", "v_end_time", "age", "gen", "emotion", "age_classes"
                                   ])
    # for each row in audio_df get the start_time and end_time
    for index, row in audio_df.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]

        # get the previous second by floor division
        start_time_floor = start_time // 1

        # get the next second by ceil division and add 1 if the ceil is equal to floor
        end_time_ceil = end_time // 1
        if end_time_ceil == start_time_floor:
            end_time_ceil += 1

        # add start_time_floor and end_time_ceil to audio_df
        # audio_df.loc[index, "start_time_floor"] = start_time_floor
        # audio_df.loc[index, "end_time_ceil"] = end_time_ceil

        # print("\n", "*" * 50, "\n", start_time, end_time, start_time_floor, end_time_ceil, "\n", "*" * 50, "\n")

        # get the rows in video_df that have the same start_time dont consider end_time
        video_df_2 = video_df[(video_df["start_time"] // 1 == start_time_floor)]
        print("Video 1 ===============================================================================")
        print(start_time, "  ", start_time_floor)
        print(end_time, "  ", end_time_ceil)
        print(video_df_2)

        if video_df_2.empty:
            # get the rows in video_df that have the same start_time dont consider end_time
            video_df_3 = video_df[(video_df["start_time"] // 2 == start_time_floor + 1)]
            print("Video 2 ===============================================================================")
            print(start_time_floor + 1)
            print(video_df_2)

            if video_df_3.empty:
                #     print("Video 3 ===============================================================================")
                #     print(start_time_floor + 2)
                #     print(video_df_2)
                #     # print("Empty")
                #     # add to new df
                #     new_df = pd.concat([new_df, pd.DataFrame(
                #         {'word': row["word"], 'a_start_time': row["start_time"], 'a_end_time': row["end_time"],
                #          'a_start_time_floor': start_time_floor, 'a_end_time_ceil': end_time_ceil,
                #          'speaker': row["speaker"], 'text': row["text"], 'confidence': row["confidence"],
                #          'processed_text': row["processed_text"], 'classes': row["classes"],
                #          'audio_age_range': row["audio_age_range"],
                #          'audio_gen': row["audio_gen"], 'hate': row["hate"], 'sentiment_word': row["sentiment_word"],
                #          'frame': pd.Series([], dtype='int64'), 'v_start_time': pd.Series([], dtype='int64'),
                #          'v_end_time': pd.Series([], dtype='int64'), 'age': pd.Series([], dtype='int64'),
                #          'gen': pd.Series([], dtype='int64'), 'emotion': pd.Series([], dtype='int64'),
                #          'age_classes': pd.Series([], dtype='int64')
                #          }, index=[0])], ignore_index=True)
                #     continue
                #
                # v_frame = video_df_3["frame"].iloc[0]
                # v_start_time = video_df_3["start_time"].iloc[0]
                # v_end_time = video_df_3["end_time"].iloc[0]
                # v_age = video_df_3["age"].iloc[0]
                # v_gen = video_df_3["gen"].iloc[0]
                # v_emotion = video_df_3["emotion"].iloc[0]
                # v_age_classes = video_df_3["age_classes"].iloc[0]
                #
                # # print(v_frame, v_start_time, v_end_time, v_age, v_gen, v_emotion, v_age_classes, v_label)
                #
                # # print(v_frame, v_start_time, v_end_time, v_age, v_gen, v_emotion)
                #
                # # add to new df
                # new_df = pd.concat([new_df, pd.DataFrame(
                #     {'word': row["word"], 'a_start_time': row["start_time"], 'a_end_time': row["end_time"],
                #      'a_start_time_floor': start_time_floor, 'a_end_time_ceil': end_time_ceil,
                #      'speaker': row["speaker"], 'text': row["text"], 'confidence': row["confidence"],
                #      'processed_text': row["processed_text"], 'classes': row["classes"],
                #      'audio_age_range': row["audio_age_range"],
                #      'audio_gen': row["audio_gen"], 'hate': row["hate"], 'sentiment_word': row["sentiment_word"],
                #      'frame': v_frame, 'v_start_time': v_start_time, 'v_end_time': v_end_time, 'age': v_age,
                #      'gen': v_gen, 'emotion': v_emotion, 'age_classes': v_age_classes
                #      }, index=[0])], ignore_index=True)
                print("Empty")
            continue

        v_frame = video_df_2["frame"].iloc[0]
        v_start_time = video_df_2["start_time"].iloc[0]
        v_end_time = video_df_2["end_time"].iloc[0]
        v_age = video_df_2["age"].iloc[0]
        v_gen = video_df_2["gen"].iloc[0]
        v_emotion = video_df_2["emotion"].iloc[0]
        v_age_classes = video_df_2["age_classes"].iloc[0]

        # print(v_frame, v_start_time, v_end_time, v_age, v_gen, v_emotion)

        # print(v_frame, v_start_time, v_end_time, v_age, v_gen, v_emotion)

        # add to new df
        new_df = pd.concat([new_df, pd.DataFrame(
            {'word': row["word"], 'a_start_time': row["start_time"], 'a_end_time': row["end_time"],
             'a_start_time_floor': start_time_floor, 'a_end_time_ceil': end_time_ceil,
             'speaker': row["speaker"], 'text': row["text"], 'confidence': row["confidence"],
             'processed_text': row["processed_text"], 'classes': row["classes"],
             'audio_age_range': row["audio_age_range"],
             'audio_gen': row["audio_gen"], 'hate': row["hate"], 'sentiment_word': row["sentiment_word"],
             'frame': v_frame, 'v_start_time': v_start_time, 'v_end_time': v_end_time, 'age': v_age,
             'gen': v_gen, 'emotion': v_emotion, 'age_classes': v_age_classes
             }, index=[0])], ignore_index=True)

    # replace null with previous value
    new_df = new_df.fillna(method='ffill')
    # remove a_start_time_floor and a_end_time_ceil
    new_df = new_df.drop(columns=["a_start_time_floor", "a_end_time_ceil"])

    # sort by a_start_time
    new_df = new_df.sort_values(by=['a_start_time'])

    # remove a_start_time and a_end_time and v_start_time and v_end_time confidence frame
    new_df = new_df.drop(columns=["a_start_time", "a_end_time", "v_start_time", "v_end_time", "confidence", "frame"])

    return new_df
