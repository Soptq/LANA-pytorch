# This sample script preprocess RAIEd2021 dataset
# Minor modification can be made to this script to make it compatible for other datasets

import random
import gc
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

parser = argparse.ArgumentParser(description="Sample Data Preprocess Script")
parser.add_argument('-t', '--train_csv', type=str, required=True,
                    help="Filepath of train.csv")
parser.add_argument('-q', '--question_csv', type=str, required=True,
                    help="Filepath of question.csv")
parser.add_argument('-s', '--split', type=float, required=True,
                    help="Testset / Trainset")
parser.add_argument('-o', '--output', type=str, required=True,
                    help="Filepath of preprocessed data")
parser.add_argument('--irt', action='store_true',
                    help="Whether to perform IRT analysis. Required if you will do leveled learning")
args = parser.parse_args()

print("Loading CSV...")
train_dtypes = {'row_id': 'int64',
                'timestamp': 'int64',
                'user_id': 'int32',
                'content_id': 'int16',
                'content_type_id': 'int8',
                'answered_correctly': 'int8',
                'user_answer': 'int8',
                'prior_question_elapsed_time': 'float32',
                'task_container_id': 'int16',
                'prior_question_had_explanation': 'boolean'}
train_df = pd.read_csv(args.train_csv, dtype=train_dtypes)
question_dtypes = {"question_id": "int16", "part": "int8"}
question_df = pd.read_csv(args.question_csv, dtype=question_dtypes)
# Align the name of key column for latter merging
question_df = question_df.rename(columns={"question_id": "content_id"})

# Formatting the timestamp
# Here we basically want to reset the timestamp of each record so that all users
# do their respective last exercise at almost the same time (Instead of 0).
# This step is usefull for splitting training/valid dataset as we dont want to
# randomly split the dataset
#
# In order to do that, we firstly need to get the max timestamp of all records
# Then we use it to minus the max time stamp of each user to represent the start
# timestamp of this specific user
#
# Insipred from https://www.kaggle.com/its7171/cv-strategy

max_timestamp_user = train_df[["user_id", "timestamp"]].groupby(["user_id"]).agg(["max"]).reset_index()
max_timestamp_user.columns = ["user_id", "max_timestamp"]
MAX_TIMESTAMP = max_timestamp_user.max_timestamp.max()
print("Generating virtual timestamp")


def reset_time(max_timestamp):
    gap = MAX_TIMESTAMP - max_timestamp
    rand_init_time = random.randint(0, gap)
    return rand_init_time


max_timestamp_user["rand_timestamp"] = max_timestamp_user.max_timestamp.progress_apply(reset_time)
train_df = train_df.merge(max_timestamp_user, on="user_id", how="left")
train_df["virtual_timestamp"] = train_df.timestamp + train_df["rand_timestamp"]

del max_timestamp_user
gc.collect()

# Merging train_df and question_df on
train_df = train_df[train_df.content_type_id == 0]  # only consider question
train_df = train_df.merge(question_df, on='content_id', how="left")  # left outer join to consider part
train_df.prior_question_elapsed_time /= 1000  # ms -> s
train_df.prior_question_elapsed_time.fillna(0, inplace=True)
train_df.prior_question_elapsed_time.clip(lower=0, upper=300, inplace=True)
train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)

del question_df
gc.collect()

train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].fillna(value=False).astype(int)

train_df = train_df.sort_values(["virtual_timestamp", "row_id"]).reset_index(drop=True)
n_content_ids = len(train_df.content_id.unique())
n_parts = len(train_df.part.unique())
print("NO. of exercises:", n_content_ids)
print("NO. of part", n_parts)
print("Shape of the dataframe after exclusion:", train_df.shape)

print("Computing question difficulty")
df_difficulty = train_df["answered_correctly"].groupby(train_df["content_id"])
train_df["popularity"] = df_difficulty.transform('size')
train_df["difficulty"] = df_difficulty.transform('sum') / train_df["popularity"]
print("Popularity max", train_df["popularity"].max(), ",Difficulty max", train_df["difficulty"].max())

del df_difficulty
gc.collect()

print("Calculating lag time")
time_dict = {}

lag_time_col = np.zeros(len(train_df), dtype=np.int64)
for ind, row in enumerate(tqdm(train_df[["user_id", "timestamp", "task_container_id"]].values)):
    if row[0] in time_dict.keys():
        # if the task_container_id is the same, the lag time is not allowed
        if row[2] == time_dict[row[0]][1]:
            lag_time_col[ind] = time_dict[row[0]][2]
        else:
            timestamp_last = time_dict[row[0]][0]
            lag_time_col[ind] = row[1] - timestamp_last
            time_dict[row[0]] = (row[1], row[2], lag_time_col[ind])
    else:
        time_dict[row[0]] = (row[1], row[2], 0)
        lag_time_col[ind] = 0
    if lag_time_col[ind] < 0:
        raise RuntimeError("Has lag_time smaller than 0.")

train_df["lag_time_s"] = lag_time_col // 1000
train_df["lag_time_m"] = lag_time_col // (60 * 1000)
train_df["lag_time_d"] = lag_time_col // (60 * 1000 * 1440)
train_df.lag_time_s.clip(lower=0, upper=300, inplace=True)
train_df.lag_time_m.clip(lower=0, upper=1440, inplace=True)
train_df.lag_time_d.clip(lower=0, upper=365, inplace=True)
train_df.lag_time_s = train_df.lag_time_s.astype(np.int)
train_df.lag_time_m = train_df.lag_time_m.astype(np.int)
train_df.lag_time_d = train_df.lag_time_d.astype(np.int)

del lag_time_col
gc.collect()

print("Add special token")
train_df.content_id = train_df.content_id + 2  # PAD and START
train_df.answered_correctly = train_df.answered_correctly + 2  # PAD and START
train_df.part = train_df.part + 1  # part has no 0.
train_df.prior_question_had_explanation = train_df.prior_question_had_explanation + 2  # PAD and START
train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time + 2
train_df.lag_time_s = train_df.lag_time_s + 2
train_df.lag_time_m = train_df.lag_time_m + 2
train_df.lag_time_d = train_df.lag_time_d + 2

print("Partitioning dataset")
train_df = train_df.sort_values(["virtual_timestamp", "row_id"]).reset_index(drop=True)
ROW_NUM = len(train_df)
train_split = train_df[:-int(ROW_NUM * args.split)]
valid_split = train_df[-int(ROW_NUM * args.split):]
new_users = len(valid_split[~valid_split.user_id.isin(train_split.user_id)].user_id.unique())
valid_question = valid_split[valid_split.content_type_id == 0]
train_question = train_split[train_split.content_type_id == 0]
print(f"{train_question.answered_correctly.mean():.3f} {valid_question.answered_correctly.mean():.3f} {new_users}")

del train_df
gc.collect()

print("Grouping users")


def group_func(r):
    return (r.content_id.values,
            r.part.values,
            r.answered_correctly.values,
            r.prior_question_elapsed_time.values,
            r.lag_time_s.values,
            r.lag_time_m.values,
            r.lag_time_d.values,
            r.prior_question_had_explanation.values)


print(train_split)
print(valid_split)

train_part = train_split[["timestamp", "user_id", "content_id", "part", "answered_correctly",
                          "content_type_id", "prior_question_elapsed_time", "lag_time_s", "lag_time_m",
                          "lag_time_d", "prior_question_had_explanation"]].groupby("user_id").progress_apply(group_func)
valid_part = valid_split[["timestamp", "user_id", "content_id", "part", "answered_correctly",
                          "content_type_id", "prior_question_elapsed_time", "lag_time_s", "lag_time_m",
                          "lag_time_d", "prior_question_had_explanation"]].groupby("user_id").progress_apply(group_func)
print(train_part.shape)
print(valid_part.shape)

# if SAVE_DATA_TO_CACHE:
train_part.to_pickle(f"{args.output}.train")
valid_part.to_pickle(f"{args.output}.valid")

if args.irt:
    from pyirt import irt
    import pickle

    print("Start to use IRT model to estimate parameters")
    irt_src = []
    for user_id, (e_id, _, answer, _, _, _, _, _) in train_part.items():
        for item_id, ans in zip(e_id, answer):
            irt_src.append((user_id, item_id, ans - 2))

    item_param, user_param = irt(irt_src, theta_bnds=[-3, 3], max_iter=100)

    f = open(f"{args.output}.user", 'wb')
    pickle.dump(user_param, f)
    f.close()

    f = open(f"{args.output}.item", 'wb')
    pickle.dump(item_param, f)
    f.close()
