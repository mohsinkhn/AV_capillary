import pandas as pd
from pathlib import Path

import config


if __name__ == "__main__":
    VAL_SPLIT_DATE = '2018-08-30' # Pick last one months for validation
    train = pd.read_csv(config.train_data)
    test = pd.read_csv(config.test_data)
    train["OrderDate"] = pd.to_datetime(train["OrderDate"], format="%d/%m/%y")
    print("Shape of train and test data", train.shape, test.shape)
    print(train.head())

    tr = train.loc[train.OrderDate <= pd.to_datetime(VAL_SPLIT_DATE)]
    val = train.loc[train.OrderDate > pd.to_datetime(VAL_SPLIT_DATE)]

    tr_users = set(tr.UserId.values)
    val = val.loc[val.UserId.isin(tr_users)]
    print("tr and val shapes are ", tr.shape, val.shape)

    # save data
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    tr.to_csv(str(Path(config.output_dir) / 'tr.csv'), index=False)
    val.to_csv(str(Path(config.output_dir) / 'val.csv'), index=False)
