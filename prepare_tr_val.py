import pandas as pd
import config

if __name__ == "__main__":
    train = pd.read_csv(config.train_data)
    test = pd.read_csv(config.test_data)
    train["OrderDate"] = pd.to_datetime(train["OrderDate"], format="%d/%m/%y")
    print("Shape of train and test data", train.shape, test.shape)
    print(train.head())

    # Pick last one months for validation
    tr = train.loc[train.OrderDate <= pd.to_datetime('2018-08-30')]
    val = train.loc[train.OrderDate > pd.to_datetime('2018-08-30')]

    tr_users = set(tr.UserId.values)
    val = val.loc[val.UserId.isin(tr_users)]
    print("tr and val shapes are ", tr.shape, val.shape)

    # save data
    tr.to_csv(config.tr_data, index=False)
    val.to_csv(config.val_data, index=False)
