import argparse
import pandas as pd
import numpy as np
np.random.seed(1001)
import random
random.seed(1001)
from scipy import sparse
from tqdm import tqdm
import implicit
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import config


#Taken from faizan's repository
def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    actual = set(actual)
    predicted = list(predicted)

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in set(predicted[:i]):
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    outs = [apk(a, p, k) for a, p in zip(actual, predicted)]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]), outs


def fit_implicit_mf(data, K=10, K1=11.8, alpha=5, l2=1e-6):
    """
    :param data: dataframe for user productid data
    :param K: BM25 paramter
    :param K1: BM25 parameter
    :param alpha: hyperparameter for MF
    :param l2: B in BM25 model
    :return:(fitted model, item_user sparse matrix, user item sparse matrix)
    """
    data = data.copy()
    data['time_diff'] = (data["OrderDate"].max() - data["OrderDate"]).dt.days + 1
    data['time_diff'] = 1/data['time_diff']
    data['time_diff'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['time_diff'].values.reshape(-1,1))
    data['time_diff_prd'] = data.pid.map(data.groupby('pid')['time_diff'].max())
    data['qnt_adj'] = data['time_diff'] * data['time_diff_prd'] * data["Quantity"]
    data = data.groupby(['uid', 'pid'])['qnt_adj'].sum().reset_index()
    item_user = sparse.csr_matrix((data['qnt_adj'].values, (data['pid'], data['uid'])))
    user_item = sparse.csr_matrix((data['qnt_adj'].values, (data['uid'], data['pid'])))
    model = implicit.nearest_neighbours.BM25Recommender(K=K, K1=K1, B=l2)
    model.fit(item_user*alpha)
    return model, item_user, user_item


def read_data(mode='val'):
    """
    Read data
    :param mode: 'val' for validation mode else loads train/test
    :return: train/tr and test/val
    """
    if mode =="val":
        train = pd.read_csv(str(Path(config.output_dir) / 'tr.csv'), parse_dates=["OrderDate"])
        test = pd.read_csv(str(Path(config.output_dir) / 'val.csv'))
    else:
        train = pd.read_csv(config.train_data)
        train["OrderDate"] = pd.to_datetime(train["OrderDate"], format="%d/%m/%y")
        test = pd.read_csv(config.test_data)
    return train, test


def recommend_user(user, models, user_item):
    prods = []
    scores = []
    score_dict = defaultdict(int)
    count_dict = defaultdict(int)
    for j, model in enumerate(models):
        items = model.recommend(user, user_item, N=15, filter_already_liked_items=False)
        for i, (prd, score) in enumerate(items):
            score_dict[prd] += score
            count_dict[prd] += 1
    for prd, score in score_dict.items():
        cnt = count_dict[prd]
        prods.append(prd)
        scores.append(score/cnt)

    ind = np.argsort(scores)[::-1]
    prods = np.array(prods)[ind][:10]
    scores = np.array(scores)[ind][:10]
    return prods, scores, user


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='val', required=True)
    parser.add_argument("--subfile", default="submission.csv")
    parser.add_argument("--K", default=1, type=float)
    parser.add_argument("--K1", default=1, type=float)
    parser.add_argument("--l2", default=0.02, type=float)
    parser.add_argument("--alpha", default=5, type=float)
    args = parser.parse_args()

    MODE = args.mode
    out_path = str(Path(config.output_dir) / args.subfile)

    train, test = read_data(MODE)
    pid_dict = {p: i for i, p in enumerate(train.productid.unique())}
    uid_dict = {u: i for i, u in enumerate(train.UserId.unique())}
    pid_rev_dict = {i: p for p, i in pid_dict.items()}
    uid_rev_dict = {i: u for u, i in uid_dict.items()}
    
    train['uid'] = train.UserId.map(uid_dict).astype(int)
    test['uid'] = test.UserId.map(uid_dict).astype(int)

    train['pid'] = train.productid.map(pid_dict).astype(int)
    print(train.head())
    print(test.head())
    models = []
    for i in range(1):
        model, item_user, user_item = fit_implicit_mf(train, alpha=args.alpha, l2=args.l2, K=args.K, K1=args.K1)

        models.append(model)

    users = test["uid"].unique()
    results = [recommend_user(user, models, user_item) for user in tqdm(users)]
    results = pd.DataFrame(results, columns=["pid_list", "scores", "uid"])

    if MODE == "val":
        results["UserId"] = results.uid.map(uid_rev_dict).astype(int)
        test['pid'] = test.productid.map(pid_dict).fillna(-1).astype(int)
        actuals = test.groupby('uid')['pid'].apply(list).reset_index()
        print(actuals.shape, results.shape)
        results = pd.merge(actuals, results, on='uid', how='left')
        score, _ = mapk(results.pid.values, results.pid_list.values, k=10)
        print("Validation score is ", score)
    else:
        results["UserId"] = results.uid.map(uid_rev_dict).astype(int)
        results["product_list"] = results.pid_list.apply(lambda x: [int(pid_rev_dict[pid]) for pid in x])
        sub = results[["UserId", 'product_list']]
        sub.to_csv(out_path, index=False)
