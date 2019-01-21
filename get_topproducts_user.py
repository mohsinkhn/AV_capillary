import argparse
import config
import pandas as pd
import numpy as np
np.random.seed(1001)
import random
random.seed(1001)
from scipy import sparse
from tqdm import tqdm
import implicit
from scipy.stats import gmean
from collections import defaultdict
from pathlib import Path

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


def get_conf(data, a=1, b=1, c=10):
    day_factor = (data["OrderDate"].max() - data["OrderDate"]).dt.days + 1
    f1= np.exp(-1*day_factor/max(day_factor))
    f2 = data["Quantity"]
    return a*f1 + b*f2 + c*f1*f2


def fit_implicit_mf(data, a=1, b=1, c=10, alpha=5, factors=96, l2=1e-6, iters=100, model='bayes'):
    #confs = get_conf(data, a, b, c)
    data = data.copy()
    data['time_diff'] = (data["OrderDate"].max() -data["OrderDate"]).dt.days
    data['time_diff'] = (1 - 0.0*data["time_diff"]/data["time_diff"].max())
    data['qnt_adj'] = data['Quantity']*data['time_diff'] 
    data = data.groupby(['uid', 'pid'])['qnt_adj'].sum().reset_index()
    item_user = sparse.csr_matrix((data['qnt_adj'].values, (data['pid'], data['uid'])))
    user_item = sparse.csr_matrix((data['qnt_adj'].values, (data['uid'], data['pid'])))
    if model == 'als':
        model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 regularization=l2,
                                                 iterations=iters,
                                                 calculate_training_loss=True)
    else:
        model = implicit.bpr.BayesianPersonalizedRanking(factors=factors, regularization=l2,
                                                         iterations=iters, use_gpu=True)
    model.fit(item_user*alpha)
    return model, item_user, user_item


def read_data(mode='val'):
    if mode =="val":
        train = pd.read_csv(config.tr_data, parse_dates=["OrderDate"])
        test = pd.read_csv(config.val_data)
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
    for model in models:
        items = model.recommend(user, user_item, N=14, filter_already_liked_items=False)
        for prd, score in items:
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
    parser.add_argument("--subfile", default="tmp.csv")
    parser.add_argument("--a", default=1, type=int)
    parser.add_argument("--b", default=1, type=int)
    parser.add_argument("--c", default=10, type=int)
    parser.add_argument("--factors", default=63, type=int)
    parser.add_argument("--l2", default=0.01, type=float)
    parser.add_argument("--iter", default=1000, type=int)
    parser.add_argument("--model", default='bayes', type=str)
    parser.add_argument("--alpha", default=10, type=float)
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

    train['pid'] = train.productid.map(pid_dict)
    print(train.head())
    print(test.head())

    # Run 5 times and average
    #user_factors = []
    #item_factors = []
    models = []
    for i in range(10):
        model, item_user, user_item = fit_implicit_mf(train, alpha=args.alpha, a=args.a, b=args.b, c=args.c, factors=args.factors,
                                                      l2=args.l2, iters=args.iter, model=args.model)
        #user_factors.append(model.user_factors)
        #item_factors.append(model.item_factors)
        models.append(model)
    #user_factors = np.mean(user_factors, axis=0)
    #item_factors = np.mean(item_factors, axis=0)
    #model.user_factors = user_factors
    #model.item_factors = item_factors
    print(model.user_factors.shape, model.item_factors.shape)
    users = test["uid"].unique()
    results = [recommend_user(user, models, user_item) for user in tqdm(users)]

    results = pd.DataFrame(results, columns=["pid_list", "scores", "uid"])
    if MODE == "val":
        test['pid'] = test.productid.map(pid_dict).fillna(-1).astype(int)
        actuals = test.groupby('uid')['pid'].apply(list).reset_index()
        print(actuals.shape, results.shape)
        results = pd.merge(actuals, results, on='uid', how='left')
        score, _ = mapk(results.pid.values, results.pid_list.values, k=10)
        print("Validation score is ", score)
    else:
        results["product_list"] = results.pid_list.apply(lambda x: [int(pid_rev_dict[pid]) for pid in x])
        results["UserId"] = results.uid.map(uid_rev_dict).astype(int)
        sub = results[["UserId", 'product_list']]
        sub.to_csv(out_path, index=False)
