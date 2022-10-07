# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.stats import beta

# the prior data
prior_data = "data/priordata.csv"
df = pd.read_csv(prior_data)
columns = ["sensor", "code", "pcs_m", "quantity"]


def g79_g81_to_gfrag_gfoam(x):
    """Converts codes G79 and G81 to Gfrag and Gfoam

    The measuring and counting of different sized plastic fragments
    is not generalized. For inference purposes all fragmented plastics
    and foams are put into one fo the two groups

    Args:
        x (str): The code that needs to be translated
    Returns:
        The key/value pair of x
    """

    these_codes = [
        "G78",
        "G79",
        "G80",
        "G75",
        "G76",
        "G77",
        "G81",
        "G82",
        "G83",
    ]

    if x in these_codes:
        new_code = dict(
            G81="Gfoam",
            G82="Gfoam",
            G83="Gfoam",
            G75="Gfrags",
            G76="Gfrags",
            G77="Gfrags",
            G78="Gfrags",
            G79="Gfrags",
            G80="Gfrags"
        )
        return new_code[x]
    else:
        return x


def choose_location(df: pd.DataFrame = df, location: str = "quai-maria-belgia") -> pd.DataFrame:
    mask = df["location"] == location
    return df[mask]


def choose_code(df: pd.DataFrame = None, code: str = "G27"):
    mask = df["code"] == code
    return df[mask]


def choose_columns(df: pd.DataFrame = df, columns: [] = columns) -> pd.DataFrame:
    return df[columns]


def choose_a_column(df: pd.DataFrame, column: str = None) -> pd.Series:
    return df[column]


def make_a_y(df: pd.DataFrame = df, location: str = "quai-maria-belgia", code: str = "G27",
        columns: [] = columns, column: str = "quantity"):

    a = choose_location(df, location)
    a = choose_code(a, code)
    a = choose_columns(a, columns)
    y = choose_a_column(a, column)

    return y


def count_instances(observed: pd.Series = None) -> []:

    counts = []
    t = observed.values
    count_limit = np.arange(0, np.max(t), step=1)

    for a_result in t:
        # each value in observed is a sample
        # the maximum value in observed is
        # the limit of the count.
        a_q = []
        for j in count_limit:
            i_s_grtr = int(a_result > j)
            a_q.append(i_s_grtr)
        counts.append(np.array(a_q))
    return sum(counts)


def make_y(df: pd.DataFrame = None, codes: [] = None, column: str = None):

    result = []
    for code in codes:
        a = choose_code(df, code)
        b = choose_a_column(a, column)
        if b.sum() == 0:
            break
        else:
            c = count_instances(b)
            result.append((code, c, len(b)))
    return result


def make_parameters(df: pd.DataFrame = None, codes: [] = str, column: str = None, n: int = None, prior: () = (1, 1)) -> np.array:
    instances = make_y(df, codes, column)
    for a_tuple in instances:
        code, c, n = a_tuple
        params = {code:{i:np.array([x + prior[0], n + prior[1]]) for i, x in enumerate(c)}}
        yield params


def make_priors(df: pd.DataFrame = None, columns: [] = None, location: str = None, column: str = None, prior=(1, 1)):

    a = choose_columns(df, columns)
    b = choose_location(a, location)
    codes = b.code.unique()

    return make_parameters(b, codes, column, prior=prior)

# scaled prior
def minimize_alpha_beta(x, mu, var):
    """
    # get the mean from the lake posterior and set variance to a limit
    mu = dists_lk["lake"]["mean"]
    var = 0.01

    #    alpha beta minimized
    scaled = optimize.root(minimize_alpha_beta, [1, 1], args=(mu, var)).x

    # returns [4]: array([9.5079, 13.6821])
    :param x:
    :param mu:
    :param var:
    :return:
    """
    alpha, beta = x[0], x[1]

    # define the mean and variance in terms of the parameters
    # mean or mu
    m = alpha / (alpha + beta)

    # standard deviation
    v = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))

    return [m - mu, v - var]


def make_code_priors(df, column="quantity"):
    codes = df.code.unique()
    priors = {}
    counts = make_y(df=df, codes=codes, column="quantity")
    return counts


def combine_priors(df: pd.DataFrame = None, columns: [] = None, locations: [] = None, column: str = None, prior=(1, 1)):

    for loc in locations:
        params = make_priors(df, columns, loc, column, prior)
        a = [pd.DataFrame(x)for x in params]
        b = pd.concat(a, axis=1)
        d = b.fillna(0)
        d.name = loc

        yield d


def assemble_priors(df: pd.DataFrame = None, columns: [] = None, locations: [] = None, column: str = None, prior=(1, 1)):

    g = combine_priors(df, columns, locations, column, prior=prior)
    h = {x.name: x for x in g}
    return h


def add_instances_by_sequence(x1, x2):
    if len(x1) > len(x2):
        fr_om_here = x2
        to_here = x1
    else:
        fr_om_here = x1
        to_here = x2
    basket = np.empty((len(to_here),), dtype=object)
    index = 0
    for i, val in enumerate(fr_om_here):
        basket[i] = val + to_here[i]
        index += 1

    for i, j in enumerate(to_here[index:]):
        new_index = index + i
        basket[new_index] = np.array([j[0], j[1] + fr_om_here[0][1]], dtype=object)

    assert len(basket) == len(to_here), "wtf"
    return basket


def make_beta_table(data: [] = None, n: int = None, a_dist: callable = beta, prior: () = (1, 1)):
    return [a_dist(x + prior[0], n-x+prior[1]) for x in data]

def climb_the_table():
    pass

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian
