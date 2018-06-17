import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import KNMI


def mean_over_att(xatt_arr, yatt_arr):
    """Returns the average value of xatt_arr for each value of yatt_arr.\n
    xatt_arr, yatt_arr = array of numbers."""

    xatt_mean = []
    yatt_mean = []

    for xatt in set(xatt_arr):
        yatt = yatt_arr[xatt_arr == xatt]

        if len(yatt) > 1:
            xatt_mean.append(xatt)
            yatt_mean.append(np.mean(yatt))

    return np.array(xatt_mean), np.array(yatt_mean)


def plot_poly(df, poly, xatt, yatt):
    """Plot the fit of xatt and yatt.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    poly = the fitted polygon.\n
    xatt, yatt = names of the attributes you want to fit.\n"""

    values = df.loc[:, [xatt, yatt]].dropna().values
    xatt_arr, yatt_arr = values[:, 0], values[:, 1]
    xatt_mean, yatt_mean = mean_over_att(xatt_arr, yatt_arr)

    sq_e = np.mean((yatt_arr - poly(xatt_arr)) ** 2)
    print("Mean squared error:", sq_e)

    plt.plot(xatt_mean, yatt_mean, "bo")
    plt.plot(xatt_mean, poly(xatt_mean), "r--")
    plt.xlabel(KNMI.attributes[xatt])
    plt.ylabel(KNMI.attributes[yatt])
    plt.show()


def poly_fit(df, xatt, yatt, dim):
    """Returns the polynomial coeficcients of the line best fitted to the given
    attributes.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    xatt, yatt = names of the attributes you want to fit.\n
    dim = dimension of the polynomial you want to fit."""

    values = df.loc[:, [xatt, yatt]].dropna().values
    xatt_arr, yatt_arr = values[:, 0], values[:, 1]
    return np.polyfit(xatt_arr, yatt_arr, dim)


def seperate_trn_dev_tst(df, dev_perc=0.1, tst_perc=0.1):
    """Splits the dataframe in a train, developer and test set.\n
    df = pandas.Dataframe, all the data extracted from your csv file.\n
    dev_perc, tst_perc = the relative size of your developer and test set.\n"""

    tst_part = int((1 - tst_perc) * len(df))
    dev_part = int((1 - tst_perc - dev_perc) * len(df))

    return np.split(df.sample(frac=1), [dev_part, tst_part])


def try_poly_fit(trn, dev, xatt, yatt, max_dim=10, prec=0.1):
    """Fits xatt and yatt to eachother for each polynomial up to the max_dim.
    And returns the best fit before overfitting.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    xatt, yatt = names of the attributes you want to fit.\n
    max_dim = maximal dimension of the polynomial you want to fit.
    prec = The minimal improvement a new dimension has to make."""

    values = trn.loc[:, [xatt, yatt]].dropna().values
    xatt_trn, yatt_trn = values[:, 0], values[:, 1]

    values = dev.loc[:, [xatt, yatt]].dropna().values
    xatt_dev, yatt_dev = values[:, 0], values[:, 1]

    past_poly = None
    past_err = np.inf
    for i in range(1, max_dim + 1):
        poly = np.poly1d(np.polyfit(xatt_trn, yatt_trn, i))
        sq_e = np.mean((yatt_dev - poly(xatt_dev)) ** 2)
        print("Mean squared error of dimension", i, "is", sq_e)

        if sq_e + prec > past_err:
            print("Overfitting is reached.\nBest dimension is", i - 1)
            return past_poly
        else:
            past_poly = poly
            past_err = sq_e

    print("Maximal dimension is reached.\nDimension is", max_dim)
    return past_poly


def main():
    reduced_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + "_reduced.csv"
    df = pd.read_csv(reduced_filename)

    trn, dev, tst = seperate_trn_dev_tst(df)
    poly = try_poly_fit(trn, dev, "DDVEC", "FHVEC")
    plot_poly(tst, poly, "DDVEC", "FHVEC")


main()
