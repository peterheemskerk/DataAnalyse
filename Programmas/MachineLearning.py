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


def poly_fit(df, xatt, yatt, dim):
    """Returns the polynomial coeficcients of the line best fitted to the given
    attributes.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    xatt, yatt = names of the attributes you want to fit.\n
    dim = dimension of the polynomial you want to fit."""

    values = df.loc[:, [xatt, yatt]].dropna().values
    xatt_arr, yatt_arr = values[:, 0], values[:, 1]
    return np.polyfit(xatt_arr, yatt_arr, dim)


def try_poly_fit(df, xatt, yatt, max_dim=10):
    """Fits xatt and yatt to eachother for each polynomial up to the max_dim..\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    xatt, yatt = names of the attributes you want to fit.\n
    max_dim = maximal dimension of the polynomial you want to fit."""

    values = df.loc[:, [xatt, yatt]].dropna().values
    xatt_arr, yatt_arr = values[:, 0], values[:, 1]
    xatt_mean, yatt_mean = mean_over_att(xatt_arr, yatt_arr)

    for i in range(1, max_dim + 1):
        poly = np.poly1d(np.polyfit(xatt_arr, yatt_arr, i))
        sq_e = np.mean((yatt_arr - poly(xatt_arr)) ** 2)
        print("Mean SQ Error =", sq_e)

        plt.plot(xatt_mean, yatt_mean, "bo")
        plt.plot(xatt_mean, poly(xatt_mean), "r--")
        plt.xlabel(KNMI.attributes[xatt])
        plt.ylabel(KNMI.attributes[yatt])
        plt.show()


def main():
    reduced_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + "_reduced.csv"
    df = pd.read_csv(reduced_filename)

    try_poly_fit(df, "DDVEC", "FHVEC")


main()
