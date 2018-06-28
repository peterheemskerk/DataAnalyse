# This program contains three majos machine learning methods:
# 1 - Multi Regression: predict the value of an attribute given other attributes
#                       of the datapoint.
# 2 - Classification:   predict in what season a few datapoints lie.
# 3 - Regression:       draw a line that shows the correlation between two
#                       attributes.


from numpy.linalg import lstsq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import KNMI


class Lq_Fit:

    @staticmethod
    def seperate_trn_dev_tst(df, dev_perc=0.1, tst_perc=0.1):
        """Splits the dataframe in a train, developer and test set.\n
        df = pandas.Dataframe, all the data extracted from your csv file.\n
        dev_perc, tst_perc = the relative size of your developer and test set.\n
        """

        tst_part = int((1 - tst_perc) * len(df))
        dev_part = int((1 - tst_perc - dev_perc) * len(df))

        return np.split(df.sample(frac=1), [dev_part, tst_part])

    def __init__(self, df, illegal=np.array([])):
        self.atts = np.array([])
        self.orders = np.array([], dtype=int)

        self.best_atts = np.array([])
        self.best_orders = np.array([], dtype=int)
        self.best_error = 0

        self.trn, self.dev, self.tst = Lq_Fit.seperate_trn_dev_tst(df)

        self.illegal = illegal
        self.all_atts = np.setdiff1d(df.columns.values[3:], illegal)

        self.p_all = np.ones(len(self.all_atts)) / len(self.all_atts)
        self.p_order = np.array([])
        self.p_del = np.array([])

    def make_X(self, atts=[], orders=[], trn=[], dev=[]):
        """Make a 'X' matrix containing all data from the attributes in the key.
        """

        if len(atts) < 1:
            atts, orders = self.atts, self.orders

        if len(trn) < 1:
            trn, dev = self.trn, self.dev

        X_trn = trn.loc[:, atts].values
        X_dev = dev.loc[:, atts].values

        X_trn = np.power(X_trn, orders)
        X_dev = np.power(X_dev, orders)

        X_trn = np.append(X_trn, np.ones([len(trn), 1]), 1)
        X_dev = np.append(X_dev, np.ones([len(dev), 1]), 1)
        return X_trn, X_dev

    def mutate(self):
        """One mutating can be:
        adding an attribute from the allowed attributes.
        adding a higher order of an existing attribute in the key.
        deleting an attribute from the key."""

        r = np.random.random_sample() * (len(self.p_all[self.p_all != 0])
                                         + len(self.p_order[self.p_order != 0])
                                         + len(self.p_del[self.p_del != 0]))

        if r < len(self.p_all[self.p_all != 0]):
            att = np.random.choice(self.all_atts, p=self.p_all)
            self.atts = np.append(self.best_atts, att)
            self.orders = np.append(self.best_orders, 1)

            self.p_all[self.all_atts == att] = 0
            if np.sum(self.p_all) > 0:
                self.p_all /= np.sum(self.p_all)

        elif r < (len(self.p_all[self.p_all != 0])
                  + len(self.p_order[self.p_order != 0])):

            att = np.random.choice(self.best_atts, p=self.p_order)
            biggest_order = np.amax(self.best_orders[self.best_atts == att])

            self.atts = np.append(self.best_atts, att)
            self.orders = np.append(self.best_orders, biggest_order + 1)

            return str(att)

        else:
            index = np.random.choice(len(self.best_atts), p=self.p_del)
            self.atts = np.delete(self.best_atts, index)
            self.orders = np.delete(self.best_orders, index)

            return index

    def reset(self):
        """Reset the object when training over new data."""

        self.atts = np.array([])
        self.orders = np.array([], dtype=int)

        self.best_atts = np.array([])
        self.best_orders = np.array([], dtype=int)
        self.best_error = 0

        self.all_atts = np.setdiff1d(self.trn.columns.values[3:], self.illegal)

        self.p_all = np.ones(len(self.all_atts)) / len(self.all_atts)
        self.p_order = np.array([])
        self.p_del = np.array([])

    def reset_p(self, fit):
        self.p_all = np.ones(len(self.all_atts))
        self.p_all[np.in1d(self.all_atts, self.atts)] = 0
        self.p_all /= np.sum(self.p_all)

        self.p_order = np.absolute(fit) / np.sum(np.absolute(fit))
        self.p_del = np.absolute(1 / fit) / np.sum(np.absolute(1 / fit))

    def season_pred(self, start_date, end_date, atts=[], orders=[]):
        """Predict what season the data is when given a start and end-date."""

        if len(atts) < 1:
            atts, orders = self.best_atts, self.best_orders

        df = self.trn
        df.append(self.dev)
        df.append(self.tst)
        df = df.loc[:, "STN":]

        trn = df.loc[(df.YYYYMMDD < start_date) | (df.YYYYMMDD > end_date)]
        tst = df.loc[(df.YYYYMMDD >= start_date) & (df.YYYYMMDD <= end_date)]

        y_trn = get_seasons(trn)
        X_trn, X_tst = self.make_X(atts, orders, trn, tst)

        pred = np.empty([X_tst.shape[0], 4])
        for season in range(4):
            season_trn = y_trn == season
            fit = lstsq(X_trn, season_trn, rcond=None)[0]
            pred[:, season] = fit.dot(X_tst.T)

        pred = np.sum(pred, axis=0)
        return pred.argmax()

    @staticmethod
    def season_fit(X_trn, X_dev, y_trn, y_dev):
        """Fit the X and y data and return the error. Specialized for seasons.
        """

        p = np.zeros(X_trn.shape[1])
        pred = np.empty([X_dev.shape[0], 4])

        for season in range(4):
            season_trn = y_trn == season

            fit = lstsq(X_trn, season_trn, rcond=None)[0]
            p += np.absolute(fit)

            pred[:, season] = fit.dot(X_dev.T)

        err = np.mean(pred.argmax(axis=1) != y_dev)

        return err, p

    def try_regr_fit(self, y_att, prec=0.01):
        """Hillclimb through the allowed attributes to find a which attributes
        are most usefull when predicting the y_att."""

        print("Start regression for:", y_att)
        self.reset()

        y_trn = self.trn.loc[:, y_att].values
        y_dev = self.dev.loc[:, y_att].values

        self.p_all = self.p_all[self.all_atts != y_att]
        self.p_all /= np.sum(self.p_all)
        self.all_atts = self.all_atts[self.all_atts != y_att]

        bench = np.mean((y_dev - np.ones(len(y_dev)) * np.mean(y_trn)) ** 2)
        print("The benchmark is", bench)

        best_err = bench
        while (np.sum(self.p_all) > 0 or np.sum(self.p_order) > 0
                                      or np.sum(self.p_del) > 0):

            mut_elem = self.mutate()
            X_trn, X_dev = self.make_X()

            fit = lstsq(X_trn, y_trn, rcond=None)[0]
            pred = fit.dot(X_dev.T)
            err = np.mean((y_dev - pred) ** 2)

            if (best_err * ((1 + prec) ** len(self.best_atts))
                > err * ((1 + prec) ** len(self.atts))):

                self.best_atts, self.best_orders = self.atts, self.orders,
                best_err = err
                self.reset_p(fit[:-1])

                print("Improvement upon benchmark =",
                      str(int((1 - err / bench) * 100)) + "%, new model:",
                      self.show_best())

            elif type(mut_elem) == str:
                self.p_order[self.best_atts == mut_elem] = 0
                if np.sum(self.p_order):
                    self.p_order /= np.sum(self.p_order)

            elif type(mut_elem) == int:
                self.p_del[mut_elem] = 0
                if np.sum(self.p_del):
                    self.p_del /= np.sum(self.p_del)

        print("Local maximum is reached. The mean squared error is:",
              self.test(y_att), "the best solution is:", self.show_best(), '\n')

        return self.best_atts, self.best_orders

    def try_season_fit(self, prec=0.01):
        """Hillclimb through the allowed attributes to find a which attributes
        are most usefull when predicting the season."""

        print("Start fitting seasons")
        self.reset()

        y_trn = get_seasons(self.trn)
        y_dev = get_seasons(self.dev)

        bench = 0.75
        print("The benchmark is", str(int((1 - bench) * 100))
              + "% of dates being guessed right")

        best_err = bench
        while (np.sum(self.p_all) > 0 or np.sum(self.p_order) > 0
                                      or np.sum(self.p_del) > 0):

            mut_elem = self.mutate()
            X_trn, X_dev = self.make_X()
            err, p = Lq_Fit.season_fit(X_trn, X_dev, y_trn, y_dev)

            if (best_err * ((1 + prec) ** len(self.best_atts))
                > err * ((1 + prec) ** len(self.atts))):

                self.best_atts, self.best_orders = self.atts, self.orders,
                best_err = err
                self.reset_p(p[:-1])

                print("Now", str(int((1 - err) * 100)) +
                      "% of dates is guessed right, new model =",
                      self.show_best())

            elif type(mut_elem) == str:
                self.p_order[self.best_atts == mut_elem] = 0
                if np.sum(self.p_order):
                    self.p_order /= np.sum(self.p_order)

            elif type(mut_elem) == int:
                self.p_del[mut_elem] = 0
                if np.sum(self.p_del):
                    self.p_del /= np.sum(self.p_del)

        print("Local maximum is reached. The mean error is:",
              self.test_season(), "the best solution is:", self.show_best(),
              '\n')

        return self.best_atts, self.best_orders

    def show_best(self):
        """Show which attributes are being used with what order in the current
        fit."""

        show = []
        for i, att in enumerate(self.best_atts):
            show.append(str(att) + "^" + str(self.best_orders[i]))

        return np.array(show)

    def test(self, y_att):
        """Test the best found fit for the y attribute using the test set."""

        y_trn = self.trn.loc[:, y_att].values
        y_tst = self.tst.loc[:, y_att].values

        X_trn, X_tst = self.make_X(self.best_atts, self.best_orders, self.trn,
                                   self.tst)

        fit = lstsq(X_trn, y_trn, rcond=None)[0]
        pred = fit.dot(X_tst.T)

        self.error = np.mean((y_tst - pred) ** 2)
        return self.error

    def test_season(self):
        """The 'test' function specialized for the classificaton problem with
        the seasons."""

        y_trn = get_seasons(self.trn)
        y_tst = get_seasons(self.tst)

        X_trn, X_tst = self.make_X(self.best_atts, self.best_orders, self.trn,
                                   self.tst)

        self.error, _ = Lq_Fit.season_fit(X_trn, X_tst, y_trn, y_tst)
        return self.error


def get_seasons(df):
    """Return a dataframe with one column which has all dates converted to
    seasons. The seasons are categorized by numbers: 0 is spring and so forth.
    df = pandas.Dataframe"""

    dates = df.loc[:, "YYYYMMDD"].values % 10000
    seasons = np.empty(len(dates), dtype=int)

    for i, season in enumerate(KNMI.SEASONS.keys()):
        start, end = KNMI.SEASONS[season]

        mask = False
        if season == "winter":
            mask = (dates >= start) | (dates <= end)

        else:
            mask = (dates >= start) & (dates <= end)

        seasons[mask] = i

    return seasons


def mean_over_att(xatt_arr, yatt_arr):
    """Returns the average value of xatt_arr for each value of yatt_arr.\n
    xatt_arr, yatt_arr = array of numbers."""

    xatt_mean = []
    yatt_mean = []

    for xatt in np.sort(xatt_arr):
        if np.isnan(xatt) or xatt in xatt_mean:
            continue
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

    if xatt in KNMI.attributes:
        plt.xlabel(KNMI.attributes[xatt])
    if yatt in KNMI.attributes:
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


def try_poly_fit(trn, dev, xatt, yatt, max_dips=3, prec=0.001):
    """Fits xatt and yatt to eachother for each polynomial up to the max_dim.
    And returns the best fit before overfitting.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    xatt, yatt = names of the attributes you want to fit.\n
    max_dips = Amount of times a new fit may have a higher squared error.
    prec = Percentage of improvement a new fit must make."""

    values = trn.loc[:, [xatt, yatt]].dropna().values
    xatt_trn, yatt_trn = values[:, 0], values[:, 1]

    values = dev.loc[:, [xatt, yatt]].dropna().values
    xatt_dev, yatt_dev = values[:, 0], values[:, 1]

    best_dim = 0
    lowest_err = np.inf

    for dim in range(1, 24):
        poly = np.poly1d(np.polyfit(xatt_trn, yatt_trn, dim))
        sq_e = np.mean((yatt_dev - poly(xatt_dev)) ** 2)
        print("Mean squared error of dimension", dim, "is", sq_e)

        if sq_e < lowest_err * (1 - prec) ** (dim - best_dim):
            lowest_err = sq_e
            best_dim = dim

        elif dim - best_dim >= max_dips:
            break

    print("Best dimension is", best_dim)
    return np.poly1d(np.polyfit(xatt_trn, yatt_trn, best_dim))


def main():
    filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + "_ml.csv"
    df = pd.read_csv(filename)


    # 1 ---- Multi-Regression -----

    # Which attributes may not be used when fitting.
    illegal = np.array(["TX", "TN", "TXH", "TNH"])
    # Make an object of the 'Lq_Fit' class.
    lq1 = Lq_Fit(df, illegal=illegal)
    # Find which attributes are most usefull when predicting 'TG'
    atts, orders = lq1.try_regr_fit("TG")
    # Show the test results of the found attributes
    lq1.test("TG")


    # 2 ---- Classification -----

    # Leaving 'illegal' empty will allow all attributes to be used.
    lq2 = Lq_Fit(df)
    # A higher 'prec' will favour less attributes to be used and vice versa.
    # The try_season_fit works almost the same as the try_regr_fit but is
    # specialised for predicting seasons.
    atts, orders = lq2.try_season_fit()
    # Show a graph in which is shown which months are predicted as what season.
    dates = []
    seasons = []
    for year in range(2000, 2005):
        for month in range(1, 13):
            start = year * 10000 + month * 100
            end = year * 10000 + month * 100 + 31
            dates.append(str(year) + "-" + str(month))

            season = lq2.season_pred(start, end, atts, orders)
            print(str(year) + '/' + str(month) + ':', season)

            seasons.append(season + 1)

    ticks = ["", "lente", "zomer", "herfst", "winter"]
    plt.bar(dates, seasons)
    plt.yticks(range(5), ticks)
    plt.xticks(rotation=-80)
    plt.xlabel("months")
    plt.suptitle("seasons as predicted by the classifier")
    plt.show()


    # 3 ---- Regression -----

    # Split your extracted data into a train, developer and test set.
    trn, dev, tst = Lq_Fit.seperate_trn_dev_tst(df)
    # Return the coefficients of the polygon that fits 'DDVEC_SIN' and 'FHVEC'
    poly = try_poly_fit(trn, dev, "DDVEC_SIN", "FHVEC")
    # Plot the found coefficients in between the data of 'DDVEC_SIN' and 'FHVEC'
    plot_poly(tst, poly, "DDVEC_SIN", "FHVEC")


if __name__ == "__main__":
    main()
