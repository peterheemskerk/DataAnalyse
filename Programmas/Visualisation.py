# This program can plot given attributes of given station over all the time
# that the station has made data.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import KNMI
import Univariate as uni
import MachineLearning as ml
import PlotBokeh as bok

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

MARKERS = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
           'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks',
           'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',
           'bd', 'gd', 'rd', 'cd', 'md', 'yd', 'kd']

SEASONS = {"spring": (321, 620), "summer": (621, 920), "autumn": (921, 1220),
           "winter": (1221, 320)}

MEAN_ATTS = ["DDVEC", "FG", "TG", "Q", "PG", "UG"]

vget_t = np.vectorize(KNMI.get_t)


def attribute_over_time(df, stn, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the time.\n
    df = pandas.DataFrame, all the data extracted from your csv file\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    stn = KNMI.stn[stn].num
    data = df.loc[(df.STN == stn) & (df.YYYYMMDD >= start)
                  & (df.YYYYMMDD <= end), ["YYYYMMDD", att]].values
    att_arr = data[:, 1]
    date_arr = data[:, 0]

    mask = att_arr != np.nan
    return att_arr[mask], date_arr[mask]


def attributes_over_time(df, stn, atts, start=19010101, end=20991231):
    """Returns a tuple of the values of all given attributes and the time.\n
    df = pandas.DataFrame, all the data extracted from your csv file\n
    stn = station, the number or the name of the station.\n
    xatt = first attribute\n
    start = start date\n
    end = end date"""

    stn = KNMI.stn[stn].num
    atts = ["YYYYMMDD"] + list(atts)

    data = df.loc[(df.STN == stn) & (df.YYYYMMDD >= start)
                  & (df.YYYYMMDD <= end), atts].values
    att_arr = data[:, 1:]
    date_arr = data[:, 0]

    mask = ~np.isnan(np.sum(att_arr, axis=1))
    return att_arr[mask], date_arr[mask]


def get_attribute_all_stn(df, att, start=19010101, end=20991231):
    """Peter. Returns a tuple of the values of the given attribute and the time.\n
    df = pandas.DataFrame, all the data extracted from your csv file\n
    all stations.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    data = df.loc[(df.YYYYMMDD >= start)
                  & (df.YYYYMMDD <= end), ["YYYYMMDD", att]].values
    return data[:, 1], data[:, 0]


def avg_over_att_all_stn(df, xatt, yatt, start=19010101, end=20991231):
    """Peter. Returns a tuple of the values of the two given attributes.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    xatt, yatt = attribute names\n
    start = start date\n
    end = end date"""

    data = df.loc[(df.YYYYMMDD >= start)
                  & (df.YYYYMMDD <= end), [xatt, yatt]].values
    xatt_arr, yatt_arr = data[:, 0], data[:, 1]

    mask = ~np.isnan(yatt_arr)
    xatt_arr = xatt_arr[mask]
    yatt_arr = yatt_arr[mask]

    xatts = []
    yatts = []
    for xatt in xatt_arr:
        if np.isnan(xatt) or xatt in xatts:
            continue
        yatt = yatt_arr[xatt_arr == xatt]

        if len(yatt) > 0:
            xatts.append(xatt)
            yatts.append(np.mean(yatt))

    return np.array(xatts), np.array(yatts)


def avg_over_year_all_stn(df, att, start=19010101, end=20991231):
    """Peter. Returns a tuple of the values of the given attribute and the years.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    average over all stations.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = get_attribute_all_stn(df, att, start, end)

    mask = ~np.isnan(att_arr)
    att_arr = att_arr[mask]
    t_arr = t_arr[mask]

    years = []
    att_avg = []
    for year in range(start // 10000, end // 10000 + 1):
        att = att_arr[t_arr // 10000 == year]

        if len(att) > 0:
            years.append(year)
            att_avg.append(np.mean(att))

    return np.array(att_avg), np.array(years)


def avg_over_month_all_stn(df, att, start=19010101, end=20991231):
    """Peter. Returns a tuple of the values of the given attribute and the months.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    average over all stations.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = get_attribute_all_stn(df, att, start, end)

    mask = ~np.isnan(att_arr)
    att_arr = att_arr[mask]
    t_arr = t_arr[mask]

    months = []
    att_avg = []
    for month in range(1, 12):
        att = att_arr[(t_arr // 100) % 100 == month]

        if len(att) > 0:
            months.append(month)
            att_avg.append(np.mean(att))

    return np.array(att_avg), np.array(months)


def avg_over_att(df, stn, xatt, yatt, start=19010101, end=20991231):
    """Returns a tuple of the values of the two given attribute.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, _ = attributes_over_time(df, stn, [xatt, yatt], start, end)

    xatt_arr = att_arr[:, 0]
    xatt_arr = xatt_arr[xatt_arr != np.nan].astype(int)
    yatt_arr = att_arr[:, 1]
    yatt_arr = yatt_arr[yatt_arr != np.nan]

    xatts = []
    yatts = []
    for x_att in set(xatt_arr):
        y_att = yatt_arr[xatt_arr == x_att]

        if y_att.size:
            yatts.append(np.mean(y_att))
            xatts.append(x_att)

    return np.array(xatts), np.array(yatts)


def avg_over_month(df, stn, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the months.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = attribute_over_time(df, stn, att, start, end)

    months = []
    att_avg = []
    for month in range(1, 13):
        att = att_arr[t_arr % 10000 // 100 == month]

        if len(att) > 1:
            att_avg.append(np.mean(att))
            months.append(month)

    return np.array(att_avg), np.array(months)


def avg_over_season(df, stn, att, start=19010101, end=10991231):
    """Returns a tuple of the values of the given attribute and the seasons.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = attribute_over_time(df, stn, att, start, end)
    t_arr %= 10000

    seasons = []
    att_avg = []
    for season in np.array(["winter", "spring", "summer", "autumn"]):
        s_start, s_end = SEASONS[season]

        att_data = []
        if season == "winter":
            att_data = att_arr[(t_arr >= s_start) | (t_arr <= s_end)]
        else:
            att_data = att_arr[(s_start <= t_arr) & (t_arr <= s_end)]

        if len(att_data) > 1:
            att_avg.append(np.mean(att_data))
            seasons.append(season)

    return np.array(att_avg), np.array(seasons)


def avg_over_year(df, stn, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the years.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = attribute_over_time(df, stn, att, start, end)

    years = []
    att_avg = []
    for year in range(start // 10000, end // 10000 + 1):
        att = att_arr[(t_arr // 10000 == year)]

        if len(att) > 0:
            years.append(year)
            att_avg.append(np.mean(att))

    return np.array(att_avg), np.array(years)


def measured_months_stn(df, stn, att=None, start=19010101, end=20991231):
    """Return the number of entries from a station for each month.\n
    df = pandas.DataFrame, all data extracted from your csv file.\n
    stn = station.\n
    att = attribute\n
    start = start date\n
    end = end date\n"""

    dates = []
    if att:
        dates = df.loc[(df.STN == stn) & (df.YYYYMMDD >= start)
                       & (df.YYYYMMDD <= end), ["YYYYMMDD", att]].values
        att_arr = dates[:, 1]
        dates = dates[:, 0]
        dates = dates[~np.isnan(att_arr)]

    else:
        dates = df.loc[(df.STN == stn) & (df.YYYYMMDD >= start)
                       & (df.YYYYMMDD <= end), "YYYYMMDD"].values

    months = np.arange(1, 13)
    counts = np.zeros(12)
    for i, month in enumerate(months):
        counts[i] = len(dates[(dates // 100 % 100) == month])

    return counts, months


def measured_years_stn(df, stn, att=None, start=19010101, end=20991231):
    """Return the number of entries from a station for each year.\n
    df = pandas.DataFrame, all data extracted from your csv file.\n
    stn = station.\n
    att = attribute\n
    start = start date\n
    end = end date\n"""

    dates = []
    if att:
        dates = df.loc[(df.STN == stn) & (df.YYYYMMDD >= start)
                       & (df.YYYYMMDD <= end), ["YYYYMMDD", att]].values
        att_arr = dates[:, 1]
        dates = dates[:, 0]
        dates = dates[~np.isnan(att_arr)]

    else:
        dates = df.loc[(df.STN == stn) & (df.YYYYMMDD >= start)
                       & (df.YYYYMMDD <= end), "YYYYMMDD"].values

    years = np.arange(start // 10000, (end // 10000) + 1)
    counts = np.empty(len(years))
    for i, year in enumerate(years):
        counts[i] = len(dates[(dates // 10000) == year])

    return counts, years


def plot_att_conditional(df, stn_arr, xatt, yatt, start=19010101, end=20991231,
                         markers=MARKERS):
    """Plot the average value of an attribute over another attribute from
    multiple stations.\n
    df = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    if len(stn_arr) < 1:
        xatts, yatts = avg_over_att_all_stn(df, xatt, yatt, start, end)
        plt.plot(xatts, yatts, markers[0])

    else:
        for i, stn in enumerate(stn_arr):
            xatts, yatts = avg_over_att(df, stn, xatt, yatt, start, end)
            plt.plot(xatts, yatts, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel(KNMI.attributes[xatt])
    plt.ylabel(KNMI.attributes[yatt])

    if len(stn_arr) >= 1:
        plt.legend()

    plt.show()


def plot_att_month(df, stn_arr, att, start=19010101, end=20991231,
                   colors=COLORS):
    """Plot the average value of an attribute over each month from multiple
    stations.\n
    df = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    all_stn = len(stn_arr) < 1
    if all_stn:
        stn_arr = np.array(KNMI.Station.all_num)

    tot_stn = len(stn_arr) - 1
    for i, stn in enumerate(stn_arr):
        att_arr, months = avg_over_month(df, stn, att, start, end)

        if all_stn:
            plt.bar(months, att_arr, color=colors[0])

        else:
            months = months.astype(float) + 0.4 * (i / tot_stn) - 0.2
            width = 0.8 / (tot_stn + 1)
            plt.bar(months, att_arr, width=width, color=colors[i],
                    label=KNMI.stn[stn].name)

    plt.xlabel("Maanden")
    plt.ylabel(KNMI.attributes[att])

    if not all_stn:
        plt.legend()

    plt.show()


def plot_att_season(df, stn_arr, att, start=19010101, end=20991231,
                    markers=MARKERS):
    """Plot the average value of an attribute over each season from multiple
    stations.
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    if len(stn_arr) < 1:
        att_arr, seasons = avg_over_month_all_stn(df, att, start, end)
        plt.plot(seasons, att_arr, markers[0])

    else:
        for i, stn in enumerate(stn_arr):
            att_arr, seasons = avg_over_season(df, stn, att, start, end)
            plt.plot(seasons, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Seizoenen")
    plt.ylabel(KNMI.attributes[att])

    if len(stn_arr) >= 1:
        plt.legend()

    plt.show()


def plot_att_year(df, stn_arr, att, start=19010101, end=20991231,
                  markers=MARKERS):
    """Plot the average value of an attribute over a year from multiple stations
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    if len(stn_arr) < 1:
        att_arr, years = avg_over_year_all_stn(df, att, start, end)
        plt.plot(years, att_arr, markers[0])

    else:

        for i, stn in enumerate(stn_arr):
            att_arr, years = avg_over_year(df, stn, att, start, end)
            plt.plot(years, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Jaren")
    plt.ylabel(KNMI.attributes[att])

    if len(stn_arr) >= 1:
        plt.legend()

    plt.show()

def plot_att_year_bok(df, stn_arr, att, start=19010101, end=20991231,
                  markers=MARKERS):
    """Plot the average value of an attribute over a year from multiple stations
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    if len(stn_arr) < 1:
        att_arr, years = avg_over_year_all_stn(df, att, start, end)
        # plt.plot(years, att_arr, markers[0])
        bok.plot_line(years, att_arr)			# bokeh

    else:

        for i, stn in enumerate(stn_arr):
            att_arr, years = avg_over_year(df, stn, att, start, end)
            # plt.plot(years, att_arr, markers[i], label=KNMI.stn[stn].name)
            bok.plot_line(years, att_arr)

    # plt.xlabel("Jaren")
    # plt.ylabel(KNMI.attributes[att])

    # if len(stn_arr) >= 1:
    #     plt.legend()

    # plt.show()

def plot_measure_month(df, stn_arr, att=None, start=19010101, end=20991231,
                       colors=COLORS):
    """Plot the number of entries the array of stations have gathered each month
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    colors = a list of colors in the plot."""

    all_stn = len(stn_arr) < 1
    if all_stn:
        stn_arr = np.array(KNMI.Station.all_num)

    bottom = np.zeros(12)
    for i, stn in enumerate(stn_arr):
        att_arr, months = measured_months_stn(df, stn, att, start, end)

        if all_stn:
            plt.bar(months, att_arr, bottom=bottom, color=colors[0])

        else:
            plt.bar(months, att_arr, bottom=bottom, color=colors[i],
                    label=KNMI.stn[stn].name)

        bottom += att_arr

    plt.xlabel("Maanden")

    if not all_stn:
        plt.legend()

    if att:
        plt.ylabel("Number of entries of: " + att)
    else:
        plt.ylabel("Number of entries")

    plt.show()


def plot_measure_year(df, stn_arr, att=None, start=19010101, end=20991231,
                      colors=COLORS):
    """Plot the number of entries the array of stations have gathered each year.
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    colors = a list colors in the plot."""

    all_stn = len(stn_arr) < 1
    if all_stn:
        stn_arr = np.array(KNMI.Station.all_num)

    years = np.arange(start // 10000, end // 10000 + 1)
    counts = np.zeros([len(stn_arr), len(years)])
    for i, stn in enumerate(stn_arr):
        mes_arr, _ = measured_years_stn(df, stn, att, start, end)
        counts[i] = mes_arr

    mask = np.any(counts, axis=0)
    counts = counts[:, mask]
    years = years[mask]

    bottom = np.zeros(len(years))
    for i, count in enumerate(counts):

        if all_stn:
            plt.bar(years, count, bottom=bottom, color=colors[0])

        else:
            plt.bar(years, count, bottom=bottom, color=colors[i],
                    label=KNMI.stn[stn_arr[i]].name)

        bottom += count

    plt.xlabel("Jaren")

    if not all_stn:
        plt.legend()

    if att:
        plt.ylabel("Number of entries of: " + att)
    else:
        plt.ylabel("Number of entries")

    plt.show()


def boxplot_att(df, att, start=19010101, end=20991231):
    '''Plots a boxplot for an attribute grouped by station. Statement 'by='STN' can easily be removed'''
    df.boxplot(by='STN', column=[att], grid=False)
    plt.xlabel('stations')
    plt.ylabel(att)
    plt.legend()
    plt.show()


def covariance(atr1, atr2):
    '''Calculates the covariance of two given attributes to indicate how much the correlate  '''
    filename = KNMI.PATH
    final_filename = filename[:filename.rindex('.')] + "_final.csv"
    df = pd.read_csv(final_filename)
    corr = df[atr1].corr(df[atr2])
    print(corr)


def main():
    reduced_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + ".csv"
    df = pd.read_csv(reduced_filename)
    trn, dev, tst = ml.Lq_Fit.seperate_trn_dev_tst(df)

    final_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + "_final.csv"
    df_final = pd.read_csv(final_filename)

    att = input("Which attribute do you want to analyse? ")
    att = att.upper()

    print("You asked for:", KNMI.attributes[att])

    uni.att_values(df, att)
    uni.boxplot_att(df, att, save=False)
    uni.histogram_att(df, att, save=False)

    plot_att_year(df, [], att)
    plot_att_year_bok(df, [], att)
    plot_att_month(df, [], att)

    for other_att in MEAN_ATTS:
        print("\nFinding correlation of", KNMI.attributes[att], "with",
                KNMI.attributes[other_att])
        print("Correlation is", df_final[att].corr(df[other_att]))

        plot_att_conditional(df, [], att, other_att)

        choice = ""
        while choice != "y" and choice != "n" and choice != "s":
            choice = input("Do you want regression over these two "
                            + "attributes?\nyes (y), no (n), "
                            + "yes with switched axis (s): ")
            choice = choice.lower()

        if choice == "y":
            poly = ml.try_poly_fit(trn, dev, att, other_att)
            ml.plot_poly(tst, poly, att, other_att)

        elif choice == "s":
            poly = ml.try_poly_fit(trn, dev, other_att, att)
            ml.plot_poly(tst, poly, other_att, att)


if __name__ == "__main__":
    main()
