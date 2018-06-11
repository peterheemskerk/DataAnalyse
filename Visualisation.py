# This program can plot given attributes of given station over all the time
# that the station has made data.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import KNMI

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

MARKERS = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
           'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks',
           'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',
           'bd', 'gd', 'rd', 'cd', 'md', 'yd', 'kd']

SEASONS = {"spring": (321, 620), "summer": (621, 920), "autumn": (921, 1220),
           "winter": (1221, 320)}

vget_t = np.vectorize(KNMI.get_t)


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

    months = np.arange(1, 13)
    att_avg = []
    for month in months:
        att = att_arr[t_arr % 10000 // 100 == month]
        att_avg.append(np.nanmean(att))

    return np.array(att_avg), months


def avg_over_season(df, stn, att, start=19010101, end=10991231):
    """Returns a tuple of the values of the given attribute and the seasons.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = attribute_over_time(df, stn, att, start, end)
    t_arr %= 10000

    seasons = np.array(["winter", "spring", "summer", "autumn"])
    att_avg = []
    for season in seasons:
        s_start, s_end = SEASONS[season]

        att_data = []
        if season == "winter":
            att_data = att_arr[(t_arr >= s_start) | (t_arr <= s_end)]
        else:
            att_data = att_arr[(s_start <= t_arr) & (t_arr <= s_end)]

        att_avg.append(np.nanmean(att_data))

    return np.array(att_avg), seasons


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
        att = att_arr[t_arr // 10000 == year]

        if len(att) > 0:
            years.append(year)
            att_avg.append(np.nanmean(att))

    return np.array(att_avg), np.array(years)


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
    return data[:, 1], data[:, 0]


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
    return data[:, 1:], data[:, 0]


def plot_att_conditional(df, stn_arr, xatt, yatt, start=19010101, end=20991231,
                         markers=MARKERS, month=np.arange(12)+1):
    """Plot the average value of an attribute over another attribute from
    multiple stations.\n
    df = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    for i, stn in enumerate(stn_arr):
        xatts, yatts = avg_over_att(df, stn, xatt, yatt, start, end)
        plt.plot(xatts, yatts, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel(KNMI.attributes[xatt])
    plt.ylabel(KNMI.attributes[yatt])
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

    tot_stn = len(stn_arr) - 1
    for i, stn in enumerate(stn_arr):
        att_arr, months = avg_over_month(df, stn, att, start, end)
        months = months.astype(float) + 0.4 * (i / tot_stn) - 0.2
        width = 0.8 / (tot_stn + 1)
        plt.bar(months, att_arr, width=width, color=colors[i],
                label=KNMI.stn[stn].name)

    plt.xlabel("Maanden")
    plt.ylabel(KNMI.attributes[att])
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

    for i, stn in enumerate(stn_arr):
        att_arr, seasons = avg_over_season(df, stn, att, start, end)
        plt.plot(seasons, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Seizoenen")
    plt.ylabel(KNMI.attributes[att])
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

    for i, stn in enumerate(stn_arr):
        att_arr, years = avg_over_year(df, stn, att, start, end)
        plt.plot(years, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Jaren")
    plt.ylabel(KNMI.attributes[att])
    plt.legend()
    plt.show()


def main():
    reduced_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + ".csv"
    df = pd.read_csv(reduced_filename)

    plot_att_year(df, [210, 270, 286], "FHX")
    plot_att_season(df, [210, 270, 286], "FHX")
    plot_att_month(df, [210, 270, 286], "FHX")
    plot_att_conditional(df, [210, 270, 286], "FHX", "FHVEC")




main()