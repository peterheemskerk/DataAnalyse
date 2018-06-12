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
                         markers=MARKERS, month=np.arange(12)+1):
    """Plot the average value of an attribute over another attribute from
    multiple stations.\n
    df = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""

    all_stn = len(stn_arr) < 1
    if all_stn:
        stn_arr = np.array(KNMI.Station.all_num)

    for i, stn in enumerate(stn_arr):
        xatts, yatts = avg_over_att(df, stn, xatt, yatt, start, end)

        if all_stn:
            plt.plot(xatts, yatts, markers[0])

        else:
            plt.plot(xatts, yatts, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel(KNMI.attributes[xatt])
    plt.ylabel(KNMI.attributes[yatt])

    if not all_stn:
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

    all_stn = len(stn_arr) < 1
    if all_stn:
        stn_arr = np.array(KNMI.Station.all_num)

    for i, stn in enumerate(stn_arr):
        att_arr, seasons = avg_over_season(df, stn, att, start, end)

        if all_stn:
            plt.plot(seasons, att_arr, markers[0])

        else:
            plt.plot(seasons, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Seizoenen")
    plt.ylabel(KNMI.attributes[att])

    if not all_stn:
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

    all_stn = len(stn_arr) < 1
    if all_stn:
        stn_arr = np.array(KNMI.Station.all_num)

    for i, stn in enumerate(stn_arr):
        att_arr, years = avg_over_year(df, stn, att, start, end)

        if all_stn and len(att_arr) > 1:
            plt.plot(years, att_arr, markers[0])

        elif len(att_arr) > 1:
            plt.plot(years, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Jaren")
    plt.ylabel(KNMI.attributes[att])

    if not all_stn:
        plt.legend()

    plt.show()


def plot_measure_month(df, stn_arr, att=None, start=19010101, end=20991231,
                       colors=COLORS):

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


# def plot_measure_year(df, stn_arr, att=None, start=19010101, end=20991231,
#                       colors=COLORS):

#     all_stn = len(stn_arr) < 1
#     if all_stn:
#         stn_arr = np.array(KNMI.Station.all_num)

#     years = np.arange(start // 10000, start // 10000 + 1)
#     counts = np.zeros(len(years), len(stn_arr))
#     for i, stn in enumerate(stn_arr):
#         mes_arr, _ = measured_years_stn(df, stn, att, start, end)
#         counts[i] = mes_arr

#     bottom = np.zeros(len(years))
#     for i, count in counts:

#         if all_stn:
#             plt.bar(count, att_arr, bottom=bottom, color=colors[0])

#         else:
#             plt.bar(count, att_arr, bottom=bottom, color=colors[i],
#                     label=KNMI.stn[stn].name)

#         bottom += att_arr

#     plt.xlabel("Jaren")

#     if not all_stn:
#         plt.legend()

#     if att:
#         plt.ylabel("Number of entries of: " + att)
#     else:
#         plt.ylabel("Number of entries")

#     plt.show()


def main():
    reduced_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + ".csv"
    df = pd.read_csv(reduced_filename)

    plot_measure_month(df, [])
    plot_measure_month(df, [210, 235, 240, 242], "FXX", end=19800101)


main()