# This program can plot given attributes of given station over all the time
# that the station has made data.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import KNMI

MARKERS = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
           'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks',
           'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',
           'bd', 'gd', 'rd', 'cd', 'md', 'yd', 'kd']

vget_t = np.vectorize(KNMI.get_t)


def get_attribute(df, stn, att, start=19010101, end=20991231):
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

def get_attribute_all_stn(df, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the time.\n
    df = pandas.DataFrame, all the data extracted from your csv file\n
    all stations.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    data = df.loc[(df.YYYYMMDD >= start)
                  & (df.YYYYMMDD <= end), ["YYYYMMDD", att]].values
    return data[:, 1], data[:, 0]

def avg_over_year(df, stn, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the years.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    stn = station, the number or the name of the station.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = get_attribute(df, stn, att, start, end)

    years = []
    att_avg = []
    for year in range(start // 10000, end // 10000 + 1):
        att = att_arr[t_arr // 10000 == year]

        if len(att) > 0:
            years.append(year)
            att_avg.append(np.mean(att))

    return np.array(att_avg), np.array(years)

def avg_over_year_all_stn(df, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the years.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    average over all stations.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = get_attribute_all_stn(df, att, start, end)

    years = []
    att_avg = []
    for year in range(start // 10000, end // 10000 + 1):
        att = att_arr[t_arr // 10000 == year]

        if len(att) > 0:
            years.append(year)
            att_avg.append(np.nanmean(att))		# used nanmean to ignore nan values

    return np.array(att_avg), np.array(years)

def avg_over_month_all_stn(df, att, start=19010101, end=20991231):
    """Returns a tuple of the values of the given attribute and the years.\n
    df = pandas.DataFrame, all the data extracted from your csv file.\n
    average over all stations.\n
    att = attribute\n
    start = start date\n
    end = end date"""

    att_arr, t_arr = get_attribute_all_stn(df, att, start, end)

    months = []
    att_avg = []
    for month in range(1, 12):
        att = att_arr[(t_arr // 100) % 100 == month]

        if len(att) > 0:
            months.append(month)
            att_avg.append(np.nanmean(att))		# used nanmean to ignore nan values

    return np.array(att_avg), np.array(months)

def plot_att_year(df, stn_arr, att, start=19010101, end=20991231, markers=MARKERS):
    """Plot the average value of an attribute over a year from multiple stations
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""
    
    if stn_arr == []:
        att_arr, years = avg_over_year_all_stn(df, att, start, end)
        plt.plot(years, att_arr, markers[0], label = 'all stations')
    else:
        for i, stn in enumerate(stn_arr):
            att_arr, years = avg_over_year(df, stn, att, start, end)
            plt.plot(years, att_arr, markers[i], label=KNMI.stn[stn].name)

    plt.xlabel("Jaren")
    plt.ylabel(KNMI.attributes[att])
    plt.legend()
    plt.show()


def plot_att_month(df, stn_arr, att, start=19010101, end=20991231, markers=MARKERS):
    """Plot the average value of an attribute over a year from multiple stations
    \ndf = pandas.DataFrame, all data extracted from your csv file.\n
    stn_arr = station array, an array of the numbers or names of the stations.\n
    att = attribute\n
    start = start date\n
    end = end date\n
    markers = a list of the shape and color of the markers in the plot."""
    
    if stn_arr == []:
        att_arr, months = avg_over_month_all_stn(df, att, start, end)
        plt.plot(months, att_arr, markers[0], label = 'all stations')
    else:
        print('not implemented yet')
        '''
        for i, stn in enumerate(stn_arr):
            att_arr, years = avg_over_year(df, stn, att, start, end)
            plt.plot(years, att_arr, markers[i], label=KNMI.stn[stn].name)
        '''
    plt.xlabel("Maanden")
    plt.ylabel(KNMI.attributes[att])
    plt.legend()
    plt.show()

def boxplot_att(df, att, start=19010101, end=20991231):
    df.boxplot(by='STN', column=[att], grid=False)
    plt.xlabel('stations')
    plt.ylabel(att)
    plt.legend()
    plt.show()

def main():
    reduced_filename = KNMI.PATH[:KNMI.PATH.rindex('.')] + ".csv"
    df = pd.read_csv(reduced_filename)

    att = 'UN'
    stations = []

    plot_att_month(df, stations, att)

    '''
    data = df.loc[(df.STN > 210) & (df.STN < 280)]
    boxplot_att(data, att)

    data = df.loc[(df.STN >= 280) & (df.STN < 320)]
    boxplot_att(data, att)

    data = df.loc[(df.STN >= 320)]
    boxplot_att(data, att)

    stations = []	
    # stations = [209, 210, 270, 286]	

    plot_att_year(df, stations, att)
    '''

main()
