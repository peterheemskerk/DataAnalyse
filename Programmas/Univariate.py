import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

plt.rcParams['figure.figsize'] = (15, 5)


def att_values(df, att):
    print("Mean: ",df[att].mean())
    print("Median: ",df[att].median())
    print("Mode: ",df[att].mode())
    print("Variance: ",df[att].var())
    print("Standard Deviation: ",df[att].std())


def boxplot_att(df, att, start=19010101, end=20991231):
    df.boxplot(column=[att], grid=False) 
    plt.ylabel(att)
    plt.show()


def histogram_att(df, att):
    maxv = dataframe[att].max()
    minv = dataframe[att].min()
    diff = int(maxv-minv)
    df[att].hist(bins=diff, range=(minv,maxv))
    plt.show()  


def main():
    dataframe = pd.read_csv('Weerdata_reduced.csv', encoding='latin1')
    Att = input("Name of attribute: ")
    att_values(dataframe, Att)
    boxplot_att(dataframe, Att)
    histogram_att(dataframe, Att)


main()
