# This program can plot given attributes of given station over all the time
# that the station has made data.

import pandas as pd
import numpy as np

# Importing bokeh and pandas (Although we dont need Pandas just yet)
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import DataRange1d

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

MARKERS = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
           'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks',
           'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',
           'bd', 'gd', 'rd', 'cd', 'md', 'yd', 'kd']

SEASONS = {"spring": (321, 620), "summer": (621, 920), "autumn": (921, 1220),
           "winter": (1221, 320)}

MEAN_ATTS = ["DDVEC", "FG", "TG", "Q", "PG", "UG"]

def plot_line(x, y):
    # Make sure x and y are of the same length.
    if len(x) != len(y):
        print('x and y are not of same length')
        return
    output_file("Line.html")

    # Create a figure
    f = figure(plot_width=1000, plot_height=600)

    # Adding some style
    f.background_fill_color="olive"
    f.background_fill_alpha=0.3
    # Add a Title to the plot
    f.title.text="KNMI dataset"
    f.title.text_font_size="25px"
    f.title.align="center"
    # Add some axis information (after all, a plot without axis descriptions is nothing more than abstract art)
    f.xaxis.axis_label="years"
    f.yaxis.axis_label="average value"		# deze nog specifiek maken
    # Add some different colors for the labels and the digits as well
    f.axis.axis_label_text_color="blue"
    f.axis.major_label_text_color="red"
    # Axes geometry
    # f.x_range=DataRange1d(start=0, end=8)
    f.y_range=DataRange1d(start=90, end=100)

    # Plot the line
    # f.line(x, y)		# plots a line
    # f.triangle(x, y)		# plots a scatter
    # f.circle(x, y)		# plots dots/circle
    f.circle(x, y, size=6, fill_alpha=0.2)

    # A webpage should open, giving you a "slightly" interactive plot
    show(f)

