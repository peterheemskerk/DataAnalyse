# This program can plot given attributes of given station over all the time
# that the station has made data.

import pandas as pd
import numpy as np
import KNMI

# Importing bokeh and pandas (Although we dont need Pandas just yet)
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import DataRange1d
from bokeh.embed import components

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

MARKERS = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko',
           'bs', 'gs', 'rs', 'cs', 'ms', 'ys', 'ks',
           'b^', 'g^', 'r^', 'c^', 'm^', 'y^', 'k^',
           'bd', 'gd', 'rd', 'cd', 'md', 'yd', 'kd']

SEASONS = {"spring": (321, 620), "summer": (621, 920), "autumn": (921, 1220),
           "winter": (1221, 320)}

MEAN_ATTS = ["DDVEC", "FG", "TG", "Q", "PG", "UG"]

def plot_line(x, y, att):
    # Make sure x and y are of the same length.
    if len(x) != len(y):
        print('x and y are not of same length')
        return

    # Create a figure
    f = figure(plot_width=600, plot_height=400)

    # Adding some style
    f.background_fill_color="olive"
    f.background_fill_alpha=0.3
    # Add a Title to the plot
    f.title.text="gemiddeld Uurvak waarin de hoogste Windstoot is gemeten"
    f.title.text_font_size="15px"
    f.title.align="center"
    # Add some axis information (after all, a plot without axis descriptions is nothing more than abstract art)
    f.xaxis.axis_label="years"
    f.yaxis.axis_label=KNMI.attributes[att]		# deze nog specifiek maken
    # Add some different colors for the labels and the digits as well
    f.axis.axis_label_text_color="blue"
    f.axis.major_label_text_color="red"
    f.axis.axis_label_text_font_size = "12pt"
    f.yaxis.major_label_text_font_size = "12pt"
    f.xaxis.major_label_text_font_size = "15pt"

    # Plot the line
    # f.line(x, y)		# plots a line
    # f.triangle(x, y)		# plots a scatter
    # f.circle(x, y)		# plots dots/circle
    f.circle(x, y, size=6, fill_alpha=0.2)
    
    script, div= components(f)
    print(script)
    print(div)
    # A webpage should open, giving you a "slightly" interactive plot
    show(f)

def plot_con(x, y, xatt, yatt):
    # Make sure x and y are of the same length.
    if len(x) != len(y):
        print('x and y are not of same length')
        return

    # Create a figure
    f = figure(plot_width=600, plot_height=400)

    # Adding some style
    f.background_fill_color="olive"
    f.background_fill_alpha=0.3
    # Add a Title to the plot
    f.title.text="Verband Luchtvochtigheid en Luchtdruk"
    f.title.text_font_size="16px"
    f.title.align="center"
    # Add some axis information (after all, a plot without axis descriptions is nothing more than abstract art)
    f.xaxis.axis_label=KNMI.attributes[xatt]
    f.yaxis.axis_label=KNMI.attributes[yatt]		# deze nog specifiek maken
    # Add some different colors for the labels and the digits as well
    f.axis.axis_label_text_color="blue"
    f.axis.major_label_text_color="red"
    f.axis.axis_label_text_font_size = "9pt"
    f.yaxis.major_label_text_font_size = "9pt"
    f.xaxis.major_label_text_font_size = "15pt"

    # Plot the line
    # f.line(x, y)		# plots a line
    f.triangle(x, y)		# plots a scatter
    # f.circle(x, y)		# plots dots/circle
    # f.circle(x, y, size=6, fill_alpha=0.2)
    
    script, div= components(f)
    print(script)
    print(div)
    # A webpage should open, giving you a "slightly" interactive plot
    show(f)

def plot_histogram(x, y, att):
    label = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    f = figure(x_range=label, plot_width=600, plot_height=400)
    
    # Adding some style
    f.background_fill_color="olive"
    f.background_fill_alpha=0.3
    # Add a Title to the plot
    f.title.text="Gemiddelde Windrichting per maand"
    f.title.text_font_size="16px"
    f.title.align="center"
    # Add some axis information (after all, a plot without axis descriptions is nothing more than abstract art)
    f.yaxis.axis_label="gemiddelde Windrichting in graden Celsius"		# deze nog specifiek maken
    # Add some different colors for the labels and the digits as well
    f.axis.axis_label_text_color="blue"
    f.axis.major_label_text_color="red"
    f.xaxis.major_label_orientation = np.pi/4
    f.axis.axis_label_text_font_size = "12pt"
    f.xaxis.major_label_text_font_size = "15pt"

    f.vbar(x=label, top=y, width=0.9)

    
    script, div= components(f)
    print(script)
    print(div)  
    show(f)

