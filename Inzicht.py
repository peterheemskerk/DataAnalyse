# For this program you need to have your KNMI txt file downloaded. This program
# will make 2 additional csv files:
# The first will be the .txt file, but converted to .csv
# The second will be the same as the first, but with all attributes and entries
# removed that don't occur often enough.

import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

START_CHAR = 6548
ATR_THRESH = 0.4
ENT_THRESH = 0.4

# Write all entries to a new file that only takes entries and atributes that
# occur a certain percentage of the time.
def reduce_entries(filename, atributes):
    reduced_filename = filename[:filename.rindex('.')] + "_reduced.csv"
    n_lines = 0
    n_atr = len(atributes) - 1

    with open(filename) as raw:
        with open(reduced_filename, 'w') as reduced:
            arr = np.array(raw.readline().split(','))
            arr = arr[atributes]
            reduced.write(','.join(arr))

            line = raw.readline()
            while line:
                arr = np.array(line.split(','))
                arr = arr[atributes]

                if ((arr != "").sum() - 1) / n_atr > ENT_THRESH:
                    n_lines += 1
                    reduced.write(",".join(arr))

                line = raw.readline()

    return reduced_filename, n_lines


# Return an array of all the atributes that occur in at least ATR_THRESH
# percent of the entries.
def find_valid_atributes(filename):
    atributes = []

    with open(filename) as knmi:
        knmi.readline()

        line = knmi.readline()
        atributes = np.zeros(len(line.split(',')))
        while line:
            atributes[np.array(line.split(',')) != ''] += 1
            line = knmi.readline()

    atributes /= atributes[-1]
    return np.arange(len(atributes))[atributes > ATR_THRESH]


# Convert the .txt file to a .csv file
def to_csv(filename):
    csv_filename = filename[:filename.rindex('.')] + ".csv"
    n_lines = 0

    with open(filename) as knmi:
        with open(csv_filename, 'w') as csv:
            knmi.seek(START_CHAR)
            csv.write(knmi.readline().replace(' ', ''))

            knmi.readline()
            line = knmi.readline()
            while line:
                n_lines += 1
                line = line.replace("\t", "")
                line = line.replace(" ", "")
                csv.write(line)

                line = knmi.readline()

    return csv_filename, n_lines


def main():

	csv_filename = input("Give the relative path to your KNMI csv file: ")

	with open(csv_filename) as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		knmi_data = []
		for row in reader:
			knmi_data.append(row)
	    	
	'''
	df=pd.read_csv(csv_filename, sep=',',header=None)
	print('values:', len(df.values))
	
	with open(csv_filename) as csvfile:
		reader = csv.DictReader(csvfile, delimiter=' ')
		print('reader', reader)
		knmi_data = []
		for row in reader:
			knmi_data.append(row)
	
	'''
	print('aantal rijen:', len(knmi_data))
	print('koppen:', knmi_data[0])
	print('eerste rij:', knmi_data[1])
	print('eerste element:', knmi_data[1][0])
	rijen = len(knmi_data)
	kolommen = len(knmi_data[1])
	gegevens = np.empty((rijen, kolommen))

	for i, row in enumerate(knmi_data):
		if i > 0:
			for j, element in enumerate(row):
				# print(i, j, row[j])
				if row[j] != '':
					gegevens[i, j] = row[j]

	# x_list = gegevens[:124, 1]
	x_list = range(124)
	y_list = gegevens[:124, 2]
	y_list2 = gegevens[:124, 3]
	y_list3 = gegevens[:124, 4]
	y_list4 = gegevens[:124, 5]
	y_list5 = gegevens[:124, 6]
	# print(len(x_list), len(y_list))

	# plt.scatter(x_list, y_list, s=area, c=colors, alpha=0.01)
	plt.scatter(x_list, y_list)
	plt.scatter(x_list, y_list2)
	plt.scatter(x_list, y_list3)
	plt.show()
	plt.scatter(x_list, y_list4)
	plt.scatter(x_list, y_list5)	
	plt.show()







main()
