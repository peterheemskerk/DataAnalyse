# For this program you need to have your KNMI txt file downloaded. This program
# will make 2 additional csv files:
# The first will be the .txt file, but converted to .csv
# The second will be the same as the first, but with all attributes and entries
# removed that don't occur often enough.

import numpy as np

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
    filename = input("Give the relative path to your KNMI text file: ")

    print("Converting", filename, "to a .csv file...", end=' ', flush=True)
    csv_filename, n_lines = to_csv(filename)
    print("Done.\nThe new file is called", csv_filename, "and has", n_lines,
          "entries.")

    print("Find out which attributes have data in at least", ATR_THRESH * 100,
          "percent of the entries...", end=' ', flush=True)
    atributes = find_valid_atributes(csv_filename)
    print("Done.\nThe valid attributes are:", atributes)

    print("Remove invalid attributes and entries that don't have at least",
          ENT_THRESH * 100, "percent of the attributes...", end=' ', flush=True)
    reduced_filename, n_lines = reduce_entries(csv_filename, atributes)
    print("Done.\nThe new file is called", reduced_filename, "and has", n_lines,
          "entries.")

main()