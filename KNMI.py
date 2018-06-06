# This program produces two dictionaries that make retrieving the information
# about the stations and the attributes smoother in programs that import this.

_PATH_KNMI = ""
_START_CHAR_STN = 690
_START_CHAR_ATT = 3554

stn = dict()
attributes = dict()

# The station class stores some information about each station
class Station:

    def __init__(self, num, name, lon, lat, alt):
        self.num = num
        self.name = name
        self.lon = lon
        self.lat = lat
        self.alt = alt

    def __str__(self):
        return "STN " + str(self.num) + ":\"" + self.name + "\""


# Read the name and explanation of the attributes and put them in the dictionary
def _read_attributes(filename):
    with open(filename) as text:
        text.seek(_START_CHAR_ATT)

        line = text.readline()[2:-1]
        while line:
            name = line[:line.index(' ')]
            explanation = line[line.index("=")+2:-2]
            attributes[name] = explanation

            line = text.readline()[2:-1]


# Read the details about the station and but them in the dictionary.
def _read_stations(filename):
    with open(filename) as text:
        text.seek(_START_CHAR_STN)

        line = text.readline()[2:-1]
        while line:
            num = int(line[:line.index(':')])

            line = line[line.index(':')+1:].lstrip()
            lon = float(line[:line.index(' ')])

            line = line[line.index(' ')+1:].lstrip()
            lat = float(line[:line.index(' ')])

            line = line[line.index(' ')+1:].lstrip()
            alt = float(line[:line.index(' ')])

            name = line[line.rindex(' ')+1:]

            station = Station(num, name, lon, lat, alt)
            stn[num] = station
            stn[name] = station

            line = text.readline()[2:-1]


def main(filename):
    if not filename:
        filename = input("Type the relative path to your KNMI text file: ")

    _read_stations(filename)
    _read_attributes(filename)


main(_PATH_KNMI)