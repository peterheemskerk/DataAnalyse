# This program produces two dictionaries that make retrieving the information
# about the stations and the attributes smoother in programs that import this.
# It also contains a useful helper function that can calculate the number of
# days between 1901/01/01 and any given date after that.

PATH = "../../KNMI.txt"

_START_CHAR_STN = 690
_START_CHAR_ATT = 3554

_YEAR0 = 1901
_MONTH0 = 1
_DAY0 = 1

stn = dict()
attributes = dict()

RED_ATT = ["DDVEC", "FHVEC", "FG", "FHX", "FHXH", "FHN", "FHNH", "FXX", "FXXH",
           "TG", "TN", "TNH", "TX", "TXH", "SQ", "SP", "PG", "PX", "PN", "UG",
           "UX", "UXH", "UNH", "EV24"]

# The station class stores some information about each station
class Station:

    all_num = []

    def __init__(self, num, name, lon, lat, alt):
        self.num = num
        self.name = name
        self.lon = lon
        self.lat = lat
        self.alt = alt

        Station.all_num.append(self.num)

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

            index = explanation.find("Zie http://")
            if index != -1:
                explanation = explanation[:index - 2]

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


def get_t(time):
    """Returns the number of days between 'time' and 1901/01/01.\n
    time = any date after 1901/01/01"""

    year = time // 10000
    d_year = year - _YEAR0
    time -= year * 10000
    d_month = time // 100 - _MONTH0
    d_day = time % 100 - _DAY0

    t = d_year * 365 + d_year // 4
    t += d_month * 31 - d_month // 2
    if d_month >= 2:
        t -= 2

        if year % 4 == 0:
            t += 1

        if d_month >= 8:
            t += 1

    return t + d_day


def _main():
    global PATH
    try:
        open(PATH)
    except IOError:
        PATH = input("Type the relative path to your KNMI text file: ")

    filename = PATH
    _read_stations(filename)
    _read_attributes(filename)


_main()