#!/usr/bin/env python

from irisSeg import irisSeg
from sys import argv


def main():
    if len(argv) != 4:
        print('Error : Please enter the file path and min and max radius \n>>iris-seg img.jpg 40 70')
    else:
        filename, _min, _max = [str(argv[i]) for i in range(1, 4)]
        irisSeg(filename,_min,_max,view_output=True)
