import os.path
import getopt
import sys
from config import *
from neuralnet import acc_increase, predict, training


def main(argv):
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = LOGLEVEL

        opts, args = getopt.getopt(argv, "htip:", ["pstring="])
    except getopt.GetoptError as msg:
        print('error :' + str(msg))
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage app.py [option] [argument] ' + '\n' + '-h Help commands' + '\n' + '-t Train network' + '\n' + '-i Increase Model' + '\n' + '-p <data string> Neural Net Predict')
            sys.exit()
        elif opt == '-t':
            training()
            sys.exit()
        elif opt == '-i':
            acc_increase()
            sys.exit()
        elif opt in ("-p", "--pstring"):
            predict(arg)
            sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])

