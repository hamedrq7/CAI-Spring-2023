
import argparse
parser = argparse.ArgumentParser(description='vis')

parser.add_argument('-l','--domains', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

print(args)
