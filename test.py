import tests
import pkgutil
import argparse
import re
from textwrap import dedent
from os.path import join

def get_args_parser():
  parser = argparse.ArgumentParser(
    description="Solves Rubik's Cubes using Thistlethwaite's algorithm."
  )
  parser.add_argument(
    '-cdb', '--cube_database', nargs=1, default=['lookuptables.db'],
    help=dedent('''
      The sqlite database where the lookup tables are stored.
      Example: 'lookuptables.db'
    ''')
  )
  parser.add_argument(
    '-f', '--filter', nargs=1, default=['.*'],
    help=dedent('''
      A regex filter to only run certain tests.
      Matches on test path relative to test folder.
    ''')
  )
  return parser

def run_all_tests(pkg, args, prefix=""):
  for test in pkgutil.iter_modules(pkg.__path__):
    importer = test[0]
    test_name = test[1]
    is_pkg = test[2]
    module = importer.find_module(test_name)\
                     .load_module(test_name)
    full_test_name = join(prefix, test_name)
    if is_pkg:
      run_all_tests(module, args, prefix=full_test_name)
    else:
      if re.search(args.filter[0], full_test_name):
        print(f"Testing {full_test_name}...")
        module.main(args)

def main():
  args_parser = get_args_parser()
  args = args_parser.parse_args()
  run_all_tests(tests, args)
  print("All tests passed! =)")

if __name__ == '__main__':
  main()
