import argparse
import os
from textwrap import dedent
import utils.cube as cube
import utils.group_builder as gb

def parse_args():
  parser = argparse.ArgumentParser(
    description="Solves Rubik's Cubes using Thistlethwaite's algorithm."
  )
  parser.add_argument(
    '-db', '--database', nargs=1,
    help=dedent('''
      The sqlite database where the lookup tables are or will be stored.
      Example: 'thistlethwaite.db'
    ''')
  )
  actions = parser.add_mutually_exclusive_group()
  actions.add_argument(
    '-b', '--build', action='store_true',
    help=dedent('''
      Build the thistlethwaite lookup tables used to generate solutions to the given database.
    ''')
  )
  actions.add_argument(
    '-s', '--solve',
    metavar='CUBESPEC', nargs=1, required=False,
    help='Find the moves to solve the given cube. TODO: describe CUBESPEC'
  )
  return parser.parse_args()

def main():
  args = parse_args()
  if args.build:
    to_build = [
      (cube.G0ModG1, 186),
      (cube.G1ModG2, 136566),
    ]
    for clazz, expected_size in to_build:
      table = clazz.__name__.lower()
      print(f'Building {table}...')
      sqlite_group = gb.SqliteGroup(args.database[0], table)
      builder = gb.GroupBuilder(clazz.ident())
      builder.build(expected_size)
      print('Saving to db...')
      builder.group.save_to(sqlite_group)
    print('All tables successfully built.')
  else:
    raise NotImplementedError("Haven't implemented solving yet.")

if __name__ == '__main__':
  main()
