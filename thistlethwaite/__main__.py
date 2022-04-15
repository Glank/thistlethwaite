import argparse
import os
from textwrap import dedent
import utils.cube as cube
import utils.group_builder as gb

def get_args_parser():
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
    '-b', '--build',
    metavar='TABLE', nargs='?', default='notset',
    help=dedent('''
      Build the thistlethwaite lookup tables used to generate solutions to the given database.
      Optionally, supply the TABLE to built: g0modg1, g1modg2, or g2modg3.
      If no table is specified, all tables will be built.
    ''')
  )
  actions.add_argument(
    '-s', '--solve',
    metavar='CUBESPEC', nargs=1, required=False,
    help='Find the moves to solve the given cube. TODO: describe CUBESPEC'
  )
  return parser

def main():
  args_parser = get_args_parser()
  args = args_parser.parse_args()
  if args.build != 'notset':
    build_specs = {
      'g0modg1': (cube.G0ModG1, 2048),
      'g1modg2': (cube.G1ModG2, 1082565),
      'g2modg3': (cube.G2ModG3, 9800),
      'g3modg4': (cube.G3ModG4, 663552),
    }
    to_build = list(build_specs.keys())
    if args.build is not None:
      if args.build in build_specs:
        to_build = [args.build]
      else:
        args_parser.error(f'Invalid table: {args.build}')
    for table in to_build:
      clazz, expected_size = build_specs[table]
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
