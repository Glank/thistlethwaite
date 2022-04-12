import thistlethwaite.utils.cube as cube
import thistlethwaite.utils.solver as slv
import random

def fuzz_solver(database):
  gen = random.Random(12345)
  all_moves = list(cube.Move.__members__.values())
  solver = slv.Solver(database)
  for trial in range(1000):
    cb = cube.G0.ident()
    for i in range(30):
      cb.do(gen.choice(all_moves))
    moves = solver.solve(cb)
    for move in moves:
      cb.do(move)
    assert cb == cube.G0.ident()

def main(args):
  fuzz_solver(args['cube_database'])
