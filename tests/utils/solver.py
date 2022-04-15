import thistlethwaite.utils.cube as cube
import thistlethwaite.utils.solver as slv
import random

def test_g2modg3_parity_error(database):
  cb = cube.G0(
    [False]*12, [0]*8, 
    [2,1,0,3]+list(range(4,12)),
    [0,3,2,1]+list(range(4,8))
  )
  solver = slv.Solver(database)
  moves = solver.solve(cb)
  for move in moves:
    cb.do(move)
  assert cb == cube.G0.ident()

def test_g0_scramble():
  cb = cube.G0.ident()
  scramble = [
    cube.Move.DOWN,
    cube.Move.RIGHT_2,
    cube.Move.DOWN_INV,
    cube.Move.FRONT_2,
    cube.Move.RIGHT_2,
    cube.Move.FRONT_2,
    cube.Move.DOWN_INV,
    cube.Move.UP,
    cube.Move.RIGHT_2,
    cube.Move.DOWN_2,
    cube.Move.BACK_2,
    cube.Move.DOWN,
    cube.Move.BACK_2,
    cube.Move.UP_INV,
  ]
  for move in scramble:
    cb.do(move)
  #print(cb)
  assert cb == cube.G0.decode(
    bytes('000000000000,00000000,810b5467a329,47561320', 'utf-8'))

def test_g2modg3_unknown_error(database):
  scramble = [cube.Move.DOWN, cube.Move.RIGHT_2, cube.Move.DOWN_INV]
  solver = slv.Solver(database)
  cb = cube.G0.ident()
  for move in scramble:
    cb.do(move)
  moves = solver.solve(cb)
  for move in moves:
    cb.do(move)
  assert cb == cube.G0.ident()

def fuzz_solver(database):
  gen = random.Random(12345)
  all_moves = list(cube.G2ModG3.valid_moves())
  solver = slv.Solver(database)
  for trial in range(1000):
    cb = cube.G0.ident()
    scramble = []
    for i in range(3):
      move = gen.choice(all_moves)
      scramble.append(move)
      cb.do(move)
    print(scramble)
    moves = solver.solve(cb)
    for move in moves:
      cb.do(move)
    assert cb == cube.G0.ident()

def main(args):
  db = args.cube_database[0]
  test_g0_scramble()
  test_g2modg3_unknown_error(db)
  test_g2modg3_parity_error(db)
  fuzz_solver(db)
