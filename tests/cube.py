import thistlethwaite.cube as cube
import numpy as np

def test_move_vec():
  if any(cube.move_vec(cube.Move.UP) != np.array([0,1,0])):
    raise Exception()
  if any(cube.move_vec(cube.Move.UP_INV) != np.array([0,1,0])):
    raise Exception()
  if any(cube.move_vec(cube.Move.LEFT_2) != np.array([-1,0,0])):
    raise Exception()

def test_move_rot():
  expected = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
  ])
  actual = cube.move_rot(cube.Move.RIGHT)
  if any(expected.flatten() != actual.flatten()):
    raise Exception()

def test_edge_vec_edge():
  count = 0
  for edge in cube.Edge.__members__.values():
    vec = cube.edge_vec(edge)
    edge_back = cube.vec_edge(vec)
    if edge != edge_back:
      raise Exception()
    count+=1
  if count != 12:
    raise Exception()

def test_edge_permutations():
  expected = [[cube.Edge.UF, cube.Edge.FR, cube.Edge.DF, cube.Edge.FL]]
  actual = cube.EDGE_PERMUTATIONS[cube.Move.FRONT]
  if len(expected) != len(actual):
    raise Exception()
  for exp, act in zip(expected,actual):
    if len(exp) != len(act):
      raise Exception()
    for e, a in zip(exp, act):
      if e != a:
        raise Exception()

  for move in cube.Move.__members__.values():
    expected_cycles = 2 if int(move)%3==2 else 1
    actual_cycles = len(cube.EDGE_PERMUTATIONS[move])
    if actual_cycles != expected_cycles:
      raise Exception(move)

def test_g0modg1():
  ident = cube.G0ModG1([False]*12)
  r = ident.copy()
  r.do(cube.Move.RIGHT)
  f = ident.copy()
  f.do(cube.Move.FRONT)
  if r != ident:
    raise Exception()
  if f == ident:
    raise Exception()

  actual = ident.copy()
  moves = [
    cube.Move.FRONT,
    cube.Move.RIGHT,
    cube.Move.LEFT_INV,
    cube.Move.BACK,
    cube.Move.UP,
    cube.Move.DOWN_INV,
  ]
  for move in moves:
    actual.do(move)
  expected = cube.G0ModG1([
    bool(o) for o in [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0]
  ])
  if actual != expected:
    raise Exception()

def test_g0modg1_encoding():
  expected = cube.G0ModG1([
    bool(o) for o in [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
  ])
  encoded = expected.encode()
  decoded = cube.G0ModG1.decode(encoded)
  if decoded != expected:
    raise Exception()

# TODO: try to build move table for solving G0ModG1 in memory

def main():
  test_move_vec()
  test_move_rot()
  test_edge_vec_edge()
  test_edge_permutations()
  test_g0modg1()
  test_g0modg1_encoding()
