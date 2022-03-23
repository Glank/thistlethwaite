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

def test_vec_angle_move():
  for move in cube.Move.__members__.values():
    vec, angle = cube.move_vec(move), cube.move_angle(move)
    act = cube.vec_angle_move(vec, angle)
    if act != move:
      raise Exception()

def test_symmetry_transforms():
  if len(cube.SYMMETRY_TRANSFORMS) != 4*2*6:
    raise Exception()
  # should be unique
  for i in range(len(cube.SYMMETRY_TRANSFORMS)):
    si = cube.SYMMETRY_TRANSFORMS[i]
    for j in range(i+1, len(cube.SYMMETRY_TRANSFORMS)):
      sj = cube.SYMMETRY_TRANSFORMS[j]
      if all(si.flatten() == sj.flatten()):
        print(cube.SYMMETRY_IDS[i])
        print(f's{i}:{si}')
        print(cube.SYMMETRY_IDS[j])
        print(f's{j}:{sj}')
        raise Exception()
  # and should be closed under multiplication
  for i in range(len(cube.SYMMETRY_TRANSFORMS)):
    si = cube.SYMMETRY_TRANSFORMS[i]
    for j in range(i+1, len(cube.SYMMETRY_TRANSFORMS)):
      sj = cube.SYMMETRY_TRANSFORMS[j]
      sij = np.matmul(si, sj)
      found = False
      for other in cube.SYMMETRY_TRANSFORMS:
        if all(other.flatten() == sij.flatten()):
          found = True
          break
      if not found:
        raise Exception(i,j)

def test_apply_transform_to_moves():
  # print(cube.SYMMETRY_IDS[0])
  # [1, 0, 0, 0]
  # No transform
  actual = cube.apply_transform_to_moves([cube.Move.RIGHT], cube.SYMMETRY_TRANSFORMS[0])
  expected = [cube.Move.RIGHT]
  if len(actual) != len(expected):
    raise Exception()
  if actual[0] != expected[0]:
    raise Exception()
  # print(cube.SYMMETRY_IDS[6])
  # [1, 90, 0, 0]
  # Rotate widdersins about the y axis
  actual = cube.apply_transform_to_moves([cube.Move.RIGHT], cube.SYMMETRY_TRANSFORMS[6])
  expected = [cube.Move.BACK]
  if len(actual) != len(expected):
    raise Exception()
  if actual[0] != expected[0]:
    raise Exception()
  # print(cube.SYMMETRY_IDS[3])
  # [1, 0, 0, 90]
  # Rotate clockwise about the z axis
  actual = cube.apply_transform_to_moves([cube.Move.RIGHT], cube.SYMMETRY_TRANSFORMS[3])
  expected = [cube.Move.DOWN]
  if len(actual) != len(expected):
    raise Exception()
  if actual[0] != expected[0]:
    raise Exception()
  # print(cube.SYMMETRY_IDS[24])
  # [-1, 0, 0, 0]
  # Just an x flip
  actual = cube.apply_transform_to_moves([cube.Move.RIGHT], cube.SYMMETRY_TRANSFORMS[24])
  expected = [cube.Move.LEFT_INV]
  if len(actual) != len(expected):
    raise Exception()
  if actual[0] != expected[0]:
    raise Exception()

def main():
  test_move_vec()
  test_move_rot()
  test_edge_vec_edge()
  test_edge_permutations()
  test_g0modg1()
  test_g0modg1_encoding()
  test_vec_angle_move()
  test_symmetry_transforms()
  test_apply_transform_to_moves()
