import thistlethwaite.utils.cube as cube
import numpy as np
import random
from textwrap import dedent
import hashlib

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

def test_invert():
  moves = [cube.Move.RIGHT, cube.Move.LEFT, cube.Move.UP_INV, cube.Move.BACK_2]
  expected = [cube.Move.BACK_2, cube.Move.UP, cube.Move.LEFT_INV, cube.Move.RIGHT_INV]
  actual = cube.invert(moves)
  assert len(actual) == len(expected)
  for act, exp in zip(actual, expected):
    assert act == exp

def test_edge_orientation_vecs():
  for flipped in [False, True]:
    for edge in cube.Edge.__members__.values():
      vec = cube.edge_vec(edge)
      orientation_vec = cube.edge_orientation_vec(vec, flipped)
      unvec = cube.edge_orientation_from_vec(vec, orientation_vec)
      assert unvec == flipped

def test_corner_vec():
  for corner in cube.Corner.__members__.values():
    vec = cube.corner_vec(corner)
    corner2 = cube.vec_corner(vec)
    assert corner2 == corner

def test_corner_orientation_vec():
  for corner in cube.Corner.__members__.values():
    corner_vec = cube.corner_vec(corner)
    for orientation in range(3):
      orientation_vec = cube.corner_orientation_vec(corner_vec, orientation)
      #print(f"{corner} {corner_vec} {orientation} {orientation_vec}")
      orientation2 = cube.corner_orientation_from_vec(corner_vec, orientation_vec)
      assert orientation2 == orientation

def test_corner_permutations_and_deltas():
  expected = [[cube.Corner.UFR, cube.Corner.DRF, cube.Corner.DFL, cube.Corner.ULF]]
  actual = cube.CORNER_PERMUTATIONS[cube.Move.FRONT]
  for exp_cycle, act_cycle in zip(expected, actual):
    for exp, act in zip(exp_cycle, act_cycle):
      assert exp == act

  expected = [1, 2, 0, 0, 2, 1, 0, 0]
  actual = cube.CORNER_ORIENTATION_DELTAS[cube.Move.FRONT]
  for exp, act in zip(expected, actual):
    assert exp == act

class CubeLikeTest:
  def __init__(self, clazz, coset_translation_method = None):
    self.clazz = clazz
    self.valid_moves = clazz.valid_moves()
    seed = hashlib.md5(bytes(str(self.clazz), 'utf-8')).digest()
    self.gen = random.Random(seed)
    self.fuzz_trials = 1000
    self.move_depth = 30
    self.coset_translation_method = coset_translation_method
  def rand_cube(self, debug=False) -> cube.TCubeLike:
    rand = self.clazz.ident()
    for i in range(self.move_depth):
      move = self.gen.choice(self.valid_moves)
      rand.do(move)
      if debug:
        print(f'{move}: {rand}')
    return rand
  def fuzz_encoding(self):
    for trial in range(self.fuzz_trials):
      cb = self.rand_cube()
      encoded = cb.encode()
      decoded = self.clazz.decode(encoded)
      assert cb == decoded
  def fuzz_moves(self):
    ident = self.clazz.ident()
    for trial in range(self.fuzz_trials):
      moves = [self.gen.choice(self.valid_moves) for _ in range(self.move_depth)]
      cb = ident.copy()
      for move in moves:
        cb.do(move)
      inverse_moves = cube.invert(moves)
      for move in inverse_moves:
        cb.do(move)
      assert cb == ident
  def fuzz_transforms(self):
    ident = self.clazz.ident()
    for trial in range(self.fuzz_trials):
      moves = [self.gen.choice(self.valid_moves) for _ in range(self.move_depth)]
      constructed = ident.copy()
      for move in moves:
        constructed.do(move)
      symmetries = list(constructed.iter_symmetries())
      sym_id, symmetry = self.gen.choice(symmetries)
      transform = cube.SYMMETRY_TRANSFORMS[sym_id]
      transformed_moves = cube.apply_transform_to_moves(moves, transform)
      constructed_transform = ident.copy()
      for move in transformed_moves:
        constructed_transform.do(move)
      if constructed_transform != symmetry:
        x_flip, y_rot, x_rot, z_rot = cube.SYMMETRY_IDS[sym_id]
        print(dedent(f"""
          Fuzz transform trial {trial} failed.
          x_flip:{x_flip}, y_rot:{y_rot}, x_rot:{x_rot}, z_rot:{z_rot}
          moves: {moves}
          transformed_moves: {transformed_moves}
          direct symmetry: {symmetry}
          move constructed symmetry: {constructed_transform}
        """))
        raise Exception()
  def fuzz_symmetries_group(self):
    # test that symmetries are invertable
    for trial in range(self.fuzz_trials):
      cb = self.rand_cube()
      _, sym = self.gen.choice(list(cb.iter_symmetries()))
      matched = False
      for _, sym_sym in sym.iter_symmetries():
        if sym_sym == cb:
          matched = True
          break
      assert matched
  def fuzz_coset_translation_method(self):
    if self.coset_translation_method is None:
      return
    for trial in range(self.fuzz_trials):
      full_cube = cube.G0.ident()
      coset_cube = self.clazz.ident() 
      for i in range(self.move_depth):
        move = self.gen.choice(self.valid_moves)
        full_cube.do(move)
        coset_cube.do(move)
      translated = self.coset_translation_method(full_cube)
      if translated != coset_cube:
        print(f'translated: {translated}')
        print(f'coset: {coset_cube}')
      assert translated == coset_cube
  def run_all(self):
    self.fuzz_moves()
    self.fuzz_encoding()
    self.fuzz_transforms()
    self.fuzz_symmetries_group()
    self.fuzz_coset_translation_method()

class G0ModG1Test(CubeLikeTest):
  def __init__(self):
    super().__init__(cube.G0ModG1, cube.G0.get_g0modg1)
class G1ModG2Test(CubeLikeTest):
  def __init__(self):
    super().__init__(cube.G1ModG2, cube.G0.get_g1modg2)
class G2ModG3Test(CubeLikeTest):
  def __init__(self):
    super().__init__(cube.G2ModG3, cube.G0.get_g2modg3)
class G3ModG4Test(CubeLikeTest):
  def __init__(self):
    super().__init__(cube.G3ModG4, cube.G0.get_g3modg4)
class G0Test(CubeLikeTest):
  def __init__(self):
    super().__init__(cube.G0)

def test_moves():
  cases = [
    (cube.Move.UP,        '000000000000,00000000,3012456789ab,30124567'),
    (cube.Move.UP_INV,    '000000000000,00000000,1230456789ab,12304567'),
    (cube.Move.UP_2,      '000000000000,00000000,2301456789ab,23014567'),
    (cube.Move.DOWN,      '000000000000,00000000,012345679ab8,01235674'),
    (cube.Move.DOWN_INV,  '000000000000,00000000,01234567b89a,01237456'),
    (cube.Move.DOWN_2,    '000000000000,00000000,01234567ab89,01236745'),
    (cube.Move.LEFT,      '000000000000,02100120,016342a7895b,02634157'),
    (cube.Move.LEFT_INV,  '000000000000,02100120,01534a27896b,05134627'),
    (cube.Move.LEFT_2,    '000000000000,00000000,01a34657892b,06534217'),
    (cube.Move.RIGHT,     '000000000000,10022001,4123856079ab,41207563'),
    (cube.Move.RIGHT_INV, '000000000000,10022001,7123056849ab,31270564'),
    (cube.Move.RIGHT_2,   '000000000000,00000000,8123756409ab,71243560'),
    (cube.Move.FRONT,     '010011000100,21001200,0523196784ab,15230467'),
    (cube.Move.FRONT_INV, '010011000100,21001200,0423916785ab,40235167'),
    (cube.Move.FRONT_2,   '000000000000,00000000,0923546781ab,54231067'),
    (cube.Move.BACK,      '000100110001,00210012,0127453b89a6,01374526'),
    (cube.Move.BACK_INV,  '000100110001,00210012,012645b389a7,01624573'),
    (cube.Move.BACK_2,    '000000000000,00000000,012b457689a3,01764532'),
  ]
  for move, encoding in cases:
    exp = cube.G0.decode(bytes(encoding, 'utf-8'))
    act = cube.G0.ident()
    act.do(move)
    if act != exp:
      ms = str(move)
      print(f'For move {ms},\nExpected: {exp}\nGot: {act}')
      assert False

def main(args):
  test_moves()
  test_move_vec()
  test_move_rot()
  test_edge_vec_edge()
  test_edge_permutations()
  test_vec_angle_move()
  test_symmetry_transforms()
  test_apply_transform_to_moves()
  test_invert()
  test_corner_vec()
  test_corner_orientation_vec()
  test_corner_permutations_and_deltas()
  G0ModG1Test().run_all()
  G1ModG2Test().run_all()
  G2ModG3Test().run_all()
  G3ModG4Test().run_all()
  G0Test().run_all()
