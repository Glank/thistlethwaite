from collections.abc import Hashable
from enum import IntEnum, auto
from typing import TypeVar, Generic, NewType
import numpy as np
import math

def rot_about_vec(vec:np.array, rads:float):
  """ Returns a rotation vector about the given vector by the given angle in radians. """
  x,y,z = tuple(vec)
  c = math.cos(rads)
  s = math.sin(rads)
  return np.array([
    [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
    [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
    [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
  ])

#### Moves [Start] ####

class Move(IntEnum):
  UP = 0
  UP_INV = auto()
  UP_2 = auto()
  DOWN = auto()
  DOWN_INV = auto()
  DOWN_2 = auto()
  LEFT = auto()
  LEFT_INV = auto()
  LEFT_2 = auto()
  RIGHT = auto()
  RIGHT_INV = auto()
  RIGHT_2 = auto()
  FRONT = auto()
  FRONT_INV = auto()
  FRONT_2 = auto()
  BACK = auto()
  BACK_INV = auto()
  BACK_2 = auto()

def move_vec(move:Move):
  """
    Returns the directional vector associated with a move.
    For instance, U, U', and U2 return <0, 1, 0>,
    R returns <1, 0, 0>,
    F returns <0, 0, 1>,
    etc,
  """
  d = int(move/3)
  return np.array([
    [0, 1, 0],
    [0, -1, 0],
    [-1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, -1]
  ][d])

def move_angle(move:Move) -> int:
  type_ = int(move)%3
  if type_ == 0:
    return 270
  elif type_ == 1:
    return 90
  else:
    return 180

_VEC_ANGLE_TO_MOVE = dict[tuple[int,int,int,int], Move]()
def vec_angle_move(vec, angle:int) -> Move:
  if not _VEC_ANGLE_TO_MOVE:
    for move in Move.__members__.values():
      key = tuple([int(v) for v in move_vec(move)]+[move_angle(move)])
      _VEC_ANGLE_TO_MOVE[key] = move
  if angle < 0:
    angle = 360-((-angle)%360)
  key = tuple([int(v) for v in list(vec)]+[angle])
  return _VEC_ANGLE_TO_MOVE[key]

def _init_inverse_moves() -> list[Move]:
  inverse_moves = list[Move]()
  for move in Move.__members__.values():
    vec = move_vec(move)
    angle = -move_angle(move)
    inverse = vec_angle_move(vec, angle)
    inverse_moves.append(inverse)
  return inverse_moves
INVERSE_MOVES = _init_inverse_moves()

def invert(moves:list[Move]) -> list[Move]:
  global INVERSE_MOVES
  inverse = list[Move]()
  for move in reversed(moves):
    inverse.append(INVERSE_MOVES[move])
  return inverse

def move_rot(move:Move):
  vec = move_vec(move)
  angle = move_angle(move)
  rads = angle*math.pi/180
  rot = rot_about_vec(vec, rads)
  for r in range(3):
    for c in range(3):
      rot[r,c] = int(round(rot[r,c]))
  return rot

#### Moves [End] ####


#### Symmetries [Start] ####

def _init_symmetry_transforms():
  symmetry_transforms = list[np.array]()
  symmetry_ids = []
  for x_flip in [1, -1]:
    step1 = np.array([
      [x_flip, 0, 0],
      [   0, 1, 0],
      [   0, 0, 1],
    ])
    for y_rot in [0,90,180,270]:
      c = int(round(math.cos(y_rot*math.pi/180)))
      s = int(round(math.sin(y_rot*math.pi/180)))
      step2 = np.matmul(step1, np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
      ]))
      for x_rot, z_rot in [(0, 0), (180, 0), (0, 90), (0, -90), (90, 0), (-90, 0)]:
        c = int(round(math.cos(x_rot*math.pi/180)))
        s = int(round(math.sin(x_rot*math.pi/180)))
        step3 = np.matmul(step2, np.array([
          [1, 0,  0],
          [0, c, -s],
          [0, s,  c],
        ]))
        c = int(round(math.cos(z_rot*math.pi/180)))
        s = int(round(math.sin(z_rot*math.pi/180)))
        step3 = np.matmul(step3, np.array([
          [c, -s, 0],
          [s,  c, 0],
          [0,  0, 1],
        ]))
        symmetry_transforms.append(step3)
        symmetry_ids.append([x_flip, y_rot, x_rot, z_rot])
  return symmetry_transforms, symmetry_ids
SYMMETRY_TRANSFORMS, SYMMETRY_IDS = _init_symmetry_transforms()

def apply_transform_to_moves(moves:list[Move], symmetry:np.array) -> list[Move]:
  new_moves = []
  for move in moves:
    angle = move_angle(move)
    vec = move_vec(move)
    new_angle = angle*np.linalg.det(symmetry)
    new_vec = np.matmul(symmetry, vec)
    new_move = vec_angle_move(new_vec, new_angle)
    new_moves.append(new_move)
  return new_moves

#### Symmetries [End] ####


#### Permutations [Start] ####

def transitions_to_cycle_notation(transitions:dict[int,int]) -> list[list[int]]:
  """ Converts a permutation in transition notation into cycle notation. """
  assert set(transitions.keys()) == set(transitions.values())
  cycles = []
  touched = list(transitions.keys())
  visited = set()
  while len(visited) < len(touched):
    cycle = list[int]()
    # get first unvisited
    start = next(iter(e for e in touched if e not in visited))
    visited.add(start)
    cycle.append(start)
    cur = transitions[start]
    while cur != start:
      visited.add(cur)
      cycle.append(cur)
      cur = transitions[cur]
    if len(cycle) > 1:
      cycles.append(cycle)
  return cycles

def apply_permutation(permutation, lst, op=None):
  """
  Applies a permutation in cycle notation.
  op, if given, should be a function taking two parameters,
    the value being permuted and it's initial position,
    and returns the new value after the permutation
  """
  if op is None:
    for cycle in permutation:
      tmp = lst[cycle[-1]]
      for i in range(len(cycle)-1, 0, -1):
        lst[cycle[i]] = lst[cycle[i-1]]
      lst[cycle[0]] = tmp
  else:
    for cycle in permutation:
      tmp = lst[cycle[-1]]
      for i in range(len(cycle)-1, 0, -1):
        lst[cycle[i]] = op(lst[cycle[i-1]], cycle[i-1])
      lst[cycle[0]] = op(tmp, cycle[-1])

#### Permutations [End] ####


#### Edge [Start] ####

class Edge(IntEnum):
  UR = 0
  UF = auto()
  UL = auto()
  UB = auto()
  FR = auto()
  FL = auto()
  BL = auto()
  BR = auto()
  DR = auto()
  DF = auto()
  DL = auto()
  DB = auto()
  
def move_edges(move:Move) -> list[Edge]:
  """ Returns all edges affected by a given move. """
  d = int(move/3)
  return [
    [Edge.UR, Edge.UF, Edge.UL, Edge.UB], #U
    [Edge.DR, Edge.DF, Edge.DL, Edge.DB], #D
    [Edge.UL, Edge.FL, Edge.DL, Edge.BL], #L
    [Edge.UR, Edge.FR, Edge.DR, Edge.BR], #R
    [Edge.UF, Edge.FR, Edge.DF, Edge.FL], #F
    [Edge.UB, Edge.BR, Edge.DB, Edge.BL], #B
  ][d]

def edge_vec(edge:Edge):
  e = int(edge)
  y = 1-int(e/4)
  if y == 0:
    x, z = [
      (1, 1),
      (-1, 1),
      (-1, -1),
      (1, -1)
    ][e%4]
  else:
    x, z = [
      (1, 0),
      (0, 1),
      (-1, 0),
      (0, -1)
    ][e%4]
  return np.array([x, y, z])

def edge_orientation_vec(edge:np.array, flipped:bool) -> np.array:
  x,y,z = tuple(edge)
  if not flipped:
    if y != 0:
      return np.array([0,y,0])
    else:
      return np.array([0,0,z])
  else:
    if y != 0:
      return np.array([x,0,z])
    else:
      return np.array([x,0,0])

def edge_orientation_from_vec(edge:np.array, orientation_vec:np.array) -> bool:
  return all(edge_orientation_vec(edge, True) == orientation_vec)

_VEC_TO_EDGE = dict[tuple[int,int,int],Edge]()
def vec_edge(vec):
  global _VEC_TO_EDGE
  if not _VEC_TO_EDGE:
    for edge in Edge.__members__.values():
      v = edge_vec(edge)
      v = tuple([int(i) for i in v])
      _VEC_TO_EDGE[v] = edge
  vec = tuple([int(i) for i in vec])
  return _VEC_TO_EDGE[vec]

def _init_edge_permutations() -> list[list[list[int]]]:
  """ Permutation of edges for each move in minimal cycle notation """
  all_edge_permutations = list[list[list[int]]]()
  for move in Move.__members__.values():
    edges = move_edges(move)
    edge_vecs = [edge_vec(e) for e in edges]
    rot = move_rot(move)
    after_vecs = [np.matmul(rot, v) for v in edge_vecs]
    after_edges = [vec_edge(v) for v in after_vecs]
    transitions = dict((edges[i], after_edges[i]) for i in range(len(edges)))
    edge_permutations = transitions_to_cycle_notation(transitions)
    all_edge_permutations.append(edge_permutations)
  return all_edge_permutations
EDGE_PERMUTATIONS = _init_edge_permutations()

def _init_edge_symmetries() -> list[list[list[int]]]:
  """ Permutations of edges for each symetry in SYMMETRY_TRANSFORMS """
  all_permutations = list[list[list[int]]]()
  for transform in SYMMETRY_TRANSFORMS:
    edges = list(Edge.__members__.values())
    edge_vecs = [edge_vec(e) for e in edges]
    after_vecs = [np.matmul(transform, ev) for ev in edge_vecs]
    after_edges = [vec_edge(v) for v in after_vecs]
    transitions = dict((edges[i], after_edges[i]) for i in range(len(edges)))
    cycles = transitions_to_cycle_notation(transitions)
    all_permutations.append(cycles)
  return all_permutations
SYMMETRY_EDGE_PERMUTATIONS = _init_edge_symmetries()

#### Edge [End] ####


#### Corner [Start] ####

class Corner(IntEnum):
  UFR = 0
  ULF = auto()
  UBL = auto()
  URB = auto()
  DRF = auto()
  DFL = auto()
  DLB = auto()
  DBR = auto()
  
def move_corners(move:Move) -> list[Corner]:
  """ Returns all cornersaffected by a given move. """
  d = int(move/3)
  return [
    [Corner.UFR, Corner.ULF, Corner.UBL, Corner.URB], #U
    [Corner.DRF, Corner.DFL, Corner.DLB, Corner.DBR], #D
    [Corner.UFR, Corner.URB, Corner.DRF, Corner.DBR], #R
    [Corner.ULF, Corner.UBL, Corner.DFL, Corner.DLB], #L
    [Corner.UFR, Corner.ULF, Corner.DRF, Corner.DFL], #F
    [Corner.UBL, Corner.URB, Corner.DLB, Corner.DBR], #B
  ][d]

def corner_vec(corner:Corner):
  y = 1 if int(corner/4) == 0 else -1
  x, z = [
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
  ][corner%4]
  return np.array([x, y, z])

_VEC_TO_CORNER = dict[tuple[int,int,int],Corner]()
def vec_corner(vec) -> Corner:
  global _VEC_TO_CORNER
  if not _VEC_TO_CORNER:
    for corner in Corner.__members__.values():
      key = tuple([int(d) for d in corner_vec(corner)])
      _VEC_TO_CORNER[key] = corner
  key = tuple([int(d) for d in vec])
  return _VEC_TO_CORNER[key]

def corner_orientation_vec(corner:np.array, orientation:int) -> np.array:
  rads = orientation/3.*math.pi*2
  norm = corner/np.linalg.norm(corner)
  rot = rot_about_vec(norm, rads)
  vec = np.array([0, corner[1], 0])
  vec = np.matmul(rot, vec)
  for i in range(3):
    vec[i] = int(round(vec[i]))
  return vec

def corner_orientation_from_vec(corner:np.array, orientation_vec:np.array) -> int:
  for i in range(3):
    if all(orientation_vec == corner_orientation_vec(corner, i)):
      return i
  raise RuntimeError("Invalid orientation vector.")

def _init_corner_permutations_and_deltas() -> tuple[ list[list[list[int]]], list[list[int]] ]:
  """ Permutation and orientation deltas of corners for each move in minimal cycle notation """
  all_corner_permutations = list[list[list[int]]]()
  all_corner_orientation_deltas = list[list[int]]()
  for move in Move.__members__.values():
    corners = move_corners(move)
    corner_vecs = [corner_vec(c) for c in corners]
    rot = move_rot(move)
    after_vecs = [np.matmul(rot, v) for v in corner_vecs]
    after_corners = [vec_corner(v) for v in after_vecs]
    transitions = dict((corners[i], after_corners[i]) for i in range(len(corners)))
    corner_permutations = transitions_to_cycle_notation(transitions)
    all_corner_permutations.append(corner_permutations)
    # orientation deltas
    deltas = [0]*8
    before_orientation_vecs = [corner_orientation_vec(v, 0) for v in corner_vecs]
    after_orientation_vecs = [np.matmul(rot, v) for v in before_orientation_vecs]
    after_orientations = [corner_orientation_from_vec(c, o) for c, o in \
      zip(after_vecs, after_orientation_vecs)]
    for c, o in zip(corners, after_orientations):
      deltas[c] = o
    all_corner_orientation_deltas.append(deltas)
  return all_corner_permutations, all_corner_orientation_deltas
CORNER_PERMUTATIONS, CORNER_ORIENTATION_DELTAS = _init_corner_permutations_and_deltas()

def _init_corner_symmetries() -> list[list[list[int]]]:
  """ Permutations of corners for each symetry in SYMMETRY_TRANSFORMS """
  all_permutations = list[list[list[int]]]()
  for transform in SYMMETRY_TRANSFORMS:
    corners = list(Corner.__members__.values())
    corner_vecs = [corner_vec(c) for c in corners]
    after_vecs = [np.matmul(transform, cv) for cv in corner_vecs]
    after_corners = [vec_corner(v) for v in after_vecs]
    transitions = dict((corners[i], after_corners[i]) for i in range(len(corners)))
    cycles = transitions_to_cycle_notation(transitions)
    all_permutations.append(cycles)
  return all_permutations
SYMMETRY_CORNER_PERMUTATIONS = _init_corner_symmetries()

#### Corner [End] ####


#### Cubes [Start] ####

TCubeLike = TypeVar("TCubeLike", bound="CubeLike")
class CubeLike(Hashable):
  def do(self, move:Move) -> None:
    raise NotImplementedError()
  def copy(self:TCubeLike) -> TCubeLike:
    raise NotImplementedError()
  def iter_symmetries(self:TCubeLike) -> tuple[int, TCubeLike]:
    raise NotImplementedError()
  def __hash__(self) -> int:
    raise NotImplementedError()
  def __eq__(self, other) -> bool:
    raise NotImplementedError()
  def encode(self) -> bytes:
    raise NotImplementedError()
  @staticmethod
  def decode(data:bytes) -> TCubeLike:
    raise NotImplementedError()
  @staticmethod
  def ident() -> TCubeLike:
    raise NotImplementedError()
  @staticmethod
  def valid_moves() -> list[Move]:
    raise NotImplementedError()

class G0ModG1(CubeLike):
  """
  Isomorphic to the quotient group <U, D, L, R, F, B>/<U, D, L, R, F2, B2>.
  """
  __SYMMETRY_KEYS = None
  @staticmethod
  def _symmetry_keys():
    global SYMMETRY_IDS
    if G0ModG1.__SYMMETRY_KEYS is None:
      G0ModG1.__SYMMETRY_KEYS = []
      for i, (x_flip, y_rot, x_rot, z_rot) in enumerate(SYMMETRY_IDS):
        if (x_rot == 0 or x_rot == 180) and (y_rot == 0 or y_rot == 180):
          G0ModG1.__SYMMETRY_KEYS.append(i)
    return G0ModG1.__SYMMETRY_KEYS
  def __init__(self, edge_orientations:list[bool]):
    self.edge_orientations = edge_orientations
  def _apply_edge_permutation(self, permutation, flip:bool):
    apply_permutation(permutation, self.edge_orientations, op=lambda o,_:o!=flip)
  def do(self, move:Move):
    global EDGE_PERMUTATIONS
    flip = move in [Move.FRONT, Move.FRONT_INV, Move.BACK, Move.BACK_INV]
    self._apply_edge_permutation(EDGE_PERMUTATIONS[move], flip) 
  def copy(self:TCubeLike) -> TCubeLike:
    return G0ModG1(self.edge_orientations.copy())
  def iter_symmetries(self:TCubeLike) -> tuple[int, TCubeLike]:
    global SYMMETRY_EDGE_PERMUTATIONS
    visited = set()
    for i in G0ModG1._symmetry_keys():
      permutation = SYMMETRY_EDGE_PERMUTATIONS[i]
      copy = self.copy()
      copy._apply_edge_permutation(permutation, False)
      if copy not in visited:
        visited.add(copy)
        yield i, copy
  def __hash__(self) -> int:
    return hash(tuple(self.edge_orientations))
  def __eq__(self, other) -> bool:
    return all(self.edge_orientations[i] == other.edge_orientations[i] \
      for i in range(len(self.edge_orientations)))
  def __str__(self) -> str:
    return ''.join('1' if o else '0' for o in self.edge_orientations)
  def encode(self) -> bytes:
    return bytes(str(self), 'utf-8')
  @staticmethod
  def decode(data:bytes) -> TCubeLike:
    return G0ModG1([o==ord('1') for o in data])
  @staticmethod
  def ident() -> TCubeLike:
    return G0ModG1([False]*12)
  @staticmethod
  def valid_moves() -> list[Move]:
    return list(Move.__members__.values())

class G1ModG2(CubeLike):
  """
  Isomorphic to the quotient group <U, D, L, R, F2, B2>/<U, D, L2, R2, F2, B2>.
  """
  __SYMMETRY_KEYS = None
  @staticmethod
  def _symmetry_keys():
    global SYMMETRY_IDS
    if G1ModG2.__SYMMETRY_KEYS is None:
      G1ModG2.__SYMMETRY_KEYS = []
      for i, (x_flip, y_rot, x_rot, z_rot) in enumerate(SYMMETRY_IDS):
        if (x_rot == 0 or x_rot == 180) \
            and (y_rot == 0 or y_rot == 180) \
            and (z_rot == 0 or z_rot == 180):
          G1ModG2.__SYMMETRY_KEYS.append(i)
    return G1ModG2.__SYMMETRY_KEYS
  def __init__(self, corner_orientations:list[int], edge_types:list[bool]):
    self.corner_orientations = corner_orientations
    self.edge_types = edge_types
  def do(self, move:Move) -> None:
    global EDGE_PERMUTATIONS, CORNER_PERMUTATIONS, CORNER_ORIENTATION_DELTAS
    apply_permutation(EDGE_PERMUTATIONS[move], self.edge_types) 
    apply_permutation(
      CORNER_PERMUTATIONS[move],
      self.corner_orientations,
      lambda o, b: (o+CORNER_ORIENTATION_DELTAS[move][b])%3
    ) 
  def copy(self:TCubeLike) -> TCubeLike:
    return G1ModG2(self.corner_orientations.copy(), self.edge_types.copy())
  def iter_symmetries(self:TCubeLike) -> tuple[int, TCubeLike]:
    global SYMMETRY_EDGE_PERMUTATIONS, SYMMETRY_CORNER_PERMUTATIONS, SYMMETRY_IDS
    visited = set()
    for i in G1ModG2._symmetry_keys():
      flip = SYMMETRY_IDS[i][0] == -1
      edge_permutation = SYMMETRY_EDGE_PERMUTATIONS[i]
      corner_permutation = SYMMETRY_CORNER_PERMUTATIONS[i]
      copy = self.copy()
      apply_permutation(edge_permutation, copy.edge_types) 
      apply_permutation(
        corner_permutation,
        copy.corner_orientations,
        lambda o,_: (3-o)%3 if flip else o
      ) 
      if copy not in visited:
        visited.add(copy)
        yield i, copy
  def __hash__(self) -> int:
    return hash(tuple(self.corner_orientations)) ^ hash(tuple(self.edge_types))
  def __eq__(self, other) -> bool:
    return all(self.corner_orientations[i] == other.corner_orientations[i] \
      for i in range(len(self.corner_orientations))) and \
      all(self.edge_types[i] == other.edge_types[i] \
      for i in range(len(self.edge_types)))
  def __str__(self) -> str:
    return ''.join(str(c) for c in self.corner_orientations) \
      + ',' + ''.join('1' if o else '0' for o in self.edge_types)
  def encode(self) -> bytes:
    return bytes(str(self), 'utf-8')
  @staticmethod
  def decode(data:bytes) -> TCubeLike:
    return G1ModG2(
      [c-ord('0') for c in data[:8]],
      [o==ord('1') for o in data[9:]]
    )
  @staticmethod
  def ident() -> TCubeLike:
    return G1ModG2(
      [0]*8,
      [False]*4+[True]*4+[False]*4
    )
  @staticmethod
  def valid_moves() -> list[Move]:
    moves = []
    for m in Move.__members__.values():
      d = m%3
      t = int(m/3)
      if t <= 3 or d == 2:
        moves.append(m)
    return moves

class G2ModG3(CubeLike):
  """
  Isomorphic to the quotient group <U, D, L2, R2, F2, B2>/<U2, D2, L2, R2, F2, B2>.
  """
  __SYMMETRY_KEYS = None
  @staticmethod
  def _symmetry_keys():
    global SYMMETRY_IDS
    if G2ModG3.__SYMMETRY_KEYS is None:
      G2ModG3.__SYMMETRY_KEYS = []
      for i, (x_flip, y_rot, x_rot, z_rot) in enumerate(SYMMETRY_IDS):
        if (x_rot == 0 or x_rot == 180) \
            and (y_rot == 0 or y_rot == 180) \
            and (z_rot == 0 or z_rot == 180):
          G2ModG3.__SYMMETRY_KEYS.append(i)
    return G2ModG3.__SYMMETRY_KEYS
  def __init__(self, edge_types:list[int], corner_types:list[int]):
    assert len(edge_types) == 12
    assert len(corner_types) == 8
    self.edge_types = edge_types
    self.corner_types = corner_types
  def do(self, move:Move) -> None:
    global EDGE_PERMUTATIONS, CORNER_PERMUTATIONS
    apply_permutation(EDGE_PERMUTATIONS[move], self.edge_types) 
    apply_permutation(CORNER_PERMUTATIONS[move], self.corner_types) 
  def copy(self:TCubeLike) -> TCubeLike:
    return G2ModG3(self.edge_types.copy(), self.corner_types.copy())
  def iter_symmetries(self:TCubeLike) -> tuple[int, TCubeLike]:
    global SYMMETRY_EDGE_PERMUTATIONS, SYMMETRY_CORNER_PERMUTATIONS, SYMMETRY_IDS
    visited = set()
    for i in G2ModG3._symmetry_keys():
      flip = SYMMETRY_IDS[i][0] == -1
      edge_permutation = SYMMETRY_EDGE_PERMUTATIONS[i]
      corner_permutation = SYMMETRY_CORNER_PERMUTATIONS[i]
      copy = self.copy()
      apply_permutation(edge_permutation, copy.edge_types) 
      apply_permutation(
        corner_permutation,
        copy.corner_types,
        lambda c,_: (1-c) if flip else c
      ) 
      if copy not in visited:
        visited.add(copy)
        yield i, copy
  def __hash__(self) -> int:
    return hash(tuple(self.edge_types)) ^ hash(tuple(self.corner_types))
  def __eq__(self, other) -> bool:
    return all(self.edge_types[i] == other.edge_types[i] \
      for i in range(len(self.edge_types))) and \
      all(self.corner_types[i] == other.corner_types[i] \
      for i in range(len(self.corner_types)))
  def __str__(self) -> str:
    return ''.join(str(e) for e in self.edge_types) \
      + ',' + ''.join(str(c) for c in self.corner_types)
  def encode(self) -> bytes:
    return bytes(str(self), 'utf-8')
  @staticmethod
  def decode(data:bytes) -> TCubeLike:
    return G2ModG3(
      [e-ord('0') for e in data[:12]],
      [c-ord('0') for c in data[13:]]
    )
  @staticmethod
  def ident() -> TCubeLike:
    return G2ModG3(
      [0,1]*2+[2]*4+[0,1]*2,
      [0,1]*2+[1,0]*2,
    )
  @staticmethod
  def valid_moves() -> list[Move]:
    moves = []
    for m in Move.__members__.values():
      d = m%3
      t = int(m/3)
      if t <= 1 or d == 2:
        moves.append(m)
    return moves

#### Cubes [End] ####
