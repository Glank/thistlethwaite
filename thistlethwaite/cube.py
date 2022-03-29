from collections.abc import Hashable
from enum import IntEnum, auto
from typing import TypeVar, Generic, NewType
import numpy as np
import math

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
  x,y,z = tuple(move_vec(move))
  angle = move_angle(move)
  c = int(round(math.cos(angle*math.pi/180)))
  s = int(round(math.sin(angle*math.pi/180)))
  return np.array([
    [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
    [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
    [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
  ])

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

def orientation_from_vec(edge:np.array, orientation_vec:np.array) -> bool:
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

def transitions_to_cycle_notation(transitions) -> list[list[int]]:
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
    for cycle in permutation:
      tmp = self.edge_orientations[cycle[-1]]
      for i in range(len(cycle)-1, 0, -1):
        self.edge_orientations[cycle[i]] = \
          self.edge_orientations[cycle[i-1]] != flip
      self.edge_orientations[cycle[0]] = tmp != flip
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

class G1ModG2(CubeLike):
  """
  Isomorphic to the quotient group <U, D, L, R, F2, B2>/<U, D, L2, R2, F2, B2>.
  """
  def __init__(self, corner_orientations:list[int], edge_types:list[bool]):
    self.corner_orientations = corner_orientations
    self.edge_types = edge_types
  def do(self, move:Move) -> None:
    # TODO
    raise NotImplementedError()
  def copy(self:TCubeLike) -> TCubeLike:
    return G1ModG2(self.corner_orientations.copy(), self.edge_types.copy())
  def iter_symmetries(self:TCubeLike) -> tuple[int, TCubeLike]:
    # TODO
    raise NotImplementedError()
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
      [ord(c)-ord('0') for c in data[:8]],
      [o==ord('1') for o in data[9:]]
    )
