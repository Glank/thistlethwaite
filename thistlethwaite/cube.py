from collection.abc import Hashable
from enum import IntEnum
from typing import TypeVar, Generic, NewType
import numpy as np

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
  d = int(move)/3
  return np.array([
    [0, 1, 0],
    [0, -1, 0],
    [-1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, -1]
  ][d])

def move_rot(move:Move):
  x,y,z = tuple(move_vec(move))
  type_ = int(move)%3
  c, s : int, int
  if type_ == 0:
    # 90 deg clockwise
    c = 0
    s = -1
  elif type_ == 1:
    # 90 deg widershins
    c = 0
    s = 1
  elif type_ == 2:
    # 180 deg
    c = -1
    s = 0
  return np.array([
    [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
    [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
    [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
  ])

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
  d = int(move)/3
  return [
    [UR, UF, UL, UB], #U
    [DR, DF, DL, DB], #D
    [UL, FL, DL, BL], #L
    [UR, FR, DR, BR], #R
    [UF, FR, DF, FL], #F
    [UB, BR, DB, BL], #B
  ][d]

def edge_vec(edge:Edge):
  e = int(edge)
  y = e/4
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

_VEC_TO_EDGE = dict[tuple[int,int,int],Edge]
def vec_edge(vec):
  global _VEC_TO_EDGE
  if not _VEC_TO_EDGE:
    for edge in Edge.__members__.values():
      v = edge_vec(edge)
      v = tuple([int(i) for i in v])
      _VEC_TO_EDGE[v] = edge
  vec = tuple([int(i) for i in vec])
  return _VEC_TO_EDGE[vec]

EDGE_PERMUTATIONS = init_edge_permutations()
def init_edge_permutations() -> list[list[list[int]]]:
  """ Permutation of edges for each move in minimal cycle notation """
  all_edge_permutations = list[list[list[int]]]
  for move in Move.__members__.values():
    edges = move_edges(move)
    edge_vecs = [edge_vec(e) for e in affected_edges]
    rot = move_rot(move)
    after_vecs = [np.matmul(rot, v) for v in edge_vecs]
    after_edges = [vec_edge(v) for v in after_vecs]
    transitions = dict((edges[i], after_edges[i]) for i in range(len(edges)))
    visited = set()
    edge_permutations = list[list[int]]
    while len(visited) < len(edges):
      cycle = list[int]
      # get first unvisited
      start = [e for e in edges if e not in visted][0]
      visited.add(start)
      cycle.append(start)
      edge = transitions[start]
      visited.add(edge)
      while edge != start:
        cycle.append(edge)
        edge = transitions[start]
        visited.add(edge)
      edge_permutations.append(cycle)
    all_edge_permutations.append(edge_permutation)

class CubeLike(Hashable):
  def do(self, move:Move) -> None:
    raise NotImplementedError()
  def copy(self) -> CubeLike
    raise NotImplementedError()
  def __hash__(self) -> int
    raise NotImplementedError()
  def __eq__(self, other) -> bool
    raise NotImplementedError()

class G0ModG1(CubeLike):
  """
  Isomorphic to the quotient group <U, D, L, R, F, B>/<U, D, L, R, F2, B2>.
  """
  def __init__(self, edge_orientations:list[bool]):
    self.edge_orientations = edge_orientations
  def do(self, move:Move):
    global EDGE_PERMUTATIONS
    flip = move in [FRONT, FRONT_INV, BACK, BACK_INV]
    permutation = EDGE_PERMUTATIONS[move]
    for cycle in permutation:
      tmp = self.edge_orientations[cycle[-1]]
      for i in range(len(cycle)-1, 0, -1):
        self.edge_orientations[cycle[i]] = \
          self.edge_orientations[cycle[i-1]] != flip
      self.edge_orientations[cycle[0]] = tmp != flip
  def copy(self):
    return G0ModG1(self.edge_orientations.copy())
  def __hash__(self) -> int:
    return hash(self.edge_orientations)
  def __eq__(self, other) -> bool:
    return all(self.edge_orientations[i] == other.edge_orientations[i] \
      for i in range(len(self.edge_orientations))

#TODO: test
