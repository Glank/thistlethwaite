from . import cube 
from . import group_builder
gb = group_builder

class Solver:
  def __init__(self, database):
    self.database = database
  def solve(self, cb:cube.G0):
    coset_translators = [
      cube.G0.get_g0modg1,
      cube.G0.get_g1modg2,
      cube.G0.get_g2modg3,
      cube.G0.get_g3modg4
    ]
    moves = []
    cb = cb.copy()
    for i, cotran in enumerate(coset_translators):
      coset_cube = cotran(cb)
      clazz = coset_cube.__class__
      table = clazz.__name__.lower()
      group = gb.SqliteGroup(self.database, table, key_clazz=clazz)
      print(coset_cube)
      decomp = group[coset_cube]
      inverse = decomp.inverse()
      for move in inverse.moves:
        moves.append(move)
        print(move)
        cb.do(move)
      print(cb)
      for j in range(i+1):
        coset = coset_translators[j](cb)
        assert coset == coset.__class__.ident()
    return moves
