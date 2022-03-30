from . import cube 
import numpy as np

class AtomicDecomposition:
  def __init__(self, moves:list[cube.Move]):
    self.moves = moves
  def transformed(self, transform):
    new_moves = cube.apply_transform_to_moves(self.moves.copy(), transform)
    return AtomicDecomposition(new_moves)
  def inverse(self):
    new_moves = cube.invert(self.moves)
    return AtomicDecomposition(new_moves)
  def applied_to(self, cube):
    cp = cube.copy()
    for move in self.moves:
      cp.do(move)
    return cp

class Group:
  def __init__(self):
    self._dict = dict[cube.CubeLike, AtomicDecomposition]()
    self._debug = False
  def __contains__(self, cb:cube.CubeLike):
    for _, sym in cb.iter_symmetries():
      if sym in self._dict:
        return True
    return False
  def __getitem__(self, cb:cube.CubeLike):
    if cb in self._dict:
      return self._dict[cb]
    for sym_i, sym in cb.iter_symmetries():
      if sym in self._dict:
        transform = cube.SYMMETRY_TRANSFORMS[sym_i]
        inv_transform = np.linalg.inv(transform)
        # S_i(cb) = _dict[sym]
        # cb = S_i^-1(_dict[sym])
        return self._dict[sym].transformed(inv_transform)
    raise KeyError(f'{cb} not in group')
  def __setitem__(self, key, value):
    assert isinstance(value, AtomicDecomposition)
    self._dict[key] = value
  def __len__(self):
    return len(self._dict)
  def keys(self):
    return self._dict.keys()

class GroupBuilder:
  def __init__(self, root:cube.CubeLike, available_moves:list[cube.Move] = None):
    self.group = Group()
    self.group[root] = AtomicDecomposition([])
    self.unexplored = set([root])
    if available_moves is None:
      available_moves = root.__class__.valid_moves()
    self.available_moves = available_moves
  def _build_step(self):
    if not self.unexplored:
      raise RuntimeError('Already fully built.')
    cur = self.unexplored.pop()
    cur_node = self.group[cur]
    for move in self.available_moves:
      nxt = cur.copy()
      nxt.do(move)
      if nxt not in self.group:
        new_node = AtomicDecomposition(cur_node.moves+[move])
        self.group[nxt] = new_node
        self.unexplored.add(nxt)
    return bool(self.unexplored)
  def build(self):
    if self.unexplored:
      while self._build_step():
        pass
  def __len__(self):
    return len(self.group)
