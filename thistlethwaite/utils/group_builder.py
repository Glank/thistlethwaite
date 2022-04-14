from . import cube 
import numpy as np
import sqlite3
from textwrap import dedent
from tqdm import tqdm

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
  def encode(self) -> bytes:
    return bytes([int(m) for m in self.moves])
  @staticmethod
  def decode(data:bytes):
    moves = [cube.Move(i) for i in data]
    return AtomicDecomposition(moves)
  def __eq__(self, other) -> bool:
    if len(self.moves) != len(other.moves):
      return False
    return all(sm == om for sm, om in zip(self.moves, other.moves))

class Group:
  def __contains__(self, cb:cube.CubeLike) -> bool:
    """ Returns true if the given cube is in the group. """
    raise NotImplementedError()
  def __getitem__(self, cb:cube.CubeLike) -> AtomicDecomposition:
    """ Returns an atomic decomposition of the given element of the group. """
    raise NotImplementedError()
  def __setitem__(self, key:cube.CubeLike, value:AtomicDecomposition) -> None:
    """ Sets the atomic decomposition for a given element. """
    raise NotImplementedError()
  def __len__(self) -> int:
    """ Returns the number of elements in the group, mod symmetries. """
    raise NotImplementedError()
  def keys(self):
    """ Iterates over all elements of the group, mod symmetries. """
    raise NotImplementedError()

class SqliteGroup(Group):
  def __init__(self, filename, table, key_clazz=None):
    self.filename = filename
    self.table = table
    self.con = sqlite3.connect(self.filename)
    self.key_clazz = key_clazz 
    query = dedent(f'''
      CREATE TABLE IF NOT EXISTS
      {table} (
        key BLOB PRIMARY KEY,
        moves BLOB
      )
    ''')
    self.con.execute(query)
  def __contains__(self, cb:cube.CubeLike) -> bool:
    keys = []
    for _, sym in cb.iter_symmetries():
      keys.append(sym.encode())
    query = f'SELECT COUNT(*) FROM {self.table} WHERE '+\
      ' OR '.join('key = ?' for _ in range(len(keys)))
    cur = self.con.cursor()
    cur.execute(query, tuple(keys))
    count = cur.fetchall()[0][0]
    return count > 0
  def __getitem__(self, cb:cube.CubeLike) -> AtomicDecomposition:
    keys = []
    sym_idxs = []
    for sym_idx, sym in cb.iter_symmetries():
      keys.append(sym.encode())
      sym_idxs.append(sym_idx)
    query = f'SELECT key, moves FROM {self.table} WHERE ' + \
      ' OR '.join('key = ?' for _ in range(len(keys))) + \
      ' LIMIT 1'
    cur = self.con.execute(query, tuple(keys))
    rows = cur.fetchall()
    if not rows:
      raise KeyError(f'{cb} not in group {self.table}')
    key, atomic_encoded = rows[0]
    sym_idx = sym_idxs[keys.index(key)]
    atomic = AtomicDecomposition.decode(atomic_encoded) 
    transform = cube.SYMMETRY_TRANSFORMS[sym_idx]
    inv_transform = np.linalg.inv(transform)
    # S_i(cb) = _dict[sym]
    # cb = S_i^-1(_dict[sym])
    return atomic.transformed(inv_transform)
  def __setitem__(self, key:cube.CubeLike, value:AtomicDecomposition) -> None:
    if self.key_clazz is None:
      self.key_clazz = key.__class__
    self.con.execute(
      f'INSERT OR REPLACE INTO {self.table} VALUES (?,?)',
      (key.encode(), value.encode())
    )
    self.con.commit()
  def add_all(self, cube_value_iter) -> None:
    def fmted_iter(cube_value_iter):
      for cb, decomp in cube_value_iter:
        yield cb.encode(), decomp.encode()
    self.con.executemany(
      f'INSERT OR REPLACE INTO {self.table} VALUES (?,?)',
      fmted_iter(cube_value_iter)
    )
    self.con.commit()
  def __len__(self) -> int:
    cur = self.con.execute(f'SELECT COUNT(*) FROM {self.table}')
    count = cur.fetchall()[0][0]
    return count
  def keys(self):
    cur = self.con.cursor()
    for row in cur.execute(f'SELECT key FROM {self.table}'):
      yield self.key_clazz.decode(row[0])

class MemoryGroup(Group):
  def __init__(self):
    self._dict = dict[cube.CubeLike, AtomicDecomposition]()
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
  def save_to(self, sqlite_group:SqliteGroup) -> None:
    sqlite_group.add_all(tqdm(self._dict.items(), total=len(self._dict)))

class GroupBuilder:
  def __init__(self, root:cube.CubeLike, group:Group = None):
    if group is None:
      self.group = MemoryGroup()
    self.group[root] = AtomicDecomposition([])
    self.unexplored = set([root])
    self.available_moves = root.__class__.valid_moves()
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
  def build(self, expected_size=None):
    if self.unexplored:
      if expected_size is not None:
        with tqdm(total=expected_size) as pbar:
          max_size = 0
          while self._build_step():
            new_size = len(self.group)
            delta = new_size-max_size
            if delta>0:
              pbar.update(delta)
              max_size = new_size
      else:
        while self._build_step():
          pass
  def __len__(self):
    return len(self.group)
