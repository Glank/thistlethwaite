import thistlethwaite.utils.cube as cube
import thistlethwaite.utils.group_builder as gb

def test_build_g0modg1():
  ident = cube.G0ModG1([False]*12)
  builder = gb.GroupBuilder(ident)
  builder.build()
  
  cb = ident.copy()
  moves = [
    cube.Move.FRONT, 
    cube.Move.LEFT,
    cube.Move.BACK,
    cube.Move.UP,
    cube.Move.DOWN_INV,
    cube.Move.RIGHT_2,
  ]
  for move in moves:
    cb.do(move)
  assert cb in builder.group
  decomp = builder.group[cb]
  inv_decomp = decomp.inverse()
  undone = cb.copy()
  for move in inv_decomp.moves:
    undone.do(move) 
  assert undone == ident

  #print('Length: {}'.format(len(builder.group)))

  # Tested on an actual cube
  cb = cube.G0ModG1([
    False, True, True, True,
    True, False, True, True,
    True, True, False, False
  ])
  builder.group._debug = True
  decomp = builder.group[cb]
  reconstructed = ident.copy()
  for move in decomp.moves:
    reconstructed.do(move)
  assert reconstructed == cb
  inv_decomp = decomp.inverse()
  undone = cb.copy()
  for move in inv_decomp.moves:
    move_s = str(cube.Move(move))
    undone.do(move) 
    #print(f'{move_s} {undone}')
  assert undone == ident

def test_decode_atomic_decomposition():
  atomic = gb.AtomicDecomposition([cube.Move.UP, cube.Move.DOWN_2])
  encoded = atomic.encode()
  decoded = gb.AtomicDecomposition.decode(encoded)
  assert decoded == atomic

def test_sqlite_group():
  group = gb.SqliteGroup(':memory:', 'g0modg1')
  ident = cube.G0ModG1.ident()
  assert ident not in group
  group[ident] = gb.AtomicDecomposition([])
  assert ident in group
  assert group[ident] == gb.AtomicDecomposition([])
  
  # Test symmetries are in group without needing to add all of them
  front = ident.copy()
  front.do(cube.Move.FRONT)
  group[front] = gb.AtomicDecomposition([cube.Move.FRONT])
  assert front in group
  back = ident.copy()
  back.do(cube.Move.BACK)
  back_atomic = gb.AtomicDecomposition([cube.Move.BACK])
  assert back in group
  assert group[back] == back_atomic

  # Test other group methods
  assert len(group) == 2
  for key in group.keys():
    assert key in [ident, front]

def test_build_g1modg2():
  builder = gb.GroupBuilder(cube.G1ModG2.ident())
  print("Starting build...")
  builder.build()
  print(len(builder))

def main(cmdline_params):
  test_sqlite_group()
  test_decode_atomic_decomposition()
  test_build_g0modg1()
  #test_build_g1modg2()
  pass
