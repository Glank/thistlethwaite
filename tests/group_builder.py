import thistlethwaite.cube as cube
import thistlethwaite.group_builder as gb

def test_build():
  ident = cube.G0ModG1([False]*12)
  builder = gb.GroupBuilder(ident, list(cube.Move.__members__.values()))
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

def main(cmdline_params):
  test_build()
  pass
