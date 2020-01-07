import numpy as np

def dlist(li):
    """keeps going into a nested list printing len of each dim"""
    c = li
    keep_going = True
    while keep_going:
        try:
            print(len(c))
            c = c[0]
        except:
            keep_going=False

def stats(var):
  if type(var) == type([]):
    var = np.array(var)
  elif type(var) == type(np.array([])):
    pass #if already a numpy array, just keep going.
  else: #assume tf.var
    var = var.numpy()
     
  print('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}'.format(var.mean(), var.var(), var.min(), var.max()))