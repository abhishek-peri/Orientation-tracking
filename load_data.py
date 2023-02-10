import pickle
import sys
import time 
import matplotlib.pyplot as plt
from utils import *
from transforms3d.euler import mat2euler
import numpy as np
from numpy.linalg import det
import pickle

def tic():
  return time.time()

def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

if not os.path.exists('../data/pickles'):
  os.mkdir('../data/pickles')

seqs = ["1","2","3","4","5","6","7","8","9"]
for j in seqs:
  ang = {'data':[]}
  dataset=j
  cfile = "../data/ECE276A_P1_2023/trainset/cam/cam" + dataset + ".p"
  ifile = "../data/ECE276A_P1_2023/trainset/imu/imuRaw" + dataset + ".p"
  vfile = "../data/ECE276A_P1_2023/trainset/vicon/viconRot" + dataset + ".p"

  ts = tic()
  # camd = read_data(cfile)
  imud = read_data(ifile)
  vicd = read_data(vfile)
  for i in range(vicd['rots'].shape[2]):
    ang['data'].append(mat2euler(vicd['rots'][:,:,i]))
  ang['data'] = np.asarray(ang['data'])
  ang['times'] = vicd['ts']

  fp =  open('../data/pickles/ang'+j,'wb')
    
  pickle.dump(ang,fp)

  fp.close()
toc(ts,"Data import")





