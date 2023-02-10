import numpy as np
from transforms3d.quaternions import quat2mat
import sys
import pickle
from tqdm import tqdm,trange
import math
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def pix_sph():
    '''
    This function converts the pixel cords to spherical cords
    returns:
    meshgrid of size (2*240*320)
    '''
    a1 = np.linspace(0,60,320)*(np.pi/180)
    a2 = np.linspace(0,45,240)*(np.pi/180)
    a3 = np.array(np.meshgrid(a1,a2))
    return a3

def spherical_rec(cords):
    '''
    This function converts the spherical coordinates to spherical coordinates
    args:
    cords: (lambda,phi) is the convention here
    returns:
    rectangular coordinates with RGB mapping intact
    '''
    rec = np.array([np.cos(cords[:,:,1])*np.cos(cords[:,:,0]),-np.cos(cords[:,:,1])*np.sin(cords[:,:,0]),-np.sin(cords[:,:,1])])
    return rec.transpose(1,2,0)

def rec_world_test(cords,R):
    W = []
    cords = cords.reshape(-1,cords.shape[2])
    for i in trange(R.shape[0]):
        W.append((R[i,:,:]@cords.T).reshape(240,320,3))
    return np.array(W)

def rec_world_sph(cords):
    '''
    This converts the rectangular coordinates to spherical coordinates
    args:
    cords: [x,y,z] with size: N*3*240*320
    returns:
    sph : cords with order [rou,theta,phi]
    '''
    temp = np.sqrt(cords[0,:,:]**2+cords[1,:,:]**2+cords[2,:,:]**2)
    sph = np.array([np.arcsin(-cords[2,:,:]/temp),np.arctan2(cords[1,:,:],cords[0,:,:]),temp])
    return sph

def spherical_cyl(cords):
    '''
    This fucntion converts the speherical coordinatest to the cylindrical coordinatese
    args:
    cords: (lambda_w,phi_world,1) assert the depth is 1
    '''
    cyl = np.array([cords[:,:,:,0]+np.pi/2,cords[:,:,:,1]+np.pi])
    return cyl

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

def quat_mat(data):
    '''
    This function converts the calibrated IMU data to euler angles using quat2euler
    '''
    eul = []
    for i in range(data.shape[0]):
        eul.append(quat2mat(data[i,:]))
    return np.array(eul)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id', type=str, required=True)
    args = parser.parse_args()

    dataset=args.seq_id
    cfile = "../data/testset/cam/cam" + dataset + ".p"
    ifile = "../data/testset/imu/imuRaw" + dataset + ".p"

    camd = read_data(cfile)
    imud = read_data(ifile)

    ts_cam = camd['ts']
    ts_vico = imud['ts']

    print(f"ts_cam {ts_cam.shape}")
    print(f"ts_vico {ts_vico.shape}")

    rgb = camd['cam']

    fp = open('./test_pickles/estimate_'+dataset+'_9','rb')
    est = pickle.load(fp)
    fp.close()
    est = quat_mat(est)
    est = est.transpose(1,2,0)
    tr = est

    vico_ts = ts_vico.T
    sph1 = pix_sph()
    sph1 = sph1.transpose(1,2,0)

    plt.figure()
    rec1 = spherical_rec(sph1)
    pan_rgb = np.zeros((1080,1920,3))
    for i in trange(ts_cam.shape[1]):
        ts = (ts_cam.T)[i]
        near_idx = np.argmax(vico_ts>ts)
        try:
            R = tr[:,:,near_idx]
        except:
            continue

        rec_world = (((R@rec1.reshape(-1,3).T)).reshape(3,240,320)) #.transpose(1,2,0)

        sph_world = rec_world_sph(rec_world)
        
        
        sph_world[0,:,:] = (sph_world[0,:,:] + np.pi/2)*(1080/np.pi)
        sph_world[1,:,:] = (sph_world[1,:,:] + np.pi)*(1920/(2*np.pi))
        
        cyl_world = sph_world.astype(int)
        
        pan_rgb[cyl_world[0,:,:],cyl_world[1,:,:],:] = rgb[:,:,:,i]
    plt.imshow(pan_rgb.astype(int))
    plt.show()