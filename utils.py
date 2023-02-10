# Contains all the helper function to construct the motion, observation models and panorama visualizations
import math
import jax.numpy as np
from jax.numpy.linalg import norm
import pickle
import sys
import os
# import jax

def q_exp(q):
    '''
    This function computes the exponential map in the quaternion kinematics model
    '''
    qs = q[0]
    # qv = np.asarray(q[1:])
    qv = q[1:]
    t = qv*np.sin(norm(qv))/norm(qv)
    q_e = math.exp(qs)*np.array([np.cos(norm(qv)),t[0],t[1],t[2]])
    return q_e

def q_log(q):
    '''
    This function computes the log map for the optimization step in the cost function
    '''
    qs = q[0]
    qv = q[1:]
    if (norm(qv)!=0):
        t = qv*np.arccos((qs/q_norm(q+1e-6)))/norm(qv)
        q_l = np.array([np.log(q_norm(q+1e-6)),t[0],t[1],t[2]])
    else:
        # print("hit")
        t = [0,0,0]
        q_l = np.array([np.log(abs(qs)+1e-6),t[0],t[1],t[2]])
    return q_l

def q_norm(q):
    '''
    This function calculates the norm of the quaternion
    '''
    qs = q[0]
    qv = q[1:]
    n_q = np.power((qs**2 + qv.T@qv),0.5)
    return n_q

def q_mul(q,p):
    '''
    This function computes the product of 2 quaternions
    '''
    qs = q[0]
    qv = q[1:]
    ps = p[0]
    pv = p[1:]
    t = qs*pv+ps*qv+np.cross(qv,pv)
    mul = np.array([qs*ps-qv.T@pv,t[0],t[1],t[2]])
    return mul

def q_rot(q,v):
    '''
    This function computes the transformation by rotation of q to a vector as in observation model
    '''
    new_v = 0
    return new_v

def q_inv(q):
    '''
    This function computes the inverse of a quaternion
    '''
    qs = q[0]
    qv = q[1:]
    q_bar = np.array([qs,-qv[0],-qv[1],-qv[2]])
    q_i = q_bar/np.power(q_norm(q)+1e-6,2)
    return q_i

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

def load_data(dataset):
    '''
    This function reads the data and returns a dict contaning all the data
    args:
    dataset: the seq_id of the input dataset
    returns:
    data: a dict with rgb images, IMU readings, VICON data as the keys
    '''
    data = {}
    cfile = "../data/ECE276A_P1_2023/trainset/cam/cam" + dataset + ".p"
    ifile = "../data/ECE276A_P1_2023/trainset/imu/imuRaw" + dataset + ".p"
    vfile = "../data/ECE276A_P1_2023/trainset/vicon/viconRot" + dataset + ".p"

    camd = read_data(cfile)
    imud = read_data(ifile)
    vicd = read_data(vfile)
    data['cam'] = camd
    data['imu'] = imud
    data['vico'] = vicd
    return data

def load_imu_data(seq,test=False):
    '''
    This function loads entire IMU only data for bias calculation
    '''
    data = []
    for item in seq:
        if(test):
            ifile = "../data/testset/imu/imuRaw" + item + ".p"
        else:
            ifile = "../data/ECE276A_P1_2023/trainset/imu/imuRaw" + item + ".p"
        imud = read_data(ifile)
        # imud['vals'][0,:] = -1*(imud['vals'][0,:])
        data.append(imud)
    return data

def load_vico_pkl(seq_id):
    '''
    This fucntion returns the ground-truth euler angles for a given seq_id for plotting
    '''
    fp = open('../data/pickles/ang'+seq_id,'rb')
    data = pickle.load(fp)
    fp.close()
    return data

def calibrate_imu_data(data,samples = 500):
    '''
    This function computes the calibrated IMU data given the raw readings
    '''
    cal_data = []
    data = data.astype(np.float64)
    data[0,:] = -1.0*(data[0,:])
    data[1,:] = -1.0*(data[1,:])
    for i in range(data.shape[0]):
        # print(data[i,:samples])
        bias = np.sum(data[i,:samples])/samples
        print(f"bias for {i} is {bias}")
        if(i<3):
            scale = 3300/(1023*300) # change this
            print(f"scale for {i} is {scale} case 1")
        else:
            scale = 3300/(1023*3.3)*(math.pi/180)
            print(f"scale for {i} is {scale} case 2") # change this
        if(i==2):
            temp = (data[i,:]-bias)*scale+1
        else:
            temp = (data[i,:]-bias)*scale
        cal_data.append(temp)
    cal_data = np.array(cal_data)
    # cal_data[2,:] = cal_data.at[2,:].set(cal_data[2,:] + 1)
    return cal_data

def grad_cal(cost):
    '''
    This function computes the gradient given the previous iteration values of Q matrix
    '''

    gq = 0
    return gq

def get_acce(data):
    acce = np.zeros((4,data.shape[1]))
    acce = acce.at[1:4,:].set(data[:3,:])
    acce = acce.T
    return acce

def get_omega(data):
    omega = data[3:,:]
    omega = omega[[1,2,0],:]
    return omega

def get_timestamps(data):
    ts = data['ts']
    return ts

def get_W(ts,omega):
    W = []
    for i in range(0,len(ts)-1):
        wt = (ts[i+1]-ts[i])*omega[i,:]
        W.append(q_exp(np.array([0,wt[0],wt[1],wt[2]])))
    return np.array(W)

# sens for gyroscope = 3.3
# sens for accelo = 300 mV/g
def scale_factor(sens):
    '''
    This function computes the scale_factor for the IMU given the sensitivity from datasheets as input
    '''
    s_f = 3300/(1023*sens)*(math.pi/180)
    return s_f

def bias(values):
    '''
    This function computes the bias given the values at which the robot is observed to be stationary
    '''
    b = sum(values)/len(values)
    return b


if __name__=="__main__":
    out = q_mul(np.array([1,2,3,4]),np.array([23,4,5,6]))
    exp_out = q_exp([0,2,3,1])
    norm_q = q_norm(exp_out)
    log_q = q_log([0.7071,0,0.7071,0])
    print(f"mul {out}")