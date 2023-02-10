from utils import *
# from main import *
from transforms3d.euler import mat2euler, quat2euler

def motion_model(start_time, end_time, pose_t,omega_t):
    '''
    takes the args as input and returns the pose t+1 using the exponential map
    '''
    tou_t  = end_time-start_time
    w_t = tou_t*omega_t/2
    q_next=q_mul(pose_t,q_exp(np.array([0,w_t[0],w_t[1],w_t[2]])))
    return q_next

def trajectory(ts,omega):
    '''
    This function computes the entire trajectory
    '''
    T = []
    q_t = np.array([1,0,0,0])
    T.append(q_t)
    T.append(q_t)
    for i in range(2,len(ts)):
        q_t_1 =  motion_model(ts[i-1],ts[i],q_t,omega[:,i])
        T.append(q_t_1)
        q_t = q_t_1 # may be change to set
    return np.array(T)

def quat_euler(data):
    '''
    This function converts the calibrated IMU data to euler angles using quat2euler
    '''
    eul = []
    for i in range(data.shape[0]):
        eul.append(quat2euler(data[i,:]))
    return np.array(eul)

def observation_model(q):
    '''
    This function implements the observation model for this project
    '''
    h_q = q_mul(q_inv(q),q_mul(np.array([0,0,0,1]),q))
    return h_q

if __name__ == "__main__":
    # q_t1 = motion_model(0,0.5,[1,0,1,0],[1,1,1])
    raw_imu = load_imu_data('2')
    print(f"raw {len(raw_imu)}")
    calib_imu = calibrate_imu_data(np.array(raw_imu[0]['vals']))
    print(f"calib {(calib_imu.shape)}")
    omega = get_omega(calib_imu)
    print(f"omega {omega.shape}")
    accel = get_acce(calib_imu)
    print(f"accel {accel.shape}")
    ts = raw_imu[0]['ts'].T
    print(f"ts {ts.shape}")
    tr = trajectory(ts,omega)
    print(f"tr {tr.shape}")
    tr_eul = quat_euler(tr)
    print(f"eul {tr_eul}")

