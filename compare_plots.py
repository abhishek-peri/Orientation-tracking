import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler, quat2euler
from main import *
from utils import *
from motion import *

def plot_gt(seq_id,ang_id):
    data = load_vico_pkl(seq_id)
    x1 = data['data'][:,0]
    x2 = data['data'][:,1]
    x3 = data['data'][:,2]
    t = data['times'].T
    print(f"x1 {x1.shape}")
    plt.plot(data['data'][:,ang_id])
    # plt.plot(t,x2)
    # plt.plot(t,x3)
    # plt.show()

def plot_imu_calib(vals,ts):
    plt.plot(vals,ts)
    plt.show()

def quat_euler(data):
    '''
    This function converts the calibrated IMU data to euler angles using quat2euler
    '''
    eul = []
    for i in range(data.shape[0]):
        eul.append(quat2euler(data[i,:]))
    return np.array(eul)

def test_obs(Q):
    H = []
    for i in range(Q.shape[0]):
        h = observation_model(Q[i,:])
        H.append(h)
    return np.array(H)

if __name__ == "__main__":
    seq_ids = ["1","2","3","4","5","6","7","8","9"]
    if not os.path.exists('../plots'):
        os.mkdir('../plots')
    for seq_id in tqdm(seq_ids):
        raw_imu = load_imu_data([seq_id])
        print(f"raw {len(raw_imu)}")
        calib_imu = calibrate_imu_data(raw_imu[0]['vals'])
        print(f"calib {(calib_imu.shape)}")
        omega = get_omega(calib_imu)
        print(f"omega {omega.shape}")
        accel = get_acce(calib_imu)
        print(f"accel {accel.shape}")
        ts = raw_imu[0]['ts'].T
        print(f"ts {ts.shape}")

        f_tr = trajectory(ts,omega)
        f1 = open('./pickles/estimate_'+seq_id+'_9','rb')
        tr = pickle.load(f1)
        f1.close()
        print(f"tr {tr.shape}")
        tr_eul = quat_euler(tr)
        f_tr_eul = quat_euler(f_tr)
        # est_a = test_obs(tr)
        # print(est_a.shape)
        print(f"eul {tr_eul.shape}")
        plt.figure()
        plot_gt(seq_id,0)
        plt.plot(tr_eul[:,0])
        plt.plot(f_tr_eul[:,0])
        plt.legend(['VICON','OPTMIZED','MOTIONMODEL'])
        # plt.show()
        plt.savefig('../plots/'+seq_id+'_0_tr.png')

        plt.figure()
        plot_gt(seq_id,1)
        plt.plot(tr_eul[:,1])
        plt.plot(f_tr_eul[:,1])
        plt.legend(['VICON','OPTMIZED','MOTIONMODEL'])
        # plt.show()
        plt.savefig('../plots/'+seq_id+'_1_tr.png')

        plt.figure()
        plot_gt(seq_id,2)
        plt.plot(tr_eul[:,2])
        plt.plot(f_tr_eul[:,2])
        plt.legend(['VICON','OPTMIZED','MOTIONMODEL'])
        # plt.show()
        plt.savefig('../plots/'+seq_id+'_2_tr.png')
    
