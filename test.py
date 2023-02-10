from utils import *
from motion import *
import jax
from tqdm import tqdm,trange

def test_cost(Q,tr,accel):
    l = 0
    for i in range(3):
        # print(f"re {q_mul(q_inv(Q[i+1,:]),tr[i+1,:])}")
        l = l+0.5*((q_norm(2*q_log(q_mul(q_inv(Q[i+1,:]),tr[i+1,:])+1e-8))))
    return l

def cost_function(Q,W,accel):
    '''
    This method constructs the cost function that needs to be optimized
    args:
    accel: [n,4]
    '''
    t1 = 0
    for i in trange(Q.shape[0]):
        if(i<1):
            # t1 = t1+ 0.5*(np.square(q_norm(2*q_log(q_mul(q_inv(Q[i+1,:]),tr[i+1,:])+1e-8))))
            t1 = t1+ 0.5*(np.square(q_norm(2*q_log(q_mul(q_inv(Q[i+1,:]),q_mul(Q[i,:],W[i,:]))+1e-8))))
            continue
        if(i==Q.shape[0]-1):
            h = observation_model(Q[i,:])
            t1 = t1+0.5*(np.square(q_norm(accel[i,:]-h)))
            break
        h = observation_model(Q[i,:])
        # print(f"h {h.shape}")
        # t1 = t1+ 0.5*(np.square(q_norm(2*q_log(q_mul(q_inv(Q[i+1,:]),tr[i+1,:])+1e-8))) + np.square(q_norm(accel[i,:]-h)))
        t1 = t1+ 0.5*(np.square(q_norm(2*q_log(q_mul(q_inv(Q[i+1,:]),q_mul(Q[i,:],W[i,:]))+1e-8))) + np.square(q_norm(accel[i,:]-h)))
    return t1


def optimization(iterations):
    '''
    This function performs the optimization steps based on the number of iterations
    '''
    alpha = 0.1
    seq_ids = ['10','11']
    for seq_id in tqdm(seq_ids):
        print(f"starting the optimization for dataset {seq_id}")
        raw_imu = load_imu_data(seq_id,True)
        print(f"raw {len(raw_imu)}")
        calib_imu = calibrate_imu_data(raw_imu[0]['vals'])
        print(f"calib {(calib_imu.shape)}")
        omega = get_omega(calib_imu)
        print(f"omega {omega.shape}")
        accel = get_acce(calib_imu)
        print(f"accel {accel.shape}")
        ts = raw_imu[0]['ts'].T
        print(f"ts {ts.shape}")
        W = get_W(ts,omega.T)
        print(f"W {W.shape}")
        tr = trajectory(ts,omega)
        print(f"tr {tr.shape}")
        tr_eul = quat_euler(tr)
        print(f"eul {tr_eul.shape}")
        Q = tr.astype(np.float32)
        print(type(Q))
        for i in trange(iterations):
            gq = jax.jacrev(cost_function)(Q,W,accel)
            loss = cost_function(Q,W,accel)
            # print(he)
            num = Q-alpha*(gq)
            Q = (num)/(np.linalg.norm(num,axis=1)+1e-6).reshape(-1,1)
            # print(Q)
            if(i%3==0):
                print(loss)
                fp = open('./test_pickles/estimate_'+seq_id[0]+'_'+str(i),'wb')
                pickle.dump(Q,fp)
                fp.close()
    return True

if __name__ == "__main__":
    if not os.path.exists('./test_pickles'):
        os.mkdir('./test_pickles')
    optimization(10)