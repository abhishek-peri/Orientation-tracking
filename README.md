# The codes are organized in the following way

main.py is the code that runs the optimization for train data. <br>
test.py is the code that runs the optimization for test data. <br>
compare_plots.py generates the plots for the orientations. <br>
load_data.py generates the pickle files for euler angles of the VICON data used for plots. <br>
utils.py holds all the functions needed to run the scripts.   <br>
motion.py has the functions for motion model and the observation model. <br>
pano_train.py is to generate panorama for train data. <br>
pano_test.py is to generate panorama for test data. <br>
test_plots.py is to generate orientation plots for test_data <br>

# The structure of the repo should reflect as below

Submission
    code
        main.py
        test.py
        ...
        ...
        ...
    data
        ECE276A_P1_2023
            trainset
                cam
                    cam1.p
                    cam2.p
                    cam8.p
                    cam9.p
                imu
                vicon
            testset
                cam
                    cam10.p
                    cam11.p
                imu
                    imuRaw10.p
                    imuRaw11.p

# Instructions to run

Run all the steps in order

1. cd code

2. python load_data.py

#change seq_id accordingly and run for all the datasets from '1' to '9' before proceding to next step. 

3. python main.py --seq_id '1'

#the below is to generate the plots. 
4. python compare_plots.py 

#the below is to optimize on test data and generate pickle files. 
5. python test.py

#the below is to generate the test orientation plots. 
6. python test_plots.py

#the below is to generate panormas for the train data. 
7. python pano_train.py

#the below is to generate panorama for the test data. 
8. python pano_test.py
