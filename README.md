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

Submission<br>
>>code<br>
      >>>  main.py<br>
      >>>  test.py<br>
      >>>  ...<br>
      >>>  ...<br>
      >>>  ...<br>
    data<br>
     >>   ECE276A_P1_2023<br>
       >>>     trainset<br>
          >>>>      cam<br>
             >>>>>       cam1.p<br>
               >>>>>    cam2.p<br>
                   >>>>> cam8.p<br>
                    >>>>> cam9.p<br>
                >>>imu<br>
                >>>vicon<br>
            testset<br>
                cam<br>
                    cam10.p<br>
                    cam11.p<br>
                imu<br>
                    imuRaw10.p<br>
                    imuRaw11.p<br>

# Instructions to run

Run all the steps in order

1. cd code<br>

2. python load_data.py<br>

#change seq_id accordingly and run for all the datasets from '1' to '9' before proceding to next step. <br>

3. python main.py --seq_id '1'<br>

#the below is to generate the plots. <br>
4. python compare_plots.py <br>

#the below is to optimize on test data and generate pickle files. <br>
5. python test.py<br>

#the below is to generate the test orientation plots. <br>
6. python test_plots.py<br>

#the below is to generate panormas for the train data. <br>
7. python pano_train.py<br>

#the below is to generate panorama for the test data. <br>
8. python pano_test.py<br>
