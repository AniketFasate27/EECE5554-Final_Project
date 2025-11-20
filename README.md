# EECE5554-Final_Project
#DC_MPU: is a arduino firmware script for the ESP32 and MPU6250 interfacing. By using FreeRTOS and Moving Average on one core of the esp and other core we are gather the data in real time. 


#data_saving.py
Python script data saving is to record the raw data form the controller and save it in CSV. It collect 3 axis of acc and 3 axis of gyro data and save it in CSV.  First we collect the healthy systems data for almost 30 min. 
Then we will collect the faults  imbalance fault trail for 30 min for the training of the ML algorithm. 
similarly we have same script is responsible for the collecting for the motor missaligment trail and bearing fault. 


feature_extraaction.py 
Using these we will extract the featres data from the CSV files we gather for different faults. 

ml_learning.py 
After the feature extraction we will train the ml algorithm Random Forest on the data we are collected. 

realtime_predications.py 
responsible for the data read the real time data from port and predict the fault on the basis on the ML algorithm. 



