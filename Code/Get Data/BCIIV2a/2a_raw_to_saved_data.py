import numpy as np
import scipy.io as sio
import glob as glob

def get_data(subject,training,path, reject_artifact=False):
    '''	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets
    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
        if False, load testing data

    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
    class_return 	numpy matrix 	size = NO_valid_trial
    '''
    NO_channels = 22
    NO_tests = 6*48 	
    Window_Length = 7*250 

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests,NO_channels,Window_Length))

    NO_valid_trial = 0
    
    # 設定要讀取的試training還是evaluation資料
    if training:
        a = sio.loadmat(path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(path+'A0'+str(subject)+'E.mat')
    
    # 讀取特定subjectT/E的資料訊息
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X 		= a_data3[0]
        a_trial 	= a_data3[1]
        a_y 		= a_data3[2]
        a_fs 		= a_data3[3]
        a_classes 	= a_data3[4]
        a_artifacts = a_data3[5]
        a_gender 	= a_data3[6]
        a_age 		= a_data3[7]

        # 針對每個train進行處理
        for trial in range(0,a_trial.size):
   
            if reject_artifact:
                if(a_artifacts[trial]==0):# 是artifact的則不管
                    data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
                    class_return[NO_valid_trial] = int(a_y[trial])
                    NO_valid_trial +=1        
            else:
                data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial +=1        


    return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

def prepare_features(path,subject, training, start_t, end_t, reject_artifact=False):

    fs = 250 
    t1 = int(start_t*fs)
    t2 = int(end_t*fs)

    T = t2-t1
    X_train, y_train = get_data(subject,training,path,reject_artifact)

    # prepare training data 	
    N_tr,N_ch,_ =X_train.shape 
    X_train = X_train[:,:,t1:t2].reshape(N_tr,1,N_ch,T)

    return X_train,y_train

# 讀取全部subject資料
def prepare_all_data_feature (path, start_t, end_t,reject_artifact): 

    big_X_train, big_y_train, big_X_test, big_y_test = [None]*9, [None]*9, [None]*9, [None]*9
    # 逐一讀取subject資料並儲存在同一陣列中
    for subject in range (0,9):
        big_X_train[subject], big_y_train[subject] = prepare_features(path,subject+1, True, start_t, end_t, reject_artifact=reject_artifact)
        big_X_test[subject], big_y_test[subject] = prepare_features(path,subject+1, False, start_t, end_t, reject_artifact=reject_artifact)
    
    big_X_train, big_y_train, big_X_test, big_y_test= np.array(big_X_train), np.array(big_y_train), np.array(big_X_test), np.array(big_y_test)

    return big_X_train, big_y_train, big_X_test, big_y_test

if __name__=='__main__':

    # 設定資料路徑及資料範圍
    path='./'
    start_t=2
    end_t=6
    reject_artifact=True

    big_X_train, big_y_train, big_X_test, big_y_test= prepare_all_data_feature (path, start_t, end_t,reject_artifact)
    
    np.save('T',big_X_train)
    np.save('T_label',big_y_train)
    np.save('E',big_X_test)
    np.save('E_label',big_y_test)

    print()