import pickle
import numpy as np

# 讀取從2b_saved_data_from_raw中儲存之資料
data_path='2b'

with open(f"{data_path}/data.pickle", 'rb') as f:
    data = pickle.load(f)
with open(f"{data_path}/labels.pickle", 'rb') as f:
    label_load = pickle.load(f)

subjects=list(data.keys())
sessions=list(data[subjects[0]].keys())
channels=['EEG:C3', 'EEG:Cz', 'EEG:C4']

# 將無用channel刪除
del data[subjects[0]][sessions[0]]['EOG:ch01']
del data[subjects[0]][sessions[0]]['EOG:ch02']
del data[subjects[0]][sessions[0]]['EOG:ch03']

# 設置開始讀取秒數、訊號長度等資訊，並將每個subject的每個session分開儲存
start_second = 3
duration = 4
fs = 250
for subject in subjects:
    
    for session in sessions:
        samples=[]
        labels=[]
        for sample_idx in range(len(data[subject][session][channels[0]])):
            sample=[]
            for channel in channels:
                sample.append(data[subject][session][channel][sample_idx][start_second*fs:(start_second+duration)*fs])
            sample=np.array([ch for ch in sample])
            
            labels.append(label_load[subject][session][sample_idx]-1)
            samples.append(sample*1000000)
        print(f"{subject} {session} data num {len(samples)}")
        np.save(f"2b/data/{subject[-1]}{session[-1]}.npy",np.array(samples))
        np.save(f"2b/label/{subject[-1]}{session[-1]}.npy",np.array(labels).flatten())

print("End")