import numpy as np

# 設定資料讀取與儲存路徑
load_path='.'
save_path=''

E_label=np.load(f'{load_path}/E_label.npy',allow_pickle=True)
E_data=np.load(f'{load_path}/E.npy',allow_pickle=True)
T_label=np.load(f'{load_path}/T_label.npy',allow_pickle=True)
T_data=np.load(f'{load_path}/T.npy',allow_pickle=True)

for idx,data in enumerate(T_data,start=1):
    np.save(f"{save_path}/data/{idx}T.npy",data.squeeze())

for idx,data in enumerate(E_data,start=1):
    np.save(f"{save_path}/data/{idx}E.npy",data.squeeze())
    
for idx,data in enumerate(T_label,start=1):
    np.save(f"{save_path}/label/{idx}T.npy",data.squeeze()-1)
    
for idx,data in enumerate(E_label,start=1):
    np.save(f"{save_path}/label/{idx}E.npy",data.squeeze()-1)