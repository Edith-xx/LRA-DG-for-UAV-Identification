import os
import numpy as np
from sklearn.model_selection import train_test_split
class LoadDataset():
    def __init__(self):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
    def load_stft_data(self, data_path, index):
        data = np.load(data_path[index], mmap_mode='r')
        print(f"Loaded data shape for {data_path[index]}: {data.shape}")
        return data
    def load_labels(self, label_path, dev_range):
        label = np.load(label_path)
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1
        sample_index_list = []
        for dev_idx in dev_range:
            num_pkt = np.count_nonzero(label == dev_idx)
            pkt_range = np.arange(0, num_pkt, dtype=int)
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
            print(f'Dev {dev_idx + 1} has {num_pkt} packets.')
        label = label[sample_index_list]
        return label
def read_train_data(data_folder='/data/czx/paper7data/SVD100/LOS',
                    label_path='/data/czx/paper7data/SVD100/label_LOS.npy',
                    dev_range=np.arange(0, 18, dtype=int),
                    class_path='/data/czx/paper7data/SVD100/class_LOS.npy',
                    class_range=np.arange(0, 6, dtype=int)):
    data_stft_all = []
    y_all = []
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
    data_files.sort()
    LoadDatasetObj = LoadDataset()
    for dev_idx in dev_range:
        print(f"Loading data for device {dev_idx + 1}...")
        data_ch0 = LoadDatasetObj.load_stft_data(data_files, dev_idx)
        data_ch0 = np.expand_dims(data_ch0, axis=1)
        y_ch0 = LoadDatasetObj.load_labels(label_path, np.array([dev_idx]))
        data_stft_all.append(data_ch0)
        y_all.append(y_ch0)
    class_ch0 = LoadDatasetObj.load_labels(class_path, class_range)
    class_ch0 = np.expand_dims(class_ch0, axis=1)
    data_stft_all = np.concatenate(data_stft_all, axis=0)
    y_all = np.concatenate(y_all)
    class_all = np.concatenate(class_ch0)
    print(f"Total data shape: {data_stft_all.shape}")
    X_train, X_val, Y_train, Y_val, Class_train, Class_val = train_test_split(data_stft_all, y_all, class_all, test_size=0.2, random_state=32, stratify=y_all)
    return X_train, X_val, Y_train, Y_val, Class_train, Class_val

def read_test_data(data_folder='/data/czx/paper7data/SVD100/NLOS',
                    label_path='/data/czx/paper7data/SVD100/label_NLOS.npy',
                    dev_range=np.arange(0, 18, dtype=int),
                    class_path='/data/czx/paper7data/SVD100/class_NLOS.npy',
                   class_range=np.arange(0, 6, dtype=int)):
    data_stft_all = []
    y_all = []
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
    data_files.sort()  # 确保文件按照正确顺序加载
    LoadDatasetObj = LoadDataset()
    for dev_idx in dev_range:
        print(f"Loading data for device {dev_idx + 1}...")
        data_ch0 = LoadDatasetObj.load_stft_data(data_files, dev_idx)
        data_ch0 = np.expand_dims(data_ch0, axis=1)
        y_ch0 = LoadDatasetObj.load_labels(label_path, np.array([dev_idx]))
        data_stft_all.append(data_ch0)
        y_all.append(y_ch0)
    data_stft_all = np.concatenate(data_stft_all, axis=0)
    class_ch0 = LoadDatasetObj.load_labels(class_path, class_range)
    class_ch0 = np.expand_dims(class_ch0, axis=1)
    y_all = np.concatenate(y_all)
    class_all = np.concatenate(class_ch0)
    X_test, _, Y_test, _, Class_test, _ = train_test_split(data_stft_all, y_all, class_all,test_size=0.5, random_state=32, stratify=y_all)
    return X_test, _, Y_test, _, Class_test, _



