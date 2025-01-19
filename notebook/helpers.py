import numpy as np
from dataset import read_one_data

def load_npy(name="TS_6_4",onhost=False,onhos_threshold=0.7):
    valid_dir = '../input/czii-cryo-et-object-identification/train'

    if onhost:
        image = read_one_data(name, static_dir=f'{valid_dir}/static/ExperimentRuns')
        label = np.load(f"../input/mask/generated{name}_mask.npz")
        label = label['arr_0']
        d,z,x,y = label.shape

        onhot = np.zeros((z,x,y))
        for i in range(d):
            if i == 0:
                continue
            onhot = np.where(label[i] > onhos_threshold, i, onhot)
        
            
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {onhot.shape}")
        return onhot,image


    else:
        image = read_one_data(name, static_dir=f'{valid_dir}/static/ExperimentRuns')
        label = np.load(f"../input/mask/generated{name}_mask.npz")

        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label['arr_0'].shape}")
        return label['arr_0'],image
    