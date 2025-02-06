import numpy as np
from dataset import read_one_data
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from helpers import *
from dataset import *
from model256 import *
from czii_helper import *
import cc3d
from scipy.optimize import linear_sum_assignment
import math


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

MOLECULES = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']

def generate_mask(dic, mask_size, default_size=-1):
    mask = np.load(f"../input/mask/train_label_{dic}.npy")

    return mask

def print_slices(vol,mask,dim=4,idx=1,depth=16):

    if dim == 3:
        plt.title(f"example of volume and mask at depth {depth}")
        plt.imshow(vol[depth, :, :], cmap='gray')
        plt.imshow(mask[depth, :, :], cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.show()

    if dim == 4:
        plt.title(f"example of volume and mask at depth {depth} of {idx}")
        plt.imshow(vol[idx,depth, :, :], cmap='gray')
        plt.imshow(mask[idx,depth, :, :], cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.show()

        
def print_volume_slices(vol, dim=4, idx=1, depth=16):
    """
    Displays slices of a volume at a specified depth.

    Parameters:
        vol: numpy array
            The volume to display.
        dim: int
            The number of dimensions of the volume (3 or 4).
        idx: int
            The index to use if the volume has 4 dimensions.
        depth: int
            The depth to display along the specified axis.
    """
    if dim == 3:
        plt.title(f"Volume slice at depth {depth}")
        plt.imshow(vol[depth, :, :], cmap='gray')
        plt.colorbar()
        plt.show()

    elif dim == 4:
        plt.title(f"Volume slice at depth {depth} of index {idx}")
        plt.imshow(vol[idx, depth, :, :], cmap='gray')
        plt.colorbar()
        plt.show()
    else:
        raise ValueError("Invalid dimension. 'dim' must be 3 or 4.")
    

def rotate_3d_volume_x_axis(
    volume: np.ndarray,
    anglex: float,
    angley: float,
    anglez: float
) -> np.ndarray:
    """
    3次元配列 (184, 184, 630) を受け取り:
      1) まず (1000, 1000, 1000) の大きな配列に中心配置する
      2) x軸回りに anglex度, y軸回りに angley度, z軸回りに anglez度 の順に回転する
      3) 回転後の (1000, 1000, 1000) の配列を返す

    ※ OpenCV(cv2) の2Dアフィン変換をスライス毎に適用した簡易的な方法です。
       計算コストが非常に大きい点に注意してください。

    Parameters
    ----------
    volume : np.ndarray, shape==(184,184,630)
        回転させたい3次元ボリューム (1チャネル想定)
    anglex : float
        x軸回りの回転角度（度数法）
    angley : float
        y軸回りの回転角度（度数法）
    anglez : float
        z軸回りの回転角度（度数法）

    Returns
    -------
    big_volume : np.ndarray, shape==(1000,1000,1000)
        3軸回転後の大きな3次元配列
    """
    # --- 1) 入力 shape をチェック ---
    X, Y, Z = volume.shape
    if (X, Y, Z) != (184, 630, 630):
        raise ValueError(f"入力のshapeが (184, 630, 630) ではありません: {volume.shape}")

    # --- 2) (1000, 1000, 1000) のボリュームを用意 (ゼロ埋め) ---
    big_volume = np.zeros((1000, 1000, 1000), dtype=volume.dtype)

    # 元ボリュームの中心 (x_center, y_center, z_center)
    x_center_in = X // 2  # = 92
    y_center_in = Y // 2  # = 92
    z_center_in = Z // 2  # = 315

    # 大きいボリュームの中心 (500, 500, 500) に合わせるためのオフセット
    # x,y,zそれぞれ (500 - 中心)
    x_offset = 500 - x_center_in  # 500 - 92 = 408
    y_offset = 500 - y_center_in  # 500 - 92 = 408
    z_offset = 500 - z_center_in  # 500 - 315=185

    # --- 3) (184,184,630) を (1000,1000,1000) の中央へコピー ---
    #     big_volume[x+408, y+408, z+185] = volume[x,y,z]
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                bx = x + x_offset
                by = y + y_offset
                bz = z + z_offset
                big_volume[bx, by, bz] = volume[x, y, z]

    # 3つの回転をまとめて行う。順番は (x) → (y) → (z) の順
    # ------------------------------------------------------------------
    # ★ 非常に大きな計算負荷 ★
    # 以下の3ステップそれぞれで 1000スライス × 1000x1000 画素のwarpAffineを行います。

    # -------- A) x軸回りに anglex 度 回転 --------
    if abs(anglex) > 1e-7:
        center_2d = (500, 500)  # 2D回転の中心座標
        Mx = cv2.getRotationMatrix2D(center_2d, anglex, 1.0)
        for x in range(1000):
            # big_volume[x,:,:] は形状 (1000,1000) => (row, col)=(y,z)
            slice_2d = big_volume[x, :, :]
            rotated_2d = cv2.warpAffine(
                slice_2d,
                Mx,
                (1000, 1000),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            big_volume[x, :, :] = rotated_2d

    # -------- B) y軸回りに angley 度 回転 --------
    if abs(angley) > 1e-7:
        center_2d = (500, 500)
        My = cv2.getRotationMatrix2D(center_2d, angley, 1.0)
        for y in range(1000):
            # big_volume[:,y,:] は形状 (1000,1000)
            #   row → x方向 (0~999), col → z方向 (0~999)
            slice_2d = big_volume[:, y, :].copy()  # メモリ連続でないのでcopy()推奨
            rotated_2d = cv2.warpAffine(
                slice_2d,
                My,
                (1000, 1000),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            # 回転後を再び big_volume[:, y, :] に書き戻す
            big_volume[:, y, :] = rotated_2d

    # -------- C) z軸回りに anglez 度 回転 --------
    if abs(anglez) > 1e-7:
        center_2d = (500, 500)
        Mz = cv2.getRotationMatrix2D(center_2d, anglez, 1.0)
        for z in range(1000):
            # big_volume[:,:,z] は形状 (1000,1000)
            #   row → x方向, col → y方向
            slice_2d = big_volume[:, :, z].copy()
            rotated_2d = cv2.warpAffine(
                slice_2d,
                Mz,
                (1000, 1000),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            big_volume[:, :, z] = rotated_2d

    return big_volume

def rotate_3d_mask_x_axis(
    volume: np.ndarray,
    anglex: float,
    angley: float,
    anglez: float
) -> np.ndarray:
    """
    3次元配列 (184, 184, 630) を受け取り:
      1) まず (1000, 1000, 1000) の大きな配列に中心配置する
      2) x軸回りに anglex度, y軸回りに angley度, z軸回りに anglez度 の順に回転する
      3) 回転後の (1000, 1000, 1000) の配列を返す

    ※ OpenCV(cv2) の2Dアフィン変換をスライス毎に適用した簡易的な方法です。
       計算コストが非常に大きい点に注意してください。

    Parameters
    ----------
    volume : np.ndarray, shape==(184,184,630)
        回転させたい3次元ボリューム (1チャネル想定)
    anglex : float
        x軸回りの回転角度（度数法）
    angley : float
        y軸回りの回転角度（度数法）
    anglez : float
        z軸回りの回転角度（度数法）

    Returns
    -------
    big_volume : np.ndarray, shape==(1000,1000,1000)
        3軸回転後の大きな3次元配列
    """
    # --- 1) 入力 shape をチェック ---
    X, Y, Z = volume.shape
    if (X, Y, Z) != (184, 630, 630):
        raise ValueError(f"入力のshapeが (184, 630, 630) ではありません: {volume.shape}")

    # --- 2) (1000, 1000, 1000) のボリュームを用意 (ゼロ埋め) ---
    big_volume = np.zeros((1000, 1000, 1000), dtype=volume.dtype)

    # 元ボリュームの中心 (x_center, y_center, z_center)
    x_center_in = X // 2  # = 92
    y_center_in = Y // 2  # = 92
    z_center_in = Z // 2  # = 315

    # 大きいボリュームの中心 (500, 500, 500) に合わせるためのオフセット
    # x,y,zそれぞれ (500 - 中心)
    x_offset = 500 - x_center_in  # 500 - 92 = 408
    y_offset = 500 - y_center_in  # 500 - 92 = 408
    z_offset = 500 - z_center_in  # 500 - 315=185

    # --- 3) (184,184,630) を (1000,1000,1000) の中央へコピー ---
    #     big_volume[x+408, y+408, z+185] = volume[x,y,z]
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                bx = x + x_offset
                by = y + y_offset
                bz = z + z_offset
                big_volume[bx, by, bz] = volume[x, y, z]

    # 3つの回転をまとめて行う。順番は (x) → (y) → (z) の順
    # ------------------------------------------------------------------
    # ★ 非常に大きな計算負荷 ★
    # 以下の3ステップそれぞれで 1000スライス × 1000x1000 画素のwarpAffineを行います。

    # -------- A) x軸回りに anglex 度 回転 --------
    if abs(anglex) > 1e-7:
        center_2d = (500, 500)  # 2D回転の中心座標
        Mx = cv2.getRotationMatrix2D(center_2d, anglex, 1.0)
        for x in range(1000):
            # big_volume[x,:,:] は形状 (1000,1000) => (row, col)=(y,z)
            slice_2d = big_volume[x, :, :]
            rotated_2d = cv2.warpAffine(
                slice_2d,
                Mx,
                (1000, 1000),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            big_volume[x, :, :] = rotated_2d

    # -------- B) y軸回りに angley 度 回転 --------
    if abs(angley) > 1e-7:
        center_2d = (500, 500)
        My = cv2.getRotationMatrix2D(center_2d, angley, 1.0)
        for y in range(1000):
            # big_volume[:,y,:] は形状 (1000,1000)
            #   row → x方向 (0~999), col → z方向 (0~999)
            slice_2d = big_volume[:, y, :].copy()  # メモリ連続でないのでcopy()推奨
            rotated_2d = cv2.warpAffine(
                slice_2d,
                My,
                (1000, 1000),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            # 回転後を再び big_volume[:, y, :] に書き戻す
            big_volume[:, y, :] = rotated_2d

    # -------- C) z軸回りに anglez 度 回転 --------
    if abs(anglez) > 1e-7:
        center_2d = (500, 500)
        Mz = cv2.getRotationMatrix2D(center_2d, anglez, 1.0)
        for z in range(1000):
            # big_volume[:,:,z] は形状 (1000,1000)
            #   row → x方向, col → y方向
            slice_2d = big_volume[:, :, z].copy()
            rotated_2d = cv2.warpAffine(
                slice_2d,
                Mz,
                (1000, 1000),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            big_volume[:, :, z] = rotated_2d

    return big_volume

def slice_big_vol(big_vol, big_mask, x_bias=0, y_bias=0, z_bias=0):
    vol_slice = big_vol[484+z_bias:516+z_bias, 180+y_bias:820+y_bias, 180+x_bias:820+x_bias]
    mask_slice = big_mask[484+z_bias:516+z_bias, 180+y_bias:820+y_bias, 180+x_bias:820+x_bias]
    return vol_slice, mask_slice

import random
import numpy as np

def process_rotation_and_slicing(data, mask_data, anglex, angley, anglez, zz_range, x_offset=92):
    """
    Rotates a 3D volume and its corresponding mask, then slices them at specified intervals.
    
    Parameters:
        data: ndarray
            The 3D volume data to rotate and slice.
        mask_data: ndarray
            The 3D mask data to rotate and slice.
        anglex: int
            Rotation angle around the x-axis.
        angley: int
            Rotation angle around the y-axis.
        anglez: int
            Rotation angle around the z-axis.
        zz_range: range
            The range of z-coordinates to iterate over for slicing.
        x_offset: int, optional
            Base x-coordinate offset (default is 92).
    
    Returns:
        Tuple of ndarrays:
            vol_slices: ndarray
                The slices of the volume data in shape (スライス数, 32, 640, 640).
            mask_slices: ndarray
                The slices of the mask data in shape (スライス数, 32, 640, 640).
    """
    # Rotate the 3D volume and mask
    big_vol = rotate_3d_volume_x_axis(data, anglex, angley, anglez)
    big_mask = rotate_3d_mask_x_axis(mask_data, anglex, angley, anglez)

    # Lists to store slices
    vol_slices = []
    mask_slices = []

    # Iterate through the specified z-range and slice
    for zz in zz_range:
        z_bias = random.randint(-100, 100)
        y_bias = random.randint(-100, 100)
        x_bias = x_offset - zz
        print(z_bias, y_bias, x_bias)

        # Slice the rotated volume and mask
        vol_slice, mask_slice = slice_big_vol(big_vol, big_mask, z_bias, y_bias, x_bias)
        
        # Append slices to the lists
        vol_slices.append(vol_slice)
        mask_slices.append(mask_slice)
        #print_slices(vol_slice, mask_slice, dim=3, depth=16)
    
    # Convert lists to numpy arrays
    vol_slices = np.array(vol_slices)  # Shape: (スライス数, 32, 640, 640)
    mask_slices = np.array(mask_slices)  # Shape: (スライス数, 32, 640, 640)

    return vol_slices, mask_slices

def probability_to_location(probability,cfg):
    _,D,H,W = probability.shape

    location={}
    for p in PARTICLE:
        p = dotdict(p)
        l = p.label

        cc, P = cc3d.connected_components(probability[l]>cfg.threshold[p.name], return_N=True)
        stats = cc3d.statistics(cc)
        zyx=stats['centroids'][1:]*10.012444
        xyz = np.ascontiguousarray(zyx[:,::-1]) 
        location[p.name]=xyz
        '''
            j=1
            z,y,x = np.where(cc==j)
            z=z.mean()
            y=y.mean()
            x=x.mean()
            print([x,y,z])
        '''
    return location

def location_to_df(location):
    location_df = []
    for p in PARTICLE:
        p = dotdict(p)
        xyz = location[p.name]
        if len(xyz)>0:
            df = pd.DataFrame(data=xyz, columns=['x','y','z'])
            #df.loc[:,'particle_type']= p.name
            df.insert(loc=0, column='particle_type', value=p.name)
            location_df.append(df)
    if location_df:
        location_df = pd.concat(location_df)
    else:
        location_df = pd.DataFrame(columns=['particle_type', 'x', 'y', 'z'])
        print("location_df is empty, ")
    return location_df

def do_one_eval(truth, predict, threshold):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric


def compute_lb(submit_df, overlay_dir):
    valid_id = list(submit_df['experiment'].unique())
    print(valid_id)

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(id, overlay_dir) #=f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df['experiment'] == id]
        for p in PARTICLE:
            p = dotdict(p)
            print('\r', id, p.name, end='', flush=True)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius* 0.5)
            eval_df.append(dotdict(
                id=id, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            ))
    print('')
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
    gb.loc[:, 'precision'] = gb['hit'] / gb['P']
    gb.loc[:, 'precision'] = gb['precision'].fillna(0)
    gb.loc[:, 'recall'] = gb['hit'] / gb['T']
    gb.loc[:, 'recall'] = gb['recall'].fillna(0)
    gb.loc[:, 'f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall'])
    gb.loc[:, 'f-beta4'] = gb['f-beta4'].fillna(0)

    gb = gb.sort_values('particle_type').reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, 'weight'] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()
    return gb, lb_score

def compute_lb_nodebug(submit_df, overlay_dir):
    valid_id = list(submit_df['experiment'].unique())
    #print(valid_id)

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(id, overlay_dir) #=f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df['experiment'] == id]
        for p in PARTICLE:
            p = dotdict(p)
            #print('\r', id, p.name, end='', flush=True)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius* 0.5)
            eval_df.append(dotdict(
                id=id, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            ))
    #print('')
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
    gb.loc[:, 'precision'] = gb['hit'] / gb['P']
    gb.loc[:, 'precision'] = gb['precision'].fillna(0)
    gb.loc[:, 'recall'] = gb['hit'] / gb['T']
    gb.loc[:, 'recall'] = gb['recall'].fillna(0)
    gb.loc[:, 'f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall'])
    gb.loc[:, 'f-beta4'] = gb['f-beta4'].fillna(0)

    gb = gb.sort_values('particle_type').reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, 'weight'] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()
    return gb, lb_score

def evaluate_cv(net, test_id, cfg, valid_dir, device,mask_size):
    """
    検証データセットでの精度、再現率、F1スコアを計算。

    Parameters:
        net: モデル
        test_id: 検証用のデータIDリスト
        cfg: コンフィグオブジェクト
        valid_dir: 検証データディレクトリ
        device: 実行デバイス (CPUまたはCUDA)

    Returns:
        gb: pd.DataFrame, 各粒子タイプのスコアデータ
        lb_score: float, 総合スコア
    """
    net.eval()
    submit_df = []

    for i, id in enumerate(test_id):
        #print(f"Evaluating CV for ID {id} ({i+1}/{len(test_id)})")
        volume = read_one_data(id, static_dir=f'{valid_dir}/static/ExperimentRuns')
        y = read_one_truth(id, f'{valid_dir}/overlay/ExperimentRuns')
        mask = generate_mask(y, mask_size)
        
        D, H, W = volume.shape

        probability = np.zeros((7, D, H, W), dtype=np.float32)
        count = np.zeros((7, D, H, W), dtype=np.float32)
        pad_volume = np.pad(volume, [[0, 0], [0, 640 - H], [0, 640 - W]], mode='constant', constant_values=0)
        pad_mask = np.pad(mask, [[0, 0], [0, 640 - H], [0, 640 - W]], mode='constant', constant_values=0)

        num_slice = 32
        mask_loss =[]
        zz = list(range(0, D - num_slice, num_slice // 2)) + [D - num_slice]
        for z in zz:
            image = pad_volume[z:z + num_slice]
            mask = pad_mask[z:z + num_slice]
            batch = {
                'image': torch.from_numpy(image).unsqueeze(0).to(device).float(),
                'mask': torch.from_numpy(mask).unsqueeze(0).to(device).long(),
            }
            with torch.amp.autocast(device_type="cuda", enabled=True):
                with torch.no_grad():
                    output = net(batch)

            #valid loglossを表示
            mask_loss.append(output['mask_loss'].item())

            prob = output['particle'][0].cpu().numpy()
            probability[:, z:z + num_slice] += prob[:, :, :H, :W]
            count[:, z:z + num_slice] += 1

        probability = probability / (count + 0.0001)
        location = probability_to_location(probability, cfg)
        df = location_to_df(location)
        df.insert(loc=0, column='experiment', value=id)
        submit_df.append(df)
        #print(f"valid_loss={np.mean(mask_loss)}")

    submit_df = pd.concat(submit_df)
    if len(submit_df) == 0:
        print("particle_type not found, skipping groupby.")
        gb = pd.DataFrame(columns=['particle_type', 'P', 'T', 'hit', 'miss', 'fp', 'precision', 'recall', 'f-beta4', 'weight'])
        lb_score = 0
        net.eval()
    else:
        #print(compute_lb(submit_df, f'{valid_dir}/overlay/ExperimentRuns'))
        gb, lb_score = compute_lb_nodebug(submit_df, f'{valid_dir}/overlay/ExperimentRuns')
        net.eval()

    return gb, lb_score,mask_loss,probability


def display_value_ratios(mask):
    """
    Display the percentage of each integer value in the given tensor.

    Parameters:
        mask (torch.Tensor): Input tensor.
    """
    # Flatten the tensor to make counting easier
    flattened = mask.flatten()

    # Count occurrences of each value
    unique_values, counts = torch.unique(flattened, return_counts=True)

    # Calculate total number of elements
    total_elements = flattened.numel()

    # Display the percentage for each value in a single line
    ratios = [f"Value {value.item()}: {((count.item() / total_elements) * 100):.2f}%" for value, count in zip(unique_values, counts)]
    print("Value Ratios: " + ", ".join(ratios))



def mask_check(test_id,mask_size,cfg):
    valid_dir = '../input/czii-cryo-et-object-identification/train'
    mask_dir = '../input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/'
    y =read_one_truth(test_id, mask_dir)
    masks = generate_mask(y, mask_size)
    masks = np.expand_dims(masks, axis=0)  # バッチ次元を追加

    predects = np.zeros((7, 184, 630, 630), dtype=np.float32)
    np.put_along_axis(predects, masks, 1, axis=0)
    predects = np.pad(predects, [[0, 0], [0, 0], [0, 640 - 630], [0, 640 - 630]], mode='constant', constant_values=0)
    predects = np.expand_dims(predects, axis=0)  # バッチ次元を追加
    predects = torch.tensor(predects).float()

    mask = np.pad(masks, [[0, 0], [0, 0], [0, 640 - 630], [0, 640 - 630]], mode='constant', constant_values=0)
    mask = torch.tensor(mask).long()

    #loss = F.cross_entropy(predects, mask, reduction='mean')
    location = probability_to_location(predects[0].numpy(), cfg)
    df = location_to_df(location)
    df.insert(loc=0, column='experiment', value=test_id)
    gb,lb_score=compute_lb(df, f'{valid_dir}/overlay/ExperimentRuns')
    return gb,lb_score


def evaluate_256cv(net, test_id, cfg, valid_dir, device,mask_size):
    """
    検証データセットでの精度、再現率、F1スコアを計算。

    Parameters:
        net: モデル
        test_id: 検証用のデータIDリスト
        cfg: コンフィグオブジェクト
        valid_dir: 検証データディレクトリ
        device: 実行デバイス (CPUまたはCUDA)

    Returns:
        gb: pd.DataFrame, 各粒子タイプのスコアデータ
        lb_score: float, 総合スコア
    """
    net.eval()
    submit_df = []

    for i, id in enumerate(test_id):
        volume = read_one_data(id, static_dir=f'{valid_dir}/static/ExperimentRuns')
        #y = read_one_truth(id, f'{valid_dir}/overlay/ExperimentRuns')
        mask = generate_mask(id, mask_size)
            
        D, H, W = volume.shape

        probability = np.zeros((7, D, H, W), dtype=np.float32)
        count = np.zeros((7, D, H, W), dtype=np.float32)
        meta_volume = volume
        meta_mask = mask

        mask_loss =[]
        num_slice = 64  # z軸のサイズ
        patch_size = 256  # x, y軸のサイズ

        # Hanning窓を用いた重みマスクを事前に作成（shape: (num_slice, patch_size, patch_size)）
        w_z = np.hanning(num_slice)      # z軸方向の重み
        w_x = np.hanning(patch_size)     # x軸方向の重み
        w_y = np.hanning(patch_size)     # y軸方向の重み
        patch_weight = w_z[:, None, None] * w_x[None, :, None] * w_y[None, None, :]

        zz = list(range(0, D - num_slice, num_slice // 2)) + [D - num_slice]
        for z in zz:
            for seg_x, seg_y in [(i % 3, i // 3) for i in range(9)]:

                image = meta_volume[z:z + num_slice, seg_x*187:seg_x*187+patch_size, seg_y*187:seg_y*187+patch_size]
                mask   = meta_mask[z:z + num_slice, seg_x*187:seg_x*187+patch_size, seg_y*187:seg_y*187+patch_size]

                batch = {
                    'image': torch.from_numpy(image).unsqueeze(0).to(device).float(),
                    'mask': torch.from_numpy(mask).unsqueeze(0).to(device).long(),
                }
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    with torch.no_grad():
                        output = net(batch)

                mask_loss.append(output['mask_loss'].item())

                # 出力された確率は shape (7, num_slice, patch_size, patch_size)
                prob = output['particle'][0].cpu().numpy()

                # 重みマスクを各チャネルにブロードキャストして乗算
                weighted_prob = prob * patch_weight[None, :, :, :]

                # 加算時に重みを反映（境界部分は小さい値になる）
                probability[:, z:z + num_slice, seg_x*187:seg_x*187+patch_size, seg_y*187:seg_y*187+patch_size] += weighted_prob
                count[:, z:z + num_slice, seg_x*187:seg_x*187+patch_size, seg_y*187:seg_y*187+patch_size] += patch_weight[None, :, :, :]
        probability = probability / (count + 0.0001)
        location = probability_to_location(probability, cfg)
        df = location_to_df(location)
        df.insert(loc=0, column='experiment', value=id)
        submit_df.append(df)
        #print(f"valid_loss={np.mean(mask_loss)}")

    submit_df = pd.concat(submit_df)
    if len(submit_df) == 0:
        print("particle_type not found, skipping groupby.")
        gb = pd.DataFrame(columns=['particle_type', 'P', 'T', 'hit', 'miss', 'fp', 'precision', 'recall', 'f-beta4', 'weight'])
        lb_score = 0
        net.eval()
    else:
        #print(compute_lb(submit_df, f'{valid_dir}/overlay/ExperimentRuns'))
        gb, lb_score = compute_lb_nodebug(submit_df, f'{valid_dir}/overlay/ExperimentRuns')
        net.eval()

    return gb, lb_score,mask_loss,probability