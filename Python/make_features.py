# make_features.py
import os
import numpy as np
import pandas as pd

IN_NPZ = r'C:\tremor_project\data\windows\windows.npz'
OUT_FEAT = r'C:\tremor_project\data\windows\features.npz'
OUT_NORM = r'C:\tremor_project\model\norm_stats.npz'

def skew(x):
    m = np.mean(x); s = np.std(x) + 1e-12
    return np.mean(((x-m)/s)**3)

def compute_feats(window): # window shape (WIN,3)
    ax = window[:,0]; ay = window[:,1]; az = window[:,2]
    mag = np.sqrt(ax*ax + ay*ay + az*az)
    feats = [
        np.mean(ax), np.mean(ay), np.mean(az),
        np.std(ax), np.std(ay), np.std(az),
        skew(ax), skew(ay), skew(az),
        np.mean(mag), np.std(mag), np.max(mag), np.median(np.abs(mag - np.median(mag)))
    ]
    return np.array(feats, dtype=np.float32)

data = np.load(IN_NPZ)
Xw = data['X']  # (n_samples, WIN, 3)
y = data['y']
n = Xw.shape[0]
F = []
for i in range(n):
    F.append(compute_feats(Xw[i]))
F = np.stack(F).astype(np.float32)
# winsorize small extremes
lo = np.percentile(F,1,axis=0); hi = np.percentile(F,99,axis=0)
for j in range(F.shape[1]):
    F[:,j] = np.clip(F[:,j], lo[j], hi[j])

mean = np.mean(F, axis=0).astype(np.float32)
std = np.std(F, axis=0).astype(np.float32)
std[std==0] = 1.0

np.savez_compressed(OUT_FEAT, X=F, y=y)
np.savez_compressed(OUT_NORM, mean=mean, std=std)
print("Saved features:", F.shape, "norm stats:", mean.shape)
