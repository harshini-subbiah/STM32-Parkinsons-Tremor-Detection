# make_windows.py
import os, glob
import numpy as np
import pandas as pd

RAW_DIR = r'C:\tremor_project\data\raw'
OUT_DIR = r'C:\tremor_project\data\windows'
os.makedirs(OUT_DIR, exist_ok=True)

FS = 200
WIN = FS * 1
STEP = FS // 2

classes = {'normal':0,'mild':1,'moderate':2,'severe':3}

X_windows = []
y_windows = []

for cls, label in classes.items():
    files = sorted(glob.glob(os.path.join(RAW_DIR, f"{cls}_*.csv")))
    for f in files:
        # read robustly: accept either 3 columns or single column "ax,ay,az"
        try:
            df = pd.read_csv(f, header=None)
            if df.shape[1] == 1:
                # split single column by comma
                df = df.iloc[:,0].astype(str).str.strip().str.strip('"').str.split(',', expand=True)
            df = df.iloc[:, :3]  # first 3 columns
            df.columns = ['ax','ay','az']
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            data = df[['ax','ay','az']].values.astype(np.float32)
        except Exception as e:
            print("Skip file (parse error):", f, e)
            continue

        if len(data) < WIN:
            continue

        for start in range(0, len(data) - WIN + 1, STEP):
            win = data[start:start+WIN]  # shape (WIN,3)
            X_windows.append(win)
            y_windows.append(label)

X = np.stack(X_windows).astype(np.float32)  # (n_samples, WIN, 3)
y = np.array(y_windows, dtype=np.int32)
np.savez_compressed(os.path.join(OUT_DIR, 'windows.npz'), X=X, y=y)
print("Saved windows:", X.shape, "labels:", y.shape)
