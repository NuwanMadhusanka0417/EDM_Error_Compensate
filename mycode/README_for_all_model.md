## Run seklected model
```powershell
python train_compensation.py --models lstm transformer mamba --stride 20 --epochs 30 --calib-seconds 10
```

## Run All models
```powershell
python train_compensation.py --models all --test-sessions 5th_Comp --stride 200 --epochs 15 --calib-seconds 10
```


Runtime — read before launching

Measured on your CPU at batch 64: LSTM 217 ms/batch → Mamba 969 ms/batch; all 7 models together ≈ 3.5 s per batch.

Config	Batches/epoch	All-7 time
--stride 20	~1050	~60 min/epoch
--stride 100	~210	~12 min/epoch
--stride 200	~105	~6 min/epoch