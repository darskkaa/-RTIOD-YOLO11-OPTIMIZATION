# Pre-Flight Checklist - RTIOD YOLO11 Training

Run these checks before starting the 3-5 hour training session.

---

## ✅ System Requirements

### GPU Check
```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x                |
|-----------------------------------------------------------------------------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 5090       On   | 00000000:01:00.0 Off |                  N/A |
| 32GB VRAM available
```

✅ **Pass:** RTX 5090 detected with 32GB VRAM  
❌ **Fail:** No GPU or insufficient VRAM → Reduce batch size to 32

---

### CUDA Check
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA Available: True
CUDA Version: 12.1
GPU: NVIDIA GeForce RTX 5090
```

✅ **Pass:** CUDA available and RTX 5090 detected  
❌ **Fail:** CUDA not available → Install CUDA 12.x

---

### PyTorch Version Check
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

**Expected output:**
```
PyTorch: 2.0.0 or higher
```

✅ **Pass:** PyTorch 2.0+  
❌ **Fail:** PyTorch < 2.0 → Run `pip install --upgrade torch torchvision`

---

### Ultralytics Check
```bash
python -c "from ultralytics import YOLO; import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
```

**Expected output:**
```
Ultralytics: 8.3.0 or higher
```

✅ **Pass:** Ultralytics 8.3.0+  
❌ **Fail:** Older version → Run `pip install --upgrade ultralytics`

---

## ✅ Configuration Verification

### Config File Check
```bash
python -c "import yaml; config = yaml.safe_load(open('config/config.yaml')); print(f'Model: {config[\"modelCheckpoint\"]}'); print(f'ImgSize: {config[\"imgSize\"]}'); print(f'Batch: {config[\"batchSize\"]}'); print(f'Epochs: {config[\"epochs\"]}'); print(f'LR: {config[\"lr\"]}'); print(f'Optimizer: {config[\"optimizer\"]}')"
```

**Expected output:**
```
Model: yolov11x.pt
ImgSize: 640
Batch: 64
Epochs: 200
LR: 0.001
Optimizer: SGD
```

✅ **Pass:** All parameters correct  
❌ **Fail:** Wrong parameters → Check config/config.yaml

---

### Data Paths Check
```bash
python -c "import yaml; data = yaml.safe_load(open('data/data.yaml')); print(f'Train: {data[\"train\"]}'); print(f'Val: {data[\"val\"]}'); print(f'Classes: {data[\"nc\"]}'); print(f'Names: {data[\"names\"]}')"
```

**Expected output:**
```
Train: ./images/train
Val: ./images/val
Classes: 4
Names: ['person', 'bicycle', 'motorcycle', 'vehicle']
```

✅ **Pass:** Data paths correct  
❌ **Fail:** Wrong paths → Check data/data.yaml

---

### Training Images Check
```bash
python -c "import os; train_count = len([f for f in os.listdir('data/images/train') if f.endswith(('.jpg', '.png'))]); val_count = len([f for f in os.listdir('data/images/val') if f.endswith(('.jpg', '.png'))]); print(f'Training images: {train_count}'); print(f'Validation images: {val_count}')"
```

**Expected output:**
```
Training images: 5000-10000 (typical for LTDv2)
Validation images: 1000-2000
```

✅ **Pass:** Images found  
❌ **Fail:** No images → Download LTDv2 dataset

---

## ✅ Disk Space Check

### Available Space
```bash
python -c "import shutil; total, used, free = shutil.disk_usage('.'); print(f'Free space: {free // (2**30)} GB')"
```

**Expected:**
```
Free space: >10 GB
```

✅ **Pass:** Sufficient space (need ~5GB for checkpoints)  
❌ **Fail:** <10GB free → Free up disk space

---

## ✅ Memory Check

### RAM Check
```bash
python -c "import psutil; mem = psutil.virtual_memory(); print(f'Total RAM: {mem.total // (2**30)} GB'); print(f'Available RAM: {mem.available // (2**30)} GB')"
```

**Expected:**
```
Total RAM: 32+ GB
Available RAM: 16+ GB
```

✅ **Pass:** Sufficient RAM  
⚠️ **Warning:** <16GB available → Close other applications

---

## ✅ Quick Test Run

### 1-Epoch Test
```bash
python -c "
from ultralytics import YOLO
import torch
import yaml

# Load config
config = yaml.safe_load(open('config/config.yaml'))

# Check GPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load model
model = YOLO(config['modelCheckpoint'])
print(f'Model loaded: {config[\"modelCheckpoint\"]}')

# Quick test (1 epoch)
print('Running 1-epoch test...')
results = model.train(
    data='data/data.yaml',
    epochs=1,
    imgsz=config['imgSize'],
    batch=config['batchSize'],
    device=device,
    verbose=False,
)
print('✅ Test passed! Ready for full training.')
"
```

**Expected:**
- Model downloads (if first time)
- Training starts
- 1 epoch completes in ~1.5-2.0 seconds
- No errors

✅ **Pass:** Test epoch completes successfully  
❌ **Fail:** Errors → Check error message

---

## ✅ Final Checklist

Before running `python train.py`, verify:

- [ ] ✅ RTX 5090 detected with 32GB VRAM
- [ ] ✅ CUDA 12.x available
- [ ] ✅ PyTorch 2.0+ installed
- [ ] ✅ Ultralytics 8.3.0+ installed
- [ ] ✅ config.yaml has yolov11x.pt, imgSize=640, batch=64
- [ ] ✅ data.yaml points to correct image paths
- [ ] ✅ Training images found in data/images/train/
- [ ] ✅ Validation images found in data/images/val/
- [ ] ✅ >10GB disk space available
- [ ] ✅ >16GB RAM available
- [ ] ✅ 1-epoch test passed

---

## 🚀 Ready to Train!

If all checks pass, run:

```bash
python train.py
```

**Expected:**
- Training starts immediately
- GPU utilization 95-100%
- VRAM usage 20-24GB
- ~1.5-2.0 seconds per epoch
- Total time: 3-5 hours for 200 epochs

**Monitor progress:**
```bash
# In another terminal
watch -n 5 nvidia-smi  # GPU monitoring
tail -f runs/thermal_detection/yolo11x_5090_optimized/results.csv  # Training metrics
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
**Error:** `CUDA out of memory`

**Solution 1:** Reduce batch size
```yaml
# In config/config.yaml
batchSize: 32  # Reduce from 64 to 32
```

**Solution 2:** Reduce resolution
```yaml
# In config/config.yaml
imgSize: 512  # Reduce from 640 to 512
```

---

### Model Not Found
**Error:** `Model 'yolov11x.pt' not found`

**Solution:** Model will auto-download on first run. Ensure internet connection.

---

### Data Not Found
**Error:** `Dataset 'data/data.yaml' not found`

**Solution:** Check paths in data/data.yaml match actual directory structure.

---

### CUDA Not Available
**Error:** `CUDA not available`

**Solution:** 
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---

### Slow Training
**Issue:** >5 seconds per epoch

**Possible causes:**
1. CPU training (no GPU detected)
2. Low GPU utilization (<80%)
3. Disk I/O bottleneck
4. Background processes using GPU

**Solution:**
1. Verify GPU: `nvidia-smi`
2. Close other GPU applications
3. Use SSD for data storage
4. Reduce workers: `workers=4` in train.py

---

## 📊 Expected Training Output

```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
  1/200   20.5G    1.234     0.567     1.123        512   640
  2/200   20.5G    1.156     0.523     1.089        512   640
  ...
 50/200   20.5G    0.456     0.234     0.567        512   640
 ...
100/200   20.5G    0.234     0.123     0.345        512   640
 ...
200/200   20.5G    0.189     0.098     0.289        512   640

Validation:
Class     Images  Instances  P      R      mAP50  mAP50-95
all       1000    5234       0.75   0.70   0.68   0.38
person    1000    2345       0.78   0.72   0.71   0.41
bicycle   1000    456        0.69   0.65   0.62   0.33
motorcycle 1000   789        0.73   0.68   0.66   0.36
vehicle   1000    1644       0.80   0.75   0.73   0.42

✅ TRAINING COMPLETE
mAP@50: 0.6800 ← TARGET: >0.65 ACHIEVED!
```

---

## 🎉 Success Criteria

Training is successful if:

✅ All 200 epochs complete without errors  
✅ GPU utilization stays 95-100%  
✅ Loss decreases steadily  
✅ mAP@50 increases over time  
✅ Final mAP@50 ≥ 0.65  
✅ Results saved to runs/thermal_detection/yolo11x_5090_optimized/  
✅ best.pt checkpoint created  

---

## 📞 Need Help?

1. Check error message carefully
2. Review OPTIMIZATION_GUIDE.md for detailed explanations
3. Verify all pre-flight checks passed
4. Check GPU memory: `nvidia-smi`
5. Reduce batch size if OOM
6. Ensure data paths are correct

---

**Ready? Run:** `python train.py` 🚀
