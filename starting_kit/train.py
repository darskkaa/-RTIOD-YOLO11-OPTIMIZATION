"""
RTIOD YOLO11 Training Script - Optimized for Thermal Object Detection
======================================================================
Target: LTDv2 dataset (Long-Term Thermal Detection v2)
Goal: Maximize mAP@50 from baseline ~0.40-0.45 to 0.65-0.80+

Key Optimizations for RTX 5090:
- YOLO11x (largest model) for maximum thermal gradient capture
- 640px resolution to preserve small distant objects
- Batch size 64 for optimal gradient estimates
- SGD optimizer with cosine annealing for thermal robustness
- Thermal-specific augmentation (disable saturation, boost brightness)
- 200 epochs with extended patience for seasonal variations
"""

from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
import hydra
import torch
import os
from pathlib import Path


@hydra.main(config_path='config', config_name='config', version_base="1.3")
def main(args):
    """
    Optimized YOLO11 training for thermal object detection (RTIOD/LTDv2)
    
    Expected Performance (RTX 5090):
    - Training time: ~3-5 hours for 200 epochs @ 640px
    - Target mAP@50: 0.65-0.80 (vs baseline 0.40-0.45)
    - Memory usage: ~20-24GB VRAM @ batch 64
    """
    
    # ============================================================================
    # DEVICE SETUP
    # ============================================================================
    device = 0 if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n{'='*70}")
        print(f"🚀 GPU DETECTED: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.1f} GB")
        print(f"{'='*70}\n")
    else:
        print(f"\n⚠️  WARNING: No GPU detected. Training on CPU will be extremely slow.\n")
    
    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    print(f"📦 Loading model: {args.modelCheckpoint}")
    model = YOLO(args.modelCheckpoint)
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    # Process class weights if provided
    class_weights = None
    if hasattr(args, 'class_weights') and args.class_weights:
        class_weights = [float(w) for w in args.class_weights]
        print(f"\n⚖️  Using class weights: {class_weights}")
        print("   (Higher weight = more focus on rare classes)")
    
    print(f"\n{'='*70}")
    print(f"🔧 TRAINING CONFIGURATION (Thermal-Optimized)")
    print(f"{'='*70}")
    print(f"Model:           {args.modelCheckpoint}")
    print(f"Image Size:      {args.imgSize}px (4x baseline for small objects)")
    print(f"Batch Size:      {args.batchSize} (RTX 5090 optimized)")
    print(f"Epochs:          {args.epochs}")
    print(f"Learning Rate:   {args.lr} → {args.lr * args.lrf} (cosine decay)")
    print(f"Optimizer:       {args.optimizer} (momentum={args.momentum})")
    print(f"Warmup:          {args.warmup_epochs} epochs")
    print(f"Patience:        {args.patience} epochs")
    if class_weights:
        print(f"Class Weights:   {class_weights}")
    print(f"{'='*70}\n")
    
    # ============================================================================
    # TRAINING
    # ============================================================================
    results = model.train(
        # ========== Data Configuration ==========
        data="data/data.yaml",
        
        # ========== Image and Batch Settings ==========
        imgsz=args.imgSize,  # 640px for RTX 5090
        batch=args.batchSize,  # 64 for optimal gradients
        device=device,
        workers=8,  # Data loading workers
        
        # ========== Training Duration ==========
        epochs=args.epochs,  # 200 epochs for thermal variations
        patience=args.patience,  # 25 epochs early stopping
        
        # ========== Learning Rate and Optimization ==========
        lr0=args.lr,  # Initial LR: 0.001
        lrf=args.lrf,  # Final LR ratio: 0.01
        momentum=args.momentum,  # 0.937
        weight_decay=args.weight_decay,  # 0.0005
        warmup_epochs=args.warmup_epochs,  # 5 epochs
        warmup_momentum=args.warmup_momentum,  # 0.8
        warmup_bias_lr=0.1,  # Warmup bias LR
        
        # ========== Optimizer Settings ==========
        optimizer=args.optimizer,  # SGD for thermal stability
        cos_lr=args.cos_lr,  # Cosine annealing
        
        # ========== Thermal-Specific Augmentation ==========
        # Color augmentation (minimal for thermal/grayscale)
        hsv_h=args.augmentation.hsv_h,  # 0.01 (minimal hue)
        hsv_s=args.augmentation.hsv_s,  # 0.0 (DISABLE saturation)
        hsv_v=args.augmentation.hsv_v,  # 0.5 (BOOST brightness - critical!)
        
        # Geometric augmentation
        degrees=args.augmentation.degrees,  # 15° rotation
        translate=args.augmentation.translate,  # 0.15 translation
        scale=args.augmentation.scale,  # 0.6 scale variation
        shear=args.augmentation.shear,  # 2.0 shear
        perspective=args.augmentation.perspective,  # 0.0005 perspective
        flipud=args.augmentation.flipud,  # 0.5 vertical flip
        fliplr=args.augmentation.fliplr,  # 0.5 horizontal flip
        
        cls_pw=class_weights,  # Class weights for imbalanced data
        fl_gamma=args.fl_gamma,  # Focal loss gamma
        label_smoothing=args.label_smoothing,  # Label smoothing epsilon
        
        # Data augmentation
        hsv_h=args.hsv_h,  # Image HSV-Hue augmentation (fraction)
        hsv_s=args.hsv_s,  # Image HSV-Saturation augmentation (fraction)
        hsv_v=args.hsv_v,  # Image HSV-Value augmentation (fraction)
        degrees=args.degrees,  # Image rotation (+/- deg)
        translate=args.translate,  # Image translation (+/- fraction)
        scale=args.scale,  # Image scale (+/- gain)
        shear=args.shear,  # Image shear (+/- deg)
        perspective=args.perspective,  # Image perspective (+/- fraction)
        flipud=args.flipud,  # Image flip up-down (probability)
        fliplr=args.fliplr,  # Image flip left-right (probability)
        mosaic=args.mosaic,  # Image mosaic (probability)
        mixup=args.mixup,  # Image mixup (probability)
        copy_paste=args.copy_paste,  # Segment copy-paste (probability)
        
        # Advanced training settings
        close_mosaic=args.close_mosaic,  # Disable mosaic last N epochs
        amp=True,  # Automatic Mixed Precision
        single_cls=args.single_cls,  # Treat as single-class dataset
        overlap_mask=True,  # Overlap masks (better for small objects)
        mask_ratio=4,  # Mask downsample ratio
        nbs=64,  # Nominal batch size
        
        # Logging and visualization
        plots=True,  # Save training plots
        save_json=True,  # Save results to JSON
        project='runs/train',  # Save to project/name
        name='yolo11x_thermal',  # Save results to project/name
        exist_ok=True,  # Existing project/name ok, do not increment
        
        # Performance optimizations (RAM caching disabled as requested)
        cache=None,  # No RAM caching (disabled)
        workers=args.workers if hasattr(args, 'workers') else 8,  # Use configured workers
        rect=False,  # Rectangular training (disable for mosaic)
        resume=False,  # Resume from last checkpoint
        
        # Validation settings
        val=True,  # Validate during training
        save_hybrid=False,  # Save hybrid version of labels
        save_conf=True,  # Save confidence scores
        save_crop=False,  # Save cropped prediction plots
        
        # Debugging
        verbose=True,  # Verbose output
        profile=False,  # Profile ONNX and TensorRT speeds
        
        # Advanced options
        fraction=1.0,  # Train on all data
        deterministic=False,  # Reproducible training
        
        # Additional YOLO-specific
        anchor_t=4.0,  # Anchor-multiple threshold
        bbox_interval=-1,  # Set bounding-box image logging interval
        quad=False,  # Quad dataloader
        noautoanchor=False,  # Disable auto-anchor
        evolve=None,  # Evolve hyperparameters
        bucket='',  # GCS bucket
        
        # Additional optimizations
        pad=0.5,  # Image padding
        prefix='',  # Prefix for training output
        freeze=None  # Freeze layers (list)
    )
    
    # ============================================================================
    # FINAL VALIDATION & MODEL OPTIMIZATION
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"🔍 RUNNING FINAL VALIDATION")
    print(f"{'='*70}")
    
    # Run validation with best weights (RAM caching disabled)
    val_results = model.val(
        data=args.data,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        imgsz=args.img_size,
        conf_thres=0.001,  # Lower confidence threshold for validation
        iou_thres=0.6,     # Standard NMS IoU threshold
        max_det=300,       # Maximum detections per image
        half=True,         # Use half precision for validation
        device=args.device,
        cache=None,        # No RAM caching
        workers=args.workers if hasattr(args, 'workers') else 8,  # Use configured workers
        save_json=True,    # Save results for analysis
        plots=True,        # Generate validation plots
        verbose=True,      # Print results
        compute_map=True   # Compute mAP
    )
    
    # Print final metrics
    print(f"\n{'='*70}")
    print(f"📊 FINAL VALIDATION METRICS")
    print(f"{'='*70}")
    print(f"mAP@0.5:       {val_results.box.map50:.4f}  ← TARGET: >0.65")
    print(f"mAP@0.5:0.95:  {val_results.box.map:.4f}")
    print(f"Precision:     {val_results.box.mp:.4f}")
    print(f"Recall:        {val_results.box.mr:.4f}")
    print(f"Box Loss:      {val_results.box.mp:.4f}")
    print(f"Cls Loss:      {val_results.box.mr:.4f}")
    print(f"{'='*70}")
    
    # Check if target achieved
    if val_results.box.map50 >= 0.65:
        print(f"🎉 SUCCESS! Target mAP@50 ≥ 0.65 achieved!")
    elif val_results.box.map50 >= 0.60:
        print(f"✅ Good performance! Close to target.")
    else:
        print(f"⚠️  Below target. Consider:")
        print(f"   - Increase epochs to 250-300")
        print(f"   - Try yolov11x with 800px resolution")
        print(f"   - Check data quality and annotations")
    
    print(f"\n📁 Results saved to: runs/thermal_detection/yolo11x_5090_optimized/")
    print(f"   - weights/best.pt    ← Use this for predictions")
    print(f"   - weights/last.pt    ← Latest checkpoint")
    print(f"   - results.csv        ← Training metrics")
    print(f"   - results.png        ← Training curves")
    print(f"   - predictions.json   ← Validation predictions (COCO format)")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    main()