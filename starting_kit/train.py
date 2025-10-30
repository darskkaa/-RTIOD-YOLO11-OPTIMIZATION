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
        
        # Advanced augmentation
        mosaic=args.augmentation.mosaic,  # 1.0 (enable mosaic)
        mixup=args.augmentation.mixup,  # 0.15 (mild mixup)
        copy_paste=args.augmentation.copy_paste,  # 0.1 (small objects)
        erasing=args.augmentation.erasing,  # 0.4 (occlusion robustness)
        crop_fraction=args.augmentation.crop_fraction,  # 1.0
        close_mosaic=args.augmentation.close_mosaic,  # 15 (disable last 15 epochs)
        
        # ========== Validation and Checkpointing ==========
        val=True,  # Validate during training
        save=True,  # Save checkpoints
        save_period=10,  # Save every 10 epochs
        cache=False,  # Don't cache (thermal images are large)
        
        # ========== Logging and Output ==========
        project='runs/thermal_detection',
        name='yolo11x_5090_optimized',
        exist_ok=False,  # Create new experiment folder
        plots=True,  # Generate training plots
        save_json=True,  # Save metrics as JSON
        verbose=True,  # Detailed logging
        
        # ========== Advanced Settings ==========
        amp=True,  # Automatic Mixed Precision (faster on 5090)
        fraction=1.0,  # Use 100% of training data
        profile=False,  # Disable profiling for speed
        overlap_mask=True,  # Overlap masks for better segmentation
        mask_ratio=4,  # Mask downsampling ratio
        dropout=0.0,  # No dropout (YOLO handles regularization)
        label_smoothing=0.0,  # No label smoothing for thermal
        nbs=64,  # Nominal batch size for scaling
        
        # ========== Multi-scale Training ==========
        rect=False,  # Rectangular training (disabled for mosaic)
        resume=False,  # Start fresh (set True to resume from last.pt)
        
        # ========== Loss Weights (YOLO defaults, tuned for COCO) ==========
        box=7.5,  # Box loss weight
        cls=0.5,  # Classification loss weight
        dfl=1.5,  # Distribution focal loss weight
    )
    
    # ============================================================================
    # FINAL VALIDATION
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE - Running Final Validation")
    print(f"{'='*70}\n")
    
    val_results = model.val(
        data="data/data.yaml",
        imgsz=args.imgSize,  # 640px
        batch=args.batchSize,  # 64
        device=device,
        save_json=True,  # Required for COCO evaluation
        plots=True,  # Generate validation plots
        conf=0.001,  # Low confidence for recall
        iou=0.6,  # IoU threshold for NMS
        max_det=300,  # Max detections per image
        verbose=True,
    )
    
    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================
    print(f"\n{'='*70}")
    print(f"📊 FINAL RESULTS (Validation Set)")
    print(f"{'='*70}")
    print(f"mAP@50:       {val_results.box.map50:.4f}  ← TARGET: >0.65")
    print(f"mAP@50-95:    {val_results.box.map:.4f}")
    print(f"Precision:    {val_results.box.mp:.4f}")
    print(f"Recall:       {val_results.box.mr:.4f}")
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