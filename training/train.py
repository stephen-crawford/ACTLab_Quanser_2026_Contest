#!/usr/bin/env python3
"""
YOLOv8 Training Script for QLabs Competition

Trains a custom YOLOv8 model on the QLabs dataset with 9 classes:
  cone, green_light, red_light, yellow_light, stop_sign,
  yield_sign, round_sign, person, car

Usage:
    python3 training/train.py
    python3 training/train.py --epochs 50 --batch 8 --model s
    python3 training/train.py --resume

The trained model is exported to models/best.pt
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on QLabs dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm'],
                        help='YOLOv8 model size: n(ano), s(mall), m(edium) (default: n)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data.yaml (default: training/data.yaml)')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_yaml = args.data or os.path.join(script_dir, 'data.yaml')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Check dataset exists
    if not os.path.isfile(data_yaml):
        print(f"ERROR: Dataset config not found: {data_yaml}")
        print("Create training/data.yaml and place dataset in training/dataset/")
        sys.exit(1)

    # Load pretrained model (transfer learning from COCO)
    model_name = f'yolov8{args.model}.pt'
    print(f"Loading pretrained {model_name} for transfer learning...")

    if args.resume:
        # Resume from last checkpoint
        last_pt = os.path.join(project_root, 'runs', 'detect', 'train', 'weights', 'last.pt')
        if not os.path.isfile(last_pt):
            print(f"ERROR: No checkpoint found at {last_pt}")
            sys.exit(1)
        model = YOLO(last_pt)
        print(f"Resuming from {last_pt}")
    else:
        model = YOLO(model_name)

    # Train
    print(f"\nTraining configuration:")
    print(f"  Model:   YOLOv8{args.model}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  ImgSize: {args.imgsz}")
    print(f"  Data:    {data_yaml}")
    print(f"  Output:  {models_dir}/best.pt")
    print()

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.join(project_root, 'runs', 'detect'),
        name='train',
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Copy best weights to models/
    best_src = os.path.join(project_root, 'runs', 'detect', 'train', 'weights', 'best.pt')
    best_dst = os.path.join(models_dir, 'best.pt')

    if os.path.isfile(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\nBest model saved to: {best_dst}")
    else:
        print(f"\nWARNING: best.pt not found at {best_src}")

    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    print(f"\nValidation mAP50:    {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

    print(f"\nTraining complete. Model at: {best_dst}")
    print("To use in the vehicle, rebuild the ROS2 package:")
    print("  colcon build --packages-select acc_stage1_mission")


if __name__ == '__main__':
    main()
