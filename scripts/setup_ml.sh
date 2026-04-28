#!/bin/bash
# TruthBridge ML Setup Script
# Run this to prepare datasets and generate hash database

echo "============================================"
echo "TruthBridge ML Dataset Preparation"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip install Pillow imagehash scipy scikit-image numpy tqdm 2>/dev/null || pip3 install Pillow imagehash scipy scikit-image numpy tqdm

echo ""
echo "🔄 Running dataset deduplication..."
echo "   This will scan all images in data/ folder,"
echo "   compute perceptual hashes, and find duplicates."
echo ""
echo "   Processing folders:"
ls -d data/*/ 2>/dev/null | sed 's|^|   - |'
echo ""
read -p "Continue? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

python3 scripts/dedupe_dataset.py

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
echo ""
echo "Hash database saved to: public/models/image_hashes.json"
echo ""
echo "To build YOLOv8 damage classifier next:"
echo "   1. pip install ultralytics"
echo "   2. python scripts/prepare_yolo_dataset.py"
echo "   3. yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640"
echo ""