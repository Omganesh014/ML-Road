"""
TruthBridge YOLO Dataset Preparation Script
Converts all datasets into YOLO format and creates train/val split.
Classes:
  0: crack
  1: pothole
  2: no_damage
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import json

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "dataset_yolo"
SPLIT_RATIO = 0.15

CLASSES = ["crack", "pothole", "no_damage"]

def polygon_to_bbox(points):
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height

def convert_polygon_label(label_path, img_w, img_h):
    lines = label_path.read_text().strip().split('\n')
    bboxes = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        points = [float(p) for p in parts[1:]]
        x_center, y_center, w, h = polygon_to_bbox(points)
        bboxes.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return bboxes

def process_dataset(source_img_dir, source_label_dir, target_cls, output_img_dir, output_label_dir):
    if not source_img_dir.exists():
        print(f"  ⚠️ Skipping (not found): {source_img_dir}")
        return 0

    img_files = list(source_img_dir.glob("*.jpg")) + list(source_img_dir.glob("*.jpeg")) + list(source_img_dir.glob("*.png"))
    count = 0

    for img_file in img_files:
        label_file = source_label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            try:
                img = Image.open(img_file)
                w, h = img.size
                bboxes = convert_polygon_label(label_file, w, h)
                for bbox in bboxes:
                    target_path = output_img_dir / f"{img_file.name}"
                    shutil.copy2(img_file, target_path)
                    with open(output_label_dir / f"{img_file.stem}.txt", 'w') as f:
                        f.write(bbox + '\n')
                    count += 1
                    break
            except Exception as e:
                print(f"    Error with {img_file}: {e}")

    return count

def prepare_concrete_crack():
    pos_dir = DATA_DIR / "Concrete Crack Images for Classification" / "Positive"
    neg_dir = DATA_DIR / "Concrete Crack Images for Classification" / "Negative"

    crack_imgs = list(pos_dir.glob("*.jpg"))[:1500] if pos_dir.exists() else []
    normal_imgs = list(neg_dir.glob("*.jpg"))[:500] if neg_dir.exists() else []

    for img in crack_imgs:
        shutil.copy2(img, OUTPUT_DIR / "images" / "train" / img.name)
        with open(OUTPUT_DIR / "labels" / "train" / f"{img.stem}.txt", 'w') as f:
            f.write("0 0.5 0.5 0.9 0.9\n")

    for img in normal_imgs:
        shutil.copy2(img, OUTPUT_DIR / "images" / "train" / img.name)
        with open(OUTPUT_DIR / "labels" / "train" / f"{img.stem}.txt", 'w') as f:
            f.write("2 0.5 0.5 0.8 0.8\n")

    return len(crack_imgs) + len(normal_imgs)

def prepare_pothole_data():
    count = 0
    src = DATA_DIR / "Pothole_Image_Data"
    if not src.exists():
        return 0

    for img in src.glob("*.jpg"):
        shutil.copy2(img, OUTPUT_DIR / "images" / "train" / img.name)
        with open(OUTPUT_DIR / "labels" / "train" / f"{img.stem}.txt", 'w') as f:
            f.write("1 0.5 0.5 0.8 0.8\n")
        count += 1
    return count

def prepare_road_damage_dataset():
    base = DATA_DIR / "Road Damage Dataset Potholes, Cracks and Manholes" / "data"
    img_dir = base / "images"
    lbl_dir = base / "labels-YOLO"

    if not img_dir.exists() or not lbl_dir.exists():
        print("  ⚠️ Road damage dataset missing expected structure")
        return 0

    img_files = list(img_dir.glob("*.jpg"))
    count = 0

    for img in img_files:
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            shutil.copy2(img, OUTPUT_DIR / "images" / "train" / img.name)
            shutil.copy2(lbl, OUTPUT_DIR / "labels" / "train" / f"{img.stem}.txt")
            count += 1

    return count

def split_train_val():
    train_imgs = list((OUTPUT_DIR / "images" / "train").glob("*"))
    random.shuffle(train_imgs)
    val_count = max(50, int(len(train_imgs) * SPLIT_RATIO))

    val_imgs = train_imgs[:val_count]
    for img in val_imgs:
        name = img.name
        shutil.move(img, OUTPUT_DIR / "images" / "val" / name)
        lbl = OUTPUT_DIR / "labels" / "train" / f"{img.stem}.txt"
        if lbl.exists():
            shutil.move(lbl, OUTPUT_DIR / "labels" / "val" / f"{img.stem}.txt")

    return len(val_imgs)

def create_data_yaml():
    content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: 3
names: {CLASSES}
"""
    with open(DATA_DIR.parent / "data.yaml", 'w') as f:
        f.write(content)
    print(f"\n✅ Created data.yaml at: {DATA_DIR.parent / 'data.yaml'}")

def main():
    print("============================================")
    print("TruthBridge YOLO Dataset Preparation")
    print("============================================\n")

    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    print("📦 Processing datasets...")

    crack_count = prepare_concrete_crack()
    print(f"  Concrete Crack: {crack_count} images")

    pothole_count = prepare_pothole_data()
    print(f"  Pothole Images: {pothole_count} images")

    road_count = prepare_road_damage_dataset()
    print(f"  Road Damage Dataset: {road_count} images")

    print("\n✂️ Creating train/val split...")
    val_count = split_train_val()
    print(f"  Moved {val_count} images to validation set")

    train_total = len(list((OUTPUT_DIR / "images" / "train").glob("*")))
    val_total = len(list((OUTPUT_DIR / "images" / "val").glob("*")))
    print(f"\n📊 Total: {train_total} train, {val_total} val images")

    create_data_yaml()

    print("\n============================================")
    print("✅ YOLO dataset ready!")
    print("============================================")
    print("\nTo train YOLOv8:")
    print("  pip install ultralytics")
    print("  yolo detect train model=yolov8n.pt data=data.yaml epochs=30 imgsz=640")
    print("\nOr with your GPU:")
    print("  yolo detect train model=yolov8n.pt data=data.yaml epochs=30 imgsz=640 device=0")
    print("")

if __name__ == "__main__":
    main()