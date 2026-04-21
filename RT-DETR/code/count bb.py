import json

# Define your file paths
files = {
    "TRAIN": r"D:\Skripsi_Raphaela\rt_detr\dataset\split_dataset\annotations\train.json",
    "VAL":   r"D:\Skripsi_Raphaela\rt_detr\dataset\split_dataset\annotations\val.json",
    "TEST":  r"D:\Skripsi_Raphaela\rt_detr\dataset\split_dataset\annotations\test.json"
}

# Process each file
for split_name, path in files.items():
    with open(path, 'r') as f:
        data = json.load(f)

    # Initialize counters
    a = b = c = d = e = 0

    # Loop through each annotation
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id == 0: a += 1
        elif cat_id == 1: b += 1
        elif cat_id == 2: c += 1
        elif cat_id == 3: d += 1
        elif cat_id == 4: e += 1

    # Print the results for this split
    print(f"--- Statistics for {split_name} ---")
    print(f"Total Bounding Boxes: {len(data['annotations'])}")
    print(f"Excavator    = {a}")
    print(f"Pillar       = {b}")
    print(f"Rock         = {c}")
    print(f"Traffic cone = {d}")
    print(f"Truck        = {e}")
    print("\n")