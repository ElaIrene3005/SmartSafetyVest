import os
import json
import random
import shutil
from collections import defaultdict

def stratified_coco_split(json_file, image_folder, output_base_dir, split_ratios=(0.7, 0.2, 0.1)):
    os.makedirs(output_base_dir, exist_ok=True)
    
    output_dirs = {
        'train': os.path.join(output_base_dir, 'train'),
        'val': os.path.join(output_base_dir, 'val'),
        'test': os.path.join(output_base_dir, 'test')
    }
    
    for split in output_dirs.values():
        os.makedirs(os.path.join(split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split, 'annotations'), exist_ok=True)

    print("Reading the monolithic JSON file...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # 1. Map each image to the categories it contains
    img_to_categories = defaultdict(set)
    for ann in annotations:
        img_to_categories[ann['image_id']].add(ann['category_id'])

    # 2. Stratified Distribution Logic
    print("Performing stratified split per class...")
    random.seed(42) # Ensure reproducible results
    
    assigned_image_ids = set()
    split_assignments = {'train': set(), 'val': set(), 'test': set()}

    # Process each category one by one
    for cat in categories:
        cat_id = cat['id']
        
        # Find images containing this category that haven't been assigned to a folder yet
        eligible_images = [img_id for img_id, cats in img_to_categories.items() 
                           if cat_id in cats and img_id not in assigned_image_ids]
        
        random.shuffle(eligible_images)
        
        # Calculate cuts for 70:20:10
        total_eligible = len(eligible_images)
        train_end = int(total_eligible * split_ratios[0])
        val_end = train_end + int(total_eligible * split_ratios[1])
        
        # Assign them
        split_assignments['train'].update(eligible_images[:train_end])
        split_assignments['val'].update(eligible_images[train_end:val_end])
        split_assignments['test'].update(eligible_images[val_end:])
        
        # Mark these images as assigned so they aren't processed twice
        assigned_image_ids.update(eligible_images)

    # 3. Catch any remaining images (e.g., background images with no bounding boxes)
    unassigned_images = [img['id'] for img in images if img['id'] not in assigned_image_ids]
    if unassigned_images:
        random.shuffle(unassigned_images)
        train_end = int(len(unassigned_images) * split_ratios[0])
        val_end = train_end + int(len(unassigned_images) * split_ratios[1])
        
        split_assignments['train'].update(unassigned_images[:train_end])
        split_assignments['val'].update(unassigned_images[train_end:val_end])
        split_assignments['test'].update(unassigned_images[val_end:])

    # Create a quick lookup for building the JSONs later
    img_id_to_split = {}
    for split_name, img_ids in split_assignments.items():
        for img_id in img_ids:
            img_id_to_split[img_id] = split_name

    # 4. Construct the final datasets
    splits_data = {
        'train': {'images': [], 'annotations': []},
        'val': {'images': [], 'annotations': []},
        'test': {'images': [], 'annotations': []}
    }

    # Distribute Image metadata
    for img in images:
        if img['id'] in img_id_to_split:
            splits_data[img_id_to_split[img['id']]]['images'].append(img)

    # Distribute Annotations
    for ann in annotations:
        if ann['image_id'] in img_id_to_split:
            splits_data[img_id_to_split[ann['image_id']]]['annotations'].append(ann)

    # 5. Copy files and save JSONs
    for split_name, split_data in splits_data.items():
        print(f"\nBuilding [{split_name.upper()}] dataset...")
        
        # Save JSON
        json_output_path = os.path.join(output_dirs[split_name], 'annotations', f'{split_name}.json')
        with open(json_output_path, 'w') as f:
            json.dump({
                "categories": categories,
                "images": split_data['images'],
                "annotations": split_data['annotations']
            }, f, indent=4)
            
        # Copy Images
        image_dest_folder = os.path.join(output_dirs[split_name], 'images')
        for img in split_data['images']:
            src_image_path = os.path.join(image_folder, img['file_name'])
            dst_image_path = os.path.join(image_dest_folder, img['file_name'])
            
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"  -> WARNING: Missing file: {img['file_name']}")
                
        print(f"  - Images: {len(split_data['images'])}")
        print(f"  - Bounding Boxes: {len(split_data['annotations'])}")

    print("\n=== STRATIFIED SPLIT COMPLETE ===")

json_path = r"D:\Skripsi_Raphaela\rt_detr\dataset\merged_dataset\final_merged_dataset.json"
image_path = r"D:\Skripsi_Raphaela\rt_detr\dataset\merged_dataset"
output_dir = r"D:\Skripsi_Raphaela\rt_detr\dataset\split_dataset"

stratified_coco_split(json_path, image_path, output_dir)