import json

def merge_coco_json(file1, file2, output_file):
    print("Reading JSON files...")
    
    # 1. Open and load both JSON files
    with open(file1, 'r') as f1:
        data1 = json.load(f1)
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    # 2. Prepare a dictionary for the merged output
    # We assume the categories (Excavator, Pillar, etc.) are exactly the same in both files
    merged_data = {
        "categories": data1["categories"], 
        "images": data1["images"].copy(),
        "annotations": data1["annotations"].copy()
    }

    # 3. Find the highest IDs in File 1 to prevent collisions when adding File 2
    max_image_id = max([img["id"] for img in data1["images"]], default=0)
    max_annot_id = max([ann["id"] for ann in data1["annotations"]], default=0)

    # Dictionary to keep track of how image IDs from File 2 are changed
    image_id_mapping = {}

    print("Processing images and annotations from File 2...")

    # 4. Process Images from File 2
    for img in data2["images"]:
        old_id = img["id"]
        # Create a new ID that continues from the highest ID in File 1
        new_id = old_id + max_image_id 
        
        # Record this change in our mapping dictionary
        image_id_mapping[old_id] = new_id 
        
        # Update the image ID and append to the merged list
        img["id"] = new_id 
        merged_data["images"].append(img)

    # 5. Process Annotations (Bounding Boxes) from File 2
    for ann in data2["annotations"]:
        # Create a new annotation ID
        ann["id"] = ann["id"] + max_annot_id 
        
        # Match this annotation to the newly updated image ID using our mapping
        ann["image_id"] = image_id_mapping[ann["image_id"]] 
        
        merged_data["annotations"].append(ann)

    # 6. Save the merged data into a new JSON file
    print(f"Saving the merged dataset to {output_file}...")
    with open(output_file, 'w') as out:
        # indent=4 makes the JSON file neatly formatted and human-readable
        json.dump(merged_data, out, indent=4) 

    print("\n=== MERGE SUCCESSFUL ===")
    print(f"Total Merged Images      : {len(merged_data['images'])}")
    print(f"Total Merged Annotations : {len(merged_data['annotations'])}")

first_file = r'D:\Skripsi_Raphaela\rt_detr\dataset\full_dataset\result_cleaned.json'
second_file = r'D:\Skripsi_Raphaela\rt_detr\dataset\full_dataset\result2_cleaned.json'
final_output = r'D:\Skripsi_Raphaela\rt_detr\dataset\full_dataset\final_merged_dataset.json'

merge_coco_json(first_file, second_file, final_output)