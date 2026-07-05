import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four


DATASET_NAME = "my_wildlife_dataset"
EXPORT_DIR = "/home/shon/Sandbox/datasets/YOLO_wildlife"
CLASSES = ["Person", "Dog", "Cat", "Tiger", "Bird", "Snake", "Bear"]


# 1. Load virtual container if the persistent dataset already exists; otherwise download it.
if fo.dataset_exists(DATASET_NAME):
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"Loaded existing dataset '{DATASET_NAME}'")
else:
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",        # This keeps the sample count manageable on an RTX 4060 Ti.
        classes=CLASSES,
        label_types=["detections"],
        label_field="ground_truth",
        max_samples=4000,
        dataset_name=DATASET_NAME,
        persistent=True,
    )
    print(f"Dataset '{DATASET_NAME}' created and saved successfully!")

# 2. Safety step: Wipe old tag attributes to prevent splitting bugs: https://github.com/voxel51/fiftyone/issues/1952
dataset.untag_samples(dataset.distinct("tags"))

# 3. Extract the list of valid class categories dynamically
# access label field by assigned name label_field="ground_truth"
class_list = dataset.distinct("ground_truth.detections.label")
print(f"Discovered classes for YOLO: {class_list}")

# 4. Partition virtual samples inside FiftyOne (80% train / 20% val)
# https://docs.voxel51.com/api/fiftyone.utils.random.html
four.random_split(
    dataset,
    {"train": 0.8, "val": 0.2},
    seed=42
)

# 5. Safe multi-split dictionary mapping
split_dict = {
    "train": dataset.match_tags("train"),
    "val": dataset.match_tags("val")
}

# 6. Export each split in YOLOv5 format: generates the images/, labels/ subfolders and the dataset.yaml file required by YOLOv8, YOLOv9, and YOLOv10.
fo.types.YOLOv5Dataset.export_dataset(
    dataset_or_view=split_dict,
    export_dir=EXPORT_DIR,            # Double-check your path permission
    label_field="ground_truth",       # Must match step 3
    classes=class_list,                            # Enforces explicit class indexes
    overwrite=True                    # Wipes partial folders and builds fresh directories
)

print(f"Dataset exported successfully to {EXPORT_DIR}")

