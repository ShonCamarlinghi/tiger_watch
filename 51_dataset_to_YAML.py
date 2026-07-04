import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four


DATASET_NAME = "my_wildlife_dataset"
EXPORT_DIR = "/home/shon/Sandbox/datasets/YOLO_wildlife"
CLASSES = ["Person", "Dog", "Cat", "Tiger", "Bird", "Snake", "Bear"]


# 1. Load the persistent dataset if it already exists; otherwise download it.
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

# 2. Assign random splits by adding "train" and "val" tags to the samples.
# https://docs.voxel51.com/api/fiftyone.utils.random.html
dataset.persistent = True
dataset.untag_samples(["train", "val"])
four.random_split(
    dataset,
    {"train": 0.8, "val": 0.2},
    seed=42,
)

# 3. Create views for each split.
train_view = dataset.match_tags("train")
val_view = dataset.match_tags("val")

# 4. Export each split in YOLOv5 format. The shared class list keeps
# dataset.yaml consistent even if one split is missing a rare class.
train_view.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="train",
    classes=CLASSES,
    overwrite=True,
)

val_view.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="val",
    classes=CLASSES,
    overwrite=True,
)

print(f"Dataset exported successfully to {EXPORT_DIR}")
