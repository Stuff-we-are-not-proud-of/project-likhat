import os
from PIL import Image
import numpy as np
import albumentations as A

# Paths (relative to the coding/ folder)
SRC_DIR = "../MajorDatasets/ResizedDataset"
OUT_DIR = "../MajorDatasets/AugmentedDataset"
os.makedirs(OUT_DIR, exist_ok=True)

# Albumentations pipeline (translation + scaling, safe params)
albumentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.MedianBlur(blur_limit=3, p=0.3),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=0,
        shear=0,
        fit_output=False,
        p=0.8
    ),
])

# Augmentation function
def augment_image(image, num_augments=5):
    image = image.convert("L")  # Ensure grayscale
    np_img = np.array(image)

    # Convert to 3 channels (albumentations expects RGB)
    np_img_rgb = np.stack([np_img]*3, axis=-1)

    augmented_images = []
    for _ in range(num_augments):
        aug = albumentation_pipeline(image=np_img_rgb)['image']
        # Convert back to grayscale
        aug_gray = Image.fromarray(np.mean(aug, axis=-1).astype(np.uint8))
        augmented_images.append(aug_gray)
    return augmented_images

# Main driver
def main():
    found_any = False

    for root, _, files in os.walk(SRC_DIR):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            found_any = True
            src_path = os.path.join(root, filename)

            try:
                image = Image.open(src_path).convert("L")
            except Exception as e:
                print(f"❌ Failed to open {filename}: {e}")
                continue

            # Get the relative path from SRC_DIR and create equivalent path in OUT_DIR
            relative_dir = os.path.relpath(root, SRC_DIR)
            output_dir = os.path.join(OUT_DIR, relative_dir)
            os.makedirs(output_dir, exist_ok=True)

            # File base name (no extension)
            base_name = os.path.splitext(filename)[0]

            # Save original image
            orig_out_path = os.path.join(output_dir, f"{base_name}_orig.png")
            image.save(orig_out_path)

            # Generate and save augmentations
            augmented_images = augment_image(image)
            for idx, aug_img in enumerate(augmented_images):
                out_path = os.path.join(output_dir, f"{base_name}_aug{idx+1}.png")
                aug_img.save(out_path)
                print(f"✅ Saved: {out_path}")

    if not found_any:
        print("⚠️ No valid images found in ResizedDataset!")
    else:
        print("🎉 Dataset augmentation complete!")


# Run
if __name__ == "__main__":
    main()
