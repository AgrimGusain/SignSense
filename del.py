import os
import glob

def keep_first_n_images(root_dir, n=500):
    for subset in ['clean', 'noisy']:
        subset_path = os.path.join(root_dir, subset)
        if not os.path.exists(subset_path):
            continue
        for letter in os.listdir(subset_path):
            letter_path = os.path.join(subset_path, letter)
            if not os.path.isdir(letter_path):
                continue
            # Get list of image files (sorted)
            images = sorted(glob.glob(os.path.join(letter_path, '*')))
            # Delete all after the first `n`
            for image_path in images[n:]:
                try:
                    os.remove(image_path)
                    print(f"Deleted: {image_path}")
                except Exception as e:
                    print(f"Failed to delete {image_path}: {e}")

# Update this to your actual dataset folder
dataset_root = 'dataset'
keep_first_n_images(dataset_root, n=500)
