import os
import argparse
import requests
import shutil
import zipfile
import random
from tqdm import tqdm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Download cat and non-cat images for training')
    parser.add_argument('--output_dir', type=str, default='cat_dataset', help='Output directory for downloaded images')
    parser.add_argument('--cat_limit', type=int, default=100, help='Maximum number of cat images to download')
    parser.add_argument('--non_cat_limit', type=int, default=100, help='Maximum number of non-cat images to download')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create directory structure
    for split in ['train', 'val']:
        for category in ['cat', 'non_cat']:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    
    # Download cat and dog images from the Oxford-IIIT Pet Dataset
    print("Downloading Oxford-IIIT Pet Dataset...")
    download_pet_dataset(output_dir, args.cat_limit, args.non_cat_limit)
    
    print(f"Done! Dataset saved to {args.output_dir}")
    print("Use the dataset with cat_classifier.py by running:")
    print(f"python cat_classifier.py --batch-size 16 --epochs 10 --output-dir output")

def download_pet_dataset(output_dir, cat_limit, non_cat_limit):
    """Download and prepare the Oxford-IIIT Pet Dataset.
    This dataset contains 37 categories of pet images, including various cat and dog breeds.
    """
    # URLs for the dataset
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    # Download and extract the dataset
    images_path = os.path.join(output_dir, "images.tar.gz")
    annotations_path = os.path.join(output_dir, "annotations.tar.gz")
    
    # Download files if they don't exist
    if not os.path.exists(images_path):
        print(f"Downloading images from {images_url}...")
        download_file(images_url, images_path)
    
    if not os.path.exists(annotations_path):
        print(f"Downloading annotations from {annotations_url}...")
        download_file(annotations_url, annotations_path)
    
    # Extract files
    extract_dir = os.path.join(output_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    print("Extracting files...")
    extract_tar_gz(images_path, extract_dir)
    extract_tar_gz(annotations_path, extract_dir)
    
    # Process the dataset
    process_pet_dataset(extract_dir, output_dir, cat_limit, non_cat_limit)
    
    # Create labels.txt files
    for split in ['train', 'val']:
        for category in ['cat', 'non_cat']:
            create_labels_file(os.path.join(output_dir, split, category), category)
    
    print("Dataset preparation complete!")

def extract_tar_gz(tar_path, extract_dir):
    """Extract a tar.gz file."""
    import tarfile
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)

def process_pet_dataset(extract_dir, output_dir, cat_limit, non_cat_limit):
    """Process the Pet dataset to create cat/non-cat categories."""
    images_dir = os.path.join(extract_dir, "images")
    if not os.path.exists(images_dir):
        images_dir = os.path.join(extract_dir)  # Try alternative location
    
    # List of cat breeds in the dataset (first letter is capitalized)
    cat_breeds = [
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx'
    ]
    
    # Get all image paths
    all_images = []
    for file in os.listdir(images_dir):
        if file.endswith('.jpg'):
            all_images.append(file)
    
    # Separate cat and non-cat images
    cat_images = []
    non_cat_images = []
    
    for img in all_images:
        # Extract the breed name (format: Breed_ID.jpg)
        parts = img.split('_')
        breed = parts[0]
        
        if breed in cat_breeds:
            cat_images.append(img)
        else:
            non_cat_images.append(img)
    
    print(f"Found {len(cat_images)} cat images and {len(non_cat_images)} non-cat images")
    
    # Limit the number of images
    random.seed(42)
    random.shuffle(cat_images)
    random.shuffle(non_cat_images)
    
    cat_images = cat_images[:cat_limit]
    non_cat_images = non_cat_images[:non_cat_limit]
    
    # Split into train/val (80/20)
    cat_split_idx = int(len(cat_images) * 0.8)
    non_cat_split_idx = int(len(non_cat_images) * 0.8)
    
    cat_train = cat_images[:cat_split_idx]
    cat_val = cat_images[cat_split_idx:]
    non_cat_train = non_cat_images[:non_cat_split_idx]
    non_cat_val = non_cat_images[non_cat_split_idx:]
    
    # Copy the images to the output directory
    print("Copying cat images to train set...")
    for img in tqdm(cat_train):
        src = os.path.join(images_dir, img)
        dst = os.path.join(output_dir, 'train', 'cat', img)
        shutil.copy2(src, dst)
    
    print("Copying cat images to validation set...")
    for img in tqdm(cat_val):
        src = os.path.join(images_dir, img)
        dst = os.path.join(output_dir, 'val', 'cat', img)
        shutil.copy2(src, dst)
    
    print("Copying non-cat images to train set...")
    for img in tqdm(non_cat_train):
        src = os.path.join(images_dir, img)
        dst = os.path.join(output_dir, 'train', 'non_cat', img)
        shutil.copy2(src, dst)
    
    print("Copying non-cat images to validation set...")
    for img in tqdm(non_cat_val):
        src = os.path.join(images_dir, img)
        dst = os.path.join(output_dir, 'val', 'non_cat', img)
        shutil.copy2(src, dst)
    
    print(f"Train set: {len(cat_train)} cat images, {len(non_cat_train)} non-cat images")
    print(f"Validation set: {len(cat_val)} cat images, {len(non_cat_val)} non-cat images")

def download_file(url, output_path):
    """Download a file from a URL to the specified output path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)

def create_labels_file(image_dir, label):
    """Create a labels.txt file with the label for each image in the directory."""
    label_file = os.path.join(image_dir, "labels.txt")
    
    with open(label_file, 'w') as f:
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                f.write(f"{filename},{label}\n")

if __name__ == "__main__":
    main() 