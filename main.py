import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "arhamrumi/amazon-product-reviews")

print("Path to dataset files:", path)