from skimage.morphology import skeletonize
import cv2
import numpy as np
import os

mixture_dir = "/media/usama/SSD/Roads_Regions_Classification/Testing_Data_Mixture_Regions_Roads_30_jan_2025/mixture_previous_merged/"
output_dir = "/media/usama/SSD/Roads_Regions_Classification/Roads_removed_dir_31_jan_2025/"
mixture_dir_out = f"{output_dir}/mixture_with_dilation_process"
roads_dir_out = f"{output_dir}/roads_with_dilation_process"
regions_dir_out = f"{output_dir}/regions_with_dilation_process"
if not os.path.exists(mixture_dir_out):
    os.makedirs(mixture_dir_out)
if not os.path.exists(roads_dir_out):
    os.makedirs(roads_dir_out)
if not os.path.exists(regions_dir_out):
    os.makedirs(regions_dir_out)


mixture_images = [f for f in os.listdir(mixture_dir)]
for mixture_image in mixture_images:
    image = cv2.imread(os.path.join(mixture_dir, mixture_image),cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread(image_path)

    # Binarization
    # _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8) 
    
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    # blurred = cv2.medianBlur(image, 3)
    # blurred = cv2.medianBlur(image,5)
    _, binary = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)

    # # Convert to boolean for skimage skeletonize
    binary_bool = binary > 0

    # # Apply skeletonization
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
    skeleton = cv2.dilate(skeleton,kernel,iterations=2)
    mask = cv2.subtract(binary, skeleton)
    # mask = cv2.dilate(mask,kernel,iterations=1)
    

    cv2.imwrite(f"{mixture_dir_out}/{mixture_image}",mask)
    cv2.imwrite(f"{mixture_dir_out}/skelton_{mixture_image}",skeleton)
    cv2.imwrite(f"{mixture_dir_out}/binary_{mixture_image}",binary)  # mask