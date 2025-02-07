from skimage.morphology import skeletonize
import cv2
import numpy as np
import os

mixture_dir = "/media/usama/SSD/Line_removal_using_skeletonization/Filled_regions_test_data_3_feb_2025/mixture_images_previous_filled/"
output_dir = "/media/usama/SSD/Line_removal_using_skeletonization/"
mixture_dir_out = f"{output_dir}/mixture_outputs_7_feb_11"
roads_dir_out = f"{output_dir}/roads_outputs_7_feb"
regions_dir_out = f"{output_dir}/regions_outputs_7_feb"
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
    # kernel = np.ones((3, 3), np.uint8) 
    
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    
    # blurred = cv2.medianBlur(image,41)
    # blurred = cv2.medianBlur(image, 3)
    # blurred = cv2.medianBlur(image,5)
    _, binary = cv2.threshold(blurred,240, 255, cv2.THRESH_BINARY)
    # binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=8)

    # # Convert to boolean for skimage skeletonize
    binary_bool = binary > 0

    # # Apply skeletonization
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
    # binary = cv2.dilate(binary,kernel,iterations=2)
    # skeleton = cv2.dilate(skeleton,kernel,iterations=2)
    mask = cv2.subtract(binary, skeleton)
    # mask = cv2.dilate(mask,kernel,iterations=1)
    # _,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)
    # mask = cv2.medianBlur(mask,5)
    # mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=8)
    

    # mask = cv2.medianBlur(mask,5)

    # _, mask = cv2.threshold(mask,200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{mixture_dir_out}/{mixture_image}",mask)
    # cv2.imwrite(f"{regions_dir_out}/skelton_{mixture_image}",skeleton)
    # cv2.imwrite(f"{regions_dir_out}/binary_{mixture_image}",binary)  # mask

    # cv2.imwrite(f"Bang_{mixture_image}",mask)
    # cv2.imwrite(f"skelton_{mixture_image}",skeleton)