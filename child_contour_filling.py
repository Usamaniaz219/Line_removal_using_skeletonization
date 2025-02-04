import cv2
import numpy as np
import os



mixture_dir = "/media/usama/SSD/Line_removal_using_skeletonization/Filled_regions_test_data_3_feb_2025/roads_images_previous/"
output_dir = "/media/usama/SSD/Line_removal_using_skeletonization/Filled_regions_test_data_3_feb_2025/"
mixture_dir_out = f"{output_dir}/mixture_images_filled"
roads_dir_out = f"{output_dir}/roads_images_filled"
regions_dir_out = f"{output_dir}/regions_images_filled"
if not os.path.exists(mixture_dir_out):
    os.makedirs(mixture_dir_out)
if not os.path.exists(roads_dir_out):
    os.makedirs(roads_dir_out)
if not os.path.exists(regions_dir_out):
    os.makedirs(regions_dir_out)



def fill_child_contours(binary_image):
    # Find contours in the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image,cv2.MORPH_CLOSE,kernel,iterations=1)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    filled_image = binary_image.copy()

    if hierarchy is not None:
        for i in range(len(contours)):
            # Fill only child contours (holes)
            if hierarchy[0][i][3] != -1:  # If it has a parent
                cv2.drawContours(filled_image, [contours[i]], -1, 255, thickness=cv2.FILLED)
    
    return filled_image



mixture_images = [f for f in os.listdir(mixture_dir)]
for image in mixture_images:
    img = cv2.imread(os.path.join(mixture_dir, image), 0)


# binary_image = cv2.imread("/media/usama/SSD/Roads_Regions_Classification/Testing_Data_Mixture_Regions_Roads_30_jan_2025/mixture/demo126_2_mask_demo126_18.jpg", cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    filled_image = fill_child_contours(binary_image)

    # cv2.imwrite("filled_image_4.png",filled_image)
    cv2.imwrite(f"{roads_dir_out}/{image}",binary_image)
