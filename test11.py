import cv2
import numpy as np
from skimage.morphology import skeletonize

# processed_mask_path = "processed_outputs_6_feb_2025/removed_filled_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_47.jpg"
# org_mask_path = "Filled_regions_test_data_3_feb_2025/mixture_images_previous/ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_47.jpg"

# org_mask = cv2.imread(processed_mask_path,cv2.IMREAD_GRAYSCALE)
# org_mask_thresh = cv2.adaptiveThreshold(org_mask, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

# org_mask_thresh_bool = org_mask_thresh > 0

# org_skeleton = skeletonize(org_mask_thresh_bool).astype(np.uint8) * 255 
# # processed_mask = cv2.subtract(org_mask,org_mask_thresh)

# cv2.imshow("thresh_mask",org_mask_thresh)
# cv2.imshow("org_skelton_mask",org_skeleton)
# cv2.imshow("org_mask",org_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import time
import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon
from shapely.validation import make_valid

def retrieve_poly(args):
    ori, contours_filled = args
    cnt_ori_2d = np.squeeze(ori)
    if cnt_ori_2d.shape[0] < 4:
        return None

    polygon_ori = Polygon(cnt_ori_2d)
    valid_polygon_ori = make_valid(polygon_ori)
  
    # print("valid polygon original", valid_polygon_ori)

    for cnt_fill in contours_filled:
        
        # print("Contour Filled",cnt_fill)
        cnt_fill_2d = np.squeeze(cnt_fill)
        if cnt_fill_2d.shape[0] <4:
            continue

        polygon_fill = Polygon(cnt_fill_2d)
        polygon_fill = make_valid(polygon_fill)
        # print("walid Polygon fill",polygon_fill)
        if valid_polygon_ori.intersects(polygon_fill):
            return ori
    return None
    

def process_image(image_path,tg_path):
    # image_name = os.path.splitext(os.path.basename(image_path))[0]
    # print(f"Processing image: {image_name}")

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tg_image = cv2.imread(tg_path,cv2.IMREAD_GRAYSCALE)

    if original is None:
        print(f"Error reading image: {image_path}")
        return None

    # Thresholding to get the original contours
    _, thresh_original = cv2.threshold(original, 20, 255, cv2.THRESH_BINARY)
    contours_original, _ = cv2.findContours(thresh_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Applying median blur and thresholding to get the filled contours
    # median = cv2.medianBlur(original, 3)
    _, thresh_median = cv2.threshold(tg_image, 25, 255, cv2.THRESH_BINARY)
    contours_filled, _ = cv2.findContours(thresh_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a blank mask
    mask = np.zeros(original.shape, dtype=np.uint8)

    # Prepare arguments for parallel processing
    args = []
    for ori in contours_original:
        # area = cv2.contourArea(ori)
        if ori.shape[0] > 3:
            # print("original contours",ori) 
            args.append((ori, contours_filled))
    # args = [(ori, contours_filled) for ori in contours_original if ori.shape[0] > 3]

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(retrieve_poly, args))

    # Drawing valid contours on the mask
    for result in results:
        # print("result",result)
        if result is not None:
            cv2.drawContours(mask, [result], -1, 255, cv2.FILLED)
    
    # mask = cv2.bitwise_and(mask,thresh_original) 
    # cv2.imwrite("mask_.jpg",mask)
    # cv2.imshow("Mask image",cv2.resize(mask , (700,800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Save the mask
    # output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
    # os.makedirs(output_subdir, exist_ok=True)
    # output_file_path = os.path.join(output_subdir, f"{image_name}_mask.jpg")
    cv2.imwrite("process.jpg", mask)

    # print(f"Image processed: {image_name}")
    return mask
# Load input images
source_image = "mixture_images_full_7_feb_2025_data_for_testing/ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_34.jpg"
target_image = "processed_outputs_7_feb_2025_mixture_11/removed_filled_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_34.jpg"

mask = process_image(source_image,target_image)


