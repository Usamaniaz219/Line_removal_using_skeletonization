import cv2
import numpy as np

mask_final_path = "Filled_regions_test_data_3_feb_2025/Filled_regions_outputs_data_4_feb_2025/mixture_outputs_with_child_cont_filled_4_feb_11/binary_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_22.jpg"

ori_mask_path = "Filled_regions_test_data_3_feb_2025/mixture_images_previous/ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_22.jpg"

mask_final = cv2.imread(mask_final_path,cv2.IMREAD_GRAYSCALE)
ori_mask = cv2.imread(ori_mask_path, cv2.IMREAD_GRAYSCALE)
# ori_mask = cv2.bitwise_not(ori_mask)
kernel = np.ones((3,3),np.uint8)
filled_region_extracted = cv2.bitwise_and(mask_final,ori_mask)
filled_region_extracted = cv2.morphologyEx(filled_region_extracted,cv2.MORPH_CLOSE,kernel,iterations=6)
# Find contours in it 
_, filled_region_extracted = cv2.threshold(filled_region_extracted, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(filled_region_extracted, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
filled_image = filled_region_extracted.copy()
for i in range(len(contours)):
    # Fill only child contours (holes)
    
    cv2.drawContours(filled_image, [contours[i]], -1, 255, thickness=cv2.FILLED)
# cv2.imwrite("filled_region_extracted.jpg",filled_image)
mask_11 = cv2.subtract(mask_final, filled_image)
cv2.imwrite("remove_filled_region_33.jpg",mask_11)
cv2.imwrite("final_image_33.jpg",mask_final)


