import cv2
import numpy as np
from skimage.morphology import skeletonize

# image_final = cv2.imread("processed_outputs_6_feb_2025/removed_filled_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_31.jpg")
# # image = cv2.imread("processed_outputs_6_feb_2025/removed_filled_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_87.jpg")
# kernel = np.ones((3, 3), np.uint8)
# # # image_inverted = cv2.bitwise_not(image)
# _,image_final = cv2.threshold(image_final,50,255,cv2.THRESH_BINARY)
# image_final_bool = image_final.astype(bool)
# skeleton = skeletonize(image_final_bool).astype(np.uint8) * 255 
# # skeleton_close = cv2.morphologyEx(skeleton,cv2.MORPH_CLOSE,kernel,iterations=4)
# cv2.imwrite("skeleton.jpg",skeleton)

# _,image = cv2.threshold(image,128,255,cv2.THRESH_BINARY)
# # image = cv2.medianBlur(image,5)
# image_final_closed = cv2.morphologyEx(image_final,cv2.MORPH_CLOSE,kernel,iterations=5)
# image_closed = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=4)

# # _, ori_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
# org_mask_bool = image > 0
# org_skeleton = skeletonize(org_mask_bool).astype(np.uint8) * 255 
# # # image_xor = cv2.bitwise_or(image,image_inverted)
# cv2.imwrite("xor_image.jpg", image_final_closed)
# cv2.imwrite("image.jpg", image_closed)





# Fill the regions inside it ?
#########################################################################
import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)
def remove_parent_contours(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to binary
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel,iterations=2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask to draw child contours
    mask = np.zeros_like(image)
    
    # Loop through contours and keep only child contours
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3]==-1:  # Check if the contour has a parent
            cv2.drawContours(mask, [contour], -1, 255, thickness=2)
            # mask = cv2.bitwise_not(mask)
    
    return mask

image_path = "processed_outputs_6_feb_2025/removed_filled_ca_lemon_grove_SwinIR_3_mask_ca_lemon_grove_SwinIR_47.jpg"
result = remove_parent_contours(image_path)
cv2.imwrite("output_image1.jpg", result)

cont_image = cv2.imread("output_image1.jpg")

cont_image_inverted = cv2.bitwise_not(cont_image)
image = cv2.imread(image_path)
image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=3)

anded_image = cv2.bitwise_and(cont_image_inverted,image)
anded_image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel,iterations=1)
# anded_image_1 = cv2.bitwise_and(cont_image,anded_image)

# anded_image_3 = cv2.subtract(anded_image,anded_image_1)

# anded_image_4 = cv2.bitwise_or(anded_image_3,image)

cv2.imshow("cont_image inverted",cont_image_inverted)

cv2.imshow("anded_image",anded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#################################################################################################



