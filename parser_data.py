import cv2
import numpy as np
import matplotlib.pyplot as plt

test_dataset = 'TestDatasetBit8'
rgb_image_dir = 'RGB/'
depth_image_dir = 'Depth/'

depth_image = cv2.imread(depth_image_dir + 'depth100.png', -1)
target_depth_image = cv2.imread('OutSider/test_depth.png', -1)

print('depth image shape:', depth_image.shape)
print('depth max value:', np.max(depth_image))
print('target depth image shape:', target_depth_image.shape)
print('target depth max value:', np.max(target_depth_image))

fig, ax = plt.subplots(2, 1)
ax[0].imshow(depth_image)
ax[1].imshow(target_depth_image)
plt.show()


