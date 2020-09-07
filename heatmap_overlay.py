import cv2
import numpy as np
import matplotlib.pyplot as plt

base = np.full((720, 1280), 0, np.uint16)

# ---------------------------------------------------
# 疊加所有儲存的np array
# ---------------------------------------------------
# for i in range(1, 4):
#     layer_2_add = np.load('base_{}00.npy' .format(i))
#     base += layer_2_add
# layer_2_add = np.load('base_last_.npy')
# base += layer_2_add

for i in range(39, 42):
    layer_2_add = np.load('base_17_{}.npy' .format(i))
    print('base_17_{}.npy' .format(i))
    base += layer_2_add
layer_2_add = np.load('base_last.npy')
base += layer_2_add

# 找經過累加之後的最大值
base_max = np.amax(base)
print("此熱圖中最大值為:", base_max)

# Normalize: uint16(65536) --> uint8(256)
if base_max <= 255:
    amp_ratio = 256//base_max
    print("amp_ratio:", amp_ratio)
    base *= amp_ratio
else:   # 通常經過累加會大於256
    shrink_ratio = base_max/256
    print("shrink_ratio:", shrink_ratio)
    base_float = base-1 / shrink_ratio  # 縮到0~255
    base = base_float.astype(np.uint8)  # 浮點數轉整數
cv2.imwrite("base.jpg", base)

# color mapping --> cmap
cm = plt.get_cmap('gist_rainbow')
colored_image = cm(base)
print(colored_image.shape)
heat_array = (colored_image[:, :, :3] * 255).astype(np.uint8)

# 存cmap圖
cv2.imwrite("./heat.jpg", heat_array)

# 疊加生成heatmap
original_img = cv2.imread("original_img.jpg", 1)
heatmap = cv2.addWeighted(original_img, 0.3, heat_array, 0.7, 0)  # 權重越大透明度越低
# 存疊加後的heatmap
cv2.imwrite('heatmap37.jpg', heatmap)
