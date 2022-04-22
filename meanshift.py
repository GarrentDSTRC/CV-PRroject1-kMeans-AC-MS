#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
#get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
#get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
#get_ipython().system('mkdir /home/aistudio/external-libraries')
#get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets.samples_generator import make_blobs


# In[ ]:


img = cv2.imread('Fig1.jpg', cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

# 3个通道展平
img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
#img_flat = np.uint8(img_flat)
img_flat.dtype='uint8'
# 迭代参数
bandwidth=estimate_bandwidth(img_flat,quantile=0.2,n_samples=500)
ms=MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(img_flat)
labels=ms.labels_
cluster_centers=ms.cluster_centers_
n_clusters=np.unique(labels)
n=len(n_clusters)
print('The number of the labels are ',n)

# In[ ]:


# 显示结果
img_output = labels.reshape((img.shape[0], img.shape[1]))


# In[ ]:


plt.subplot(121), plt.imshow(img), plt.title('input')
plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('meanshift')
plt.savefig('Fig01.jpg')
plt.show()


# In[ ]:


rows, cols = img.shape[0],img.shape[1]
Fig = np.zeros([rows, cols],dtype='uint8')
DATA=Fig
for k in range (4):
    for i in range(rows):
        for j in range(cols):
            if (img_output[i,j]==k):  # 0.53 0.65
                Fig[i, j] = 0
            else:
                Fig[i, j] = 1
    plt.subplot(2,2,(k+1)), plt.imshow(Fig), plt.title('label-%s'%(k))
    DATA=np.concatenate((DATA,Fig),axis=0)
plt.savefig('Fig2.jpg')
plt.show()
Fig=DATA[(4*rows):(5*rows-1),:]
"""
   shuffle = color.rgb2gray(la)
io.imshow(shuffle)
"""


# In[ ]:

# In[ ]:



# In[ ]:


##erosion dilation
kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype='uint8')

Fig_erosion=cv2.erode(Fig, kernel)
plt.imshow(Fig_erosion, 'gray')

Fig_bitwise=Fig^Fig_erosion
plt.imshow(Fig_bitwise, 'gray')

plt.subplot(121), plt.imshow(Fig_erosion, 'gray'), plt.title('erosion')
plt.subplot(122), plt.imshow(Fig_bitwise, 'gray'), plt.title('erosion-contours')
plt.savefig('Fig3.jpg')
plt.show()


# In[ ]:


contours, hierarchy = cv2.findContours(Fig_bitwise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('The number of cells is',len(contours))

temp = np.ones(Fig_bitwise.shape, np.uint8) * 255
# 画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
output=cv2.drawContours(temp, contours, -1, (0, 255, 0), 3)
#cv2.imshow("contours", temp)
cv2.imwrite("TheContours.jpg", output)
cv2.waitKey(0)


# In[ ]:





# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
