# This is a sample Python script.
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def seg_kmeans_color():
    img = cv2.imread('Fig1.jpg', cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    # 3个通道展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS, 20, 0.01)
    flags = cv2.KMEANS_PP_CENTERS

    # 聚类
    compactness, labels, centers = cv2.kmeans(img_flat,4, None, criteria, 10, flags)



    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))

    rows, cols = img.shape[0],img.shape[1]
    Fig = np.zeros([rows, cols],dtype='uint8')
    for i in range(rows):
        for j in range(cols):
            if (img_output[i,j]==2):  # 0.53 0.65
                Fig[i, j] = 0
            else:
                Fig[i, j] = 1
    """
   shuffle = color.rgb2gray(la)
    io.imshow(shuffle)
    """

    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(Fig, 'gray'), plt.title('kmeans-label3')
    plt.show()
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
    plt.show()

    contours, hierarchy = cv2.findContours(Fig_bitwise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('The number of cells is',len(contours))

    temp = np.ones(Fig_bitwise.shape, np.uint8) * 255
    # 画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
    output=cv2.drawContours(temp, contours, -1, (0, 255, 0), 3)
    #cv2.imshow("contours", temp)
    cv2.imwrite("TheContours.jpg", output)
    cv2.waitKey(0)

if __name__ == '__main__':
    seg_kmeans_color()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
