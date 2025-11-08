import cv2
import numpy as np

def sift_feature_matching(img1, img2):
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()

    # 检测关键点和计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比值测试，以获取最佳匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # 绘制匹配结果
    result = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result

if __name__ == "__main__":
    # 读取图像
    img1 = cv2.imread('1_6.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('1_0.png', cv2.IMREAD_GRAYSCALE)

    # 执行SIFT特征匹配
    matched_image = sift_feature_matching(img1, img2)

    # 显示匹配结果
    cv2.imshow('SIFT Feature Matching', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

