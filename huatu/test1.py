import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw_line_between_points(image_path, point1, point2):
    # 加载图像
    image = mpimg.imread(image_path)

    # 创建画布并显示图像
    fig, ax = plt.subplots()
    ax.imshow(image)

    # 绘制两点之间的线段
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-')

    # 显示结果
    plt.show()

# 示例用法
image_path = '0.png'  # 替换为你的图像路径
point1 = (1315.5, 560)  # 第一个点的坐标 (x1, y1)
point2 = (1317.5, 560.5)  # 第二个点的坐标 (x2, y2)
draw_line_between_points(image_path, point1, point2)
