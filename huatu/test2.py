import matplotlib.pyplot as plt

# 已知的10个点的图像坐标
points = [
    [1315.5, 560],
    [1317.5, 560.5],
    [1328, 551.5],
    [1318.5, 559.5],
    [1318, 560],
    [1316.5, 559],
    [1317, 559.5],
    [1311.5, 559.5],
    [1318, 560],
    [1313, 559]
]

# 提取x和y坐标
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]

# 绘制散点图
plt.scatter(x_coords, y_coords)

# 连接点
for i in range(len(points) - 1):
    plt.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], 'r-')

# 连接最后一个点和第一个点
plt.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], 'r-')

# 显示图像
plt.show()
