import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 设置风力大小和方向
wind_speed = 30
wind_direction = 45  # 风向角度，45度表示斜向上

# 计算风场矢量分量
wind_u = wind_speed * np.cos(np.radians(wind_direction))
wind_v = wind_speed * np.sin(np.radians(wind_direction))

# 创建网格点，增加网格点间距以使箭头间有间隔
x, y = np.meshgrid(np.linspace(-20, 20, 20), np.linspace(-20, 20, 20))

# 计算每个网格点的风场矢量分量
u = wind_u * np.ones_like(x)
v = wind_v * np.ones_like(y)

# 读取图片
image_path = '0.png'  # 请替换为你的图片路径
img = imread(image_path)

# 绘制图片
plt.imshow(img, extent=[-100, 100, -100, 100])

# 绘制风场矢量箭头，设置颜色为蓝色，缩小箭头大小，增加箭头间的间隔
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=0.8, color='blue')

# 设置坐标轴范围和标签
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()
