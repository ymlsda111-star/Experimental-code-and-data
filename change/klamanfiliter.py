import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, F, B, u, Q, H, R, P):
        """
        :param F: 状态转移矩阵，固定
        :param B: 控制矩阵，固定
        :param u: 控制向量，例如加速度
        :param Q: 外界噪音，协方差矩阵，固定
        :param H: 单位转化矩阵，固定
        :param R: 观测值的协方差矩阵，固定
        :param P: 状态协方差矩阵，更新变化
        """
        self.F = F
        self.B = B
        self.u = u
        self.Q = Q
        self.H = H
        self.R = R
        # K为卡尔曼增益，基于H、P、R计算，更新变化
        self.K = 0
        self.P = P

    def predict(self, x0):
        x1 = np.dot(self.F, x0) + self.B * self.u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return x1

    def update(self, x1, z):
        P_ = np.dot(np.dot(self.H, self.P), self.H.T)
        self.K = np.dot(P_, np.linalg.inv(P_ + self.R))
        x_ = x1 + np.dot(self.K, (z - np.dot(self.H, x1)))
        # 更新P
        self.P = P_ - np.dot(self.K, np.dot(self.H, P_))
        return x_


if __name__ == "__main__":
    """
    构建初始速度为v0的匀加速运动的一维物体
    加速度为a, 作为控制信息
    状态为位置p和速度v
    用卡尔曼滤波来进行跟踪
    """
    # 总时间
    T = 20
    # 速度和加速度初始化值
    v0 = 0
    a = 1
    # 物体状态真实值
    p = [v0 * t + 0.5 * a * t ** 2 for t in range(T)]  # 真实位置
    p = np.array(p)
    v = [v0 + a * t for t in range(T)]  # 真实速度
    v = np.array(v)
    # 构建观测值
    # # 模拟白噪音,精确到小数点后两位
    sd_p = 3  # 位置标准差设置为3
    sd_v = 1  # 速度标准差设置为1
    noise_p = np.round(np.random.normal(0, sd_p, T), 2)  # 位置噪音
    noise_v = np.round(np.random.normal(0, sd_v, T), 2)  # 速度噪音
    # # 带有噪音的观测值
    z = np.array([p + noise_p, v + noise_v])
    # 实例化滤波器
    t = 1  # 时间间隔
    n = 2  # 物体状态[p, v]的维度
    F = np.array([[1, t], [0, 1]])  # 状态转移矩阵, [p_1, v_1] = [[1, t], [0, 1]] * [p_0, v_0]
    B = np.array([[0.5 * t ** 2], [t]])  # 控制矩阵, 描述控制变量和物体状态之间的关系
    u = a  # 控制向量，此例子中为加速度
    P = np.eye(n)  # 状态协方差矩阵初始化
    Q = np.array([[1, 0], [0, 1]])  # 外界噪音协方差矩阵
    H = np.eye(n)  # 单位转化矩阵
    R = np.array([[sd_p ** 2, 0], [0, sd_v ** 2]])  # 观测值协方差矩阵
    kf = KalmanFilter(F, B, u, Q, H, R, P)

    # 绘制真实值、观测值、滤波输出值三者之间的动态关系
    fusion = np.zeros([2, T - 1])
    # 初始化
    x = z[:, 0].reshape((2, 1))
    # 绘图的x、y轴
    ax = []
    ay_real = []
    ay_sensor = []
    ay_fusion = []
    # 将 figure 设置为交互模式，figure 不用 plt.show() 也可以显示
    plt.ion()
    for i in range(T - 1):
        x = kf.predict(x)
        x = kf.update(x, z[:, i + 1].reshape((2, 1)))
        fusion[0, i] = x[0, 0]
        fusion[1, i] = x[1, 0]
        ax.append(i + 1)
        ay_real.append(p[i + 1])
        ay_sensor.append(p[i + 1] + noise_p[i + 1])
        ay_fusion.append(x[0, 0])
        plt.clf()
        plt.plot(ax, ay_real, color='k', marker='o', markersize=5, label='real position')
        plt.plot(ax, ay_sensor, color='b', marker='^', markersize=5, label='sensory position')
        plt.plot(ax, ay_fusion, color='r', marker='s', markersize=5, label='fusion position')
        plt.legend()
        plt.pause(0.5)  # 暂停一段时间，不然画得太快会卡住显示不出来
        plt.ioff()  # 将 figure 设置为阻塞模式，也是 figure 的默认模式，figure 必须用 plt.show() 才能显示
    plt.show()

