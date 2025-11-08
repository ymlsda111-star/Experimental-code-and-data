
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif' ] =['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus' ] =False  # 用来正常显示负号


# 定义一个产生符合高斯分布的函数,均值为loc=0.0,标准差为scale=sigma,输出的大小为size
def gaussian_distribution_generator(sigma):
    return np.random.normal(loc=0.0 ,scale=sigma ,size=None)

# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1 ,1] ,[0 ,1]])

# 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性
Q = np.array([[0.01 ,0] ,[0 ,0.01]])

# 测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差
R = np.array([[1 ,0] ,[0 ,1]])

# 传输矩阵/状态观测矩阵H
H = np.array([[1 ,0] ,[0 ,1]])

# 控制输入矩阵B
B = None

# 初始位置和速度
X0 = np.array([[0] ,[1]])

# 状态估计协方差矩阵P初始化
P =np.array([[1, 0], [0, 1]])

if __name__ == "__main__":
    # ---------------------初始化-----------------------------
    # 真实值初始化 这里还要再写一遍np.array是保证它的类型是数组array
    X_true = np.array(X0)
    # 后验估计值Xk的初始化
    X_posterior = np.array(X0)
    # 第k次误差的协方差矩阵的初始化
    P_posterior = np.array(P)

    # 创建状态变量的真实值的矩阵 状态变量1：速度 状态变量2：位置
    speed_true = []
    position_true = []

    # 创建测量值矩阵
    speed_measure = []
    position_measure = []

    # 创建状态变量的先验估计值
    speed_prior_est = []
    position_prior_est = []

    # 创建状态变量的后验估计值
    speed_posterior_est = []
    position_posterior_est = []

    # ---------------------循环迭代-----------------------------
    # 设置迭代次数为30次

    for i in range(30):
        # --------------------建模真实值-----------------------------

        # 生成过程噪声w w=[w1,w2].T(列向量)
        # Q[0,0]是过程噪声w的协方差矩阵的第一行第一列，即w1的方差，Q[1,1]是协方差矩阵的第二行第二列，即为w2的方差
        # python的np.random.normal(loc,scale,size)函数中scale输入的是标准差，所以要开方
        Q_sigma = np.array([[math.sqrt(Q[0, 0]), Q[0, 1]], [Q[1, 0], math.sqrt(Q[1, 1])]])
        w = np.array([[gaussian_distribution_generator(Q_sigma[0, 0])],
                      [gaussian_distribution_generator(Q_sigma[1, 1])]])
        # print('00',Q[0,0],'它的类型是',type(Q[0,0]))
        # print('开根号的00', Q_sigma[0, 0], '它的类型是', type(Q_sigma[0, 0]))
        # print('00的平方根',math.sqrt(Q[0,0]),"它的类型是",type(math.sqrt(Q[0,0])))
        # print('w[',i,']=',w)

        # 真实值X_true 得到当前时刻的状态;之前我一直在想它怎么完成从Xk-1到Xk的更新，实际上在代码里面直接做迭代就行了，这里是不涉及数组下标的！！！
        # dot函数用于矩阵乘法，对于二维数组，它计算的是矩阵乘积
        X_true = np.dot(A, X_true) + w

        # 速度的真实值是speed_true 使用append函数可以把每一次循环中产生的拼接在一起，形成一个新的数组speed_true
        speed_true.append(X_true[1, 0])
        position_true.append(X_true[0, 0])
        # print(speed_true)

        # --------------------生成观测值-----------------------------
        # 生成过程噪声
        R_sigma = np.array([[math.sqrt(R[0, 0]), R[0, 1]], [R[1, 0], math.sqrt(R[1, 1])]])
        v = np.array(
            [[gaussian_distribution_generator(R_sigma[0, 0])], [gaussian_distribution_generator(R_sigma[1, 1])]])

        # 生成观测值Z_measure 取H为单位阵
        Z_measure = np.dot(H, X_true) + v
        speed_measure.append(Z_measure[1, 0]),
        position_measure.append(Z_measure[0, 0])

        # --------------------进行先验估计-----------------------------
        # 开始时间更新
        # step1:基于k-1时刻的后验估计值X_posterior建模预测k时刻的系统状态先验估计值X_prior
        # 此时模型控制输入U=0
        X_prior = np.dot(A, X_posterior)
        # 把k时刻先验预测值赋给两个状态分量的先验预测值 speed_prior_est = X_prior[1,0];position_prior_est=X_prior[0,0]
        # 再利用append函数把每次循环迭代后的分量值拼接成一个完整的数组
        speed_prior_est.append(X_prior[1, 0])
        position_prior_est.append(X_prior[0, 0])

        # step2:基于k-1时刻的误差ek-1的协方差矩阵P_posterior和过程噪声w的协方差矩阵Q 预测k时刻的误差的协方差矩阵的先验估计值 P_prior
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q

        # --------------------进行状态更新-----------------------------
        # step3:计算k时刻的卡尔曼增益K
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(H, k1) + R
        # k3 = np.dot(np.dot(H, P_prior), H.T) + R  k2和k3是两种写法，都可以
        K = np.dot(k1, np.linalg.inv(k2))

        # step4:利用卡尔曼增益K 进行校正更新状态，得到k时刻的后验状态估计值 X_posterior
        X_posterior_1 = Z_measure - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        # 把k时刻后验预测值赋给两个状态分量的后验预测值 speed_posterior_est = X_posterior[1,0];position_posterior_est = X_posterior[0,0]
        speed_posterior_est.append(X_posterior[1, 0])
        position_posterior_est.append(X_posterior[0, 0])

        # step5:更新k时刻的误差的协方差矩阵 为估计k+1时刻的最优值做准备
        P_posterior_1 = np.eye(2) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)

    # ---------------------再从step5回到step1 其实全程只要搞清先验后验 k的自增是隐藏在循环的过程中的 然后用分量speed和position的append去记录每一次循环的结果-----------------------------

    # 可视化显示 画出速度比较和位置比较
    if True:
        # 画出1行2列的多子图
        fig, axs = plt.subplots(1, 2)
        # 速度
        axs[0].plot(speed_true, "-", color="blue", label="速度真实值", linewidth="1")
        axs[0].plot(speed_measure, "-", color="grey", label="速度测量值", linewidth="1")
        axs[0].plot(speed_prior_est, "-", color="green", label="速度先验估计值", linewidth="1")
        axs[0].plot(speed_posterior_est, "-", color="red", label="速度后验估计值", linewidth="1")
        axs[0].set_title("speed")
        axs[0].set_xlabel('k')
        axs[0].legend(loc='upper left')

        # 位置
        axs[1].plot(position_true, "-", color="blue", label="位置真实值", linewidth="1")
        axs[1].plot(position_measure, "-", color="grey", label="位置测量值", linewidth="1")
        axs[1].plot(position_prior_est, "-", color="green", label="位置先验估计值", linewidth="1")
        axs[1].plot(position_posterior_est, "-", color="red", label="位置后验估计值", linewidth="1")
        axs[1].set_title("position")
        axs[1].set_xlabel('k')
        axs[1].legend(loc='upper left')

        #     调整每个子图之间的距离
        plt.tight_layout()
        plt.figure(figsize=(60, 40))
        plt.show()