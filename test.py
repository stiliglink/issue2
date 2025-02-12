import numpy as np

# 设置积分网格大小
N_theta = 100  # 调整精度 (可选: 100, 200, ...)
N_phi = 100

# 创建 theta 和 phi 的网格
theta_vals = np.linspace(0, np.pi, N_theta)
phi_vals   = np.linspace(0, 2*np.pi, N_phi)
theta, phi = np.meshgrid(theta_vals, phi_vals)

# 函数定义 (你可以替换 test(θ,φ) 为你的函数)
def test(theta, phi):
    return np.exp(-10000 * theta) * np.cos(phi)  # 示例函数

# 计算函数值
f_vals = test(theta, phi)

# 第一步: 先对 theta 方向做梯形积分
int_theta = np.trapz(f_vals, theta_vals, axis=0)

# 第二步: 再对 phi 方向做梯形积分
integral = np.trapz(int_theta, phi_vals)

print(f"梯形法二重积分结果: {integral}")

