import numpy as np
import numpy as np
from scipy.integrate import dblquad

me=0.511


def w(theta,phi,s):
    return np.array([[np.exp(-1j*phi/2)*np.cos((theta+(1/2-s))/2)], [np.exp(1j*phi/2)*np.sin((theta+(1/2-s))/2)]])
def w_dagger(theta,phi,s):
    return np.array([np.exp(1j*phi/2)*np.cos((theta+(1/2-s))/2), np.exp(-1j*phi/2)*np.sin((theta+(1/2-s))/2)])


def u(epsilon, theta, phi, s):
    res= np.block([[np.sqrt(epsilon + me)*w(theta,phi,s)],[2*np.sqrt(epsilon - me)*s*w(theta,phi,s)]])
    return res  

def u_f(epsilon_f,theta_f,phi_f,s_f):
    res= np.block([np.sqrt(epsilon_f + me)*w_dagger(theta_f,phi_f,s_f),2*np.sqrt(epsilon_f - me)*s_f*w_dagger(theta_f,phi_f,s_f)])
    return res

v = np.array([[1], [2], [3]])
s = np.array([1, 2, 3])
#print()


# 定义复数函数
def f(x, y):
    return np.exp(1j * (x + y))  

# 定义积分区间
a, b = 0, np.pi  # x 的积分区间
c, d = 0, np.pi  # y 的积分区间

# 使用 dblquad 进行数值积分
# 注意：dblquad 要求函数签名为 f(y, x)，因此需要调整参数顺序
result_real, error_real = dblquad(lambda y, x: f(x, y).real, a, b, lambda x: c, lambda x: d)
result_imag, error_imag = dblquad(lambda y, x: f(x, y).imag, a, b, lambda x: c, lambda x: d)

# 组合实部和虚部
result = result_real + 1j * result_imag
error = error_real + 1j * error_imag

# 输出结果
print("双重积分结果:", result)
print("误差估计:", error)
