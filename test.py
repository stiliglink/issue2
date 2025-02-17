import numpy as np
import vegas

# 定义被积函数
def integrand(x):
    return np.sin(x[0]) * np.cos(x[1])

# 创建 vegas 积分器
integrator = vegas.Integrator([[0, 1], [0, 1]])


result = integrator(integrand, nitn=20, neval=10000)

print(result.summary())
 