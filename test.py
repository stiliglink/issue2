import numpy as np
import vegas
import numpy as np
from mpi4py import MPI
import vegas


# 加载 .npz 文件
data = np.load('res_0220_154636.npz')

# 查看文件中的数组名称
print(data.files)

# 访问特定的数组
array1 = data['omega_values']
array2 = data['results']

# 打印数组内容
print(array1)
print(array2)

 