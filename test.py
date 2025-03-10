import numpy as np
import math
from numba import jit
import vegas
from mpi4py import MPI
import time  # 导入 time 模块
import sys
from scipy.integrate import quad

# 获取当前时间，生成带时间戳的文件名
current_time = time.strftime("%m%d_%H%M%S")
output_file = f"res_{current_time}.txt" 


# 在主程序开始时记录时间
start_time = time.time()


me=0.511
e=0.303
Z=29
l=10
fact_l=math.factorial(abs(l))
# 这里sigma是坐标空间的
sigma_perp=1000
sigma_z=500     # 0.1nm左右
P_z=5
b_perp0=5000
C_in=-Z*e**3*np.pi/(2*np.pi)**(3)*(4*np.pi)**(3/4)*np.sqrt(sigma_z/fact_l)*sigma_perp*(sigma_perp)**(abs(l))
# C_in=-Z*e**3*np.pi/(2*np.pi)**(3)*(4*np.pi)**(3/4)*np.sqrt(2**l*sigma_z/fact_l)*sigma_perp/32   # no_ilphi_r 情况，且仅针对l=10

C_out=1/256/np.pi**6

# 定义 2x2 单位矩阵和泡利矩阵
I_2 = np.eye(2, dtype=np.complex128)
I_4 = np.eye(4, dtype=np.complex128)
sigma1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# 定义 Gamma 矩阵
gamma0 = np.block([[I_2, np.zeros((2, 2))], [np.zeros((2, 2)), -I_2]])
gamma1 = np.block([[np.zeros((2, 2)), sigma1], [-sigma1, np.zeros((2, 2))]])
gamma2 = np.block([[np.zeros((2, 2)), sigma2], [-sigma2, np.zeros((2, 2))]])
gamma3 = np.block([[np.zeros((2, 2)), sigma3], [-sigma3, np.zeros((2, 2))]])


# @jit(nopython=True)
# def u(epsilon, theta, phi, s):
#     w_component = w(theta, phi, s)
#     upper_part = np.sqrt(epsilon + me) * w_component
#     lower_part = 2 * np.sqrt(epsilon - me) * s * w_component
#     res = np.empty((len(upper_part) + len(lower_part), upper_part.shape[1]), dtype=np.complex128)
#     res[:len(upper_part), :] = upper_part
#     res[len(upper_part):, :] = lower_part
#     return res


# @jit(nopython=True)
# def u_f(epsilon_f, theta_f, phi_f, s_f):
#     w_dagger_component= w_dagger(theta_f, phi_f, s_f)
#     component_1 = np.sqrt(epsilon_f + me) * w_dagger_component
#     component_2 = -2 * np.sqrt(epsilon_f - me) * s_f * w_dagger_component
#     res = np.zeros(2 * len(component_1), dtype=np.complex128) 
#     res[:len(component_1)] = component_1
#     res[len(component_1):] = component_2
#     return res




@jit(nopython=True)
def photon_polar_conj(lamb,theta_k,phi_k):
    res=np.array([0,
                     1/np.sqrt(2)*(-lamb*np.cos(theta_k)*np.cos(phi_k)-1j*np.sin(phi_k)),
                     1/np.sqrt(2)*(-lamb*np.cos(theta_k)*np.sin(phi_k)+1j*np.cos(phi_k)),
                     lamb/np.sqrt(2)*np.sin(theta_k)], dtype=np.complex128)
    return res



@jit(nopython=True)
def four_vec(energy,m,theta,phi):
    p=np.sqrt(energy**2-m**2)
    res=np.array([energy,p*np.sin(theta)*np.cos(phi),p*np.sin(theta)*np.sin(phi),p*np.cos(theta)])
    return res

@jit(nopython=True)
def three_vec(energy,m,theta,phi):
    p=np.sqrt(energy**2-m**2)
    res=np.array([p*np.sin(theta)*np.cos(phi),p*np.sin(theta)*np.sin(phi),p*np.cos(theta)])
    return res

# @jit(nopython=True)
# def two_vec_mod(energy,m,theta,phi):
#     p=np.sqrt(energy**2-m**2)
#     two_vec=np.array([p*np.sin(theta)*np.cos(phi),p*np.sin(theta)*np.sin(phi)])
#     res=np.sqrt(two_vec@two_vec)
#     return res

@jit(nopython=True)
def four_vec_slash(energy,m,theta,phi):
    four_vec_=four_vec(energy,m,theta,phi)
    res= four_vec_[0]*gamma0
    res +=-four_vec_[1]*gamma1
    res +=-four_vec_[2]*gamma2
    res +=-four_vec_[3]*gamma3
    return res

@jit(nopython=True)
def photon_polar_slash(lamb,theta_k,phi_k):
    photon_polar_=photon_polar_conj(lamb,theta_k,phi_k)
    res= photon_polar_[0]*gamma0
    res +=-photon_polar_[1]*gamma1
    res +=-photon_polar_[2]*gamma2
    res +=-photon_polar_[3]*gamma3
    return res
    
@jit(nopython=True)
def u(energy,theta,phi,spin):
    fvp=four_vec(energy,me,theta,phi)
    if spin > 0:
        part1=np.sqrt(fvp[0]+me)
        part2=0
        part3=fvp[3]/np.sqrt(fvp[0]+me)
        part4=(fvp[1]+1j*fvp[2])/np.sqrt(fvp[0]+me)  
    if spin < 0:
        part1=0
        part2=np.sqrt(fvp[0]+me)
        part3=(fvp[1]-1j*fvp[2])/np.sqrt(fvp[0]+me)
        part4=(-fvp[3])/np.sqrt(fvp[0]+me)
    res=np.array([part1,part2,part3,part4],dtype=np.complex128)
    return res

@jit(nopython=True)
def u_f(energy,theta,phi,spin):
    fvp=four_vec(energy,me,theta,phi)
    if spin > 0:
        part1=np.sqrt(fvp[0]+me)
        part2=0
        part3=-fvp[3]/np.sqrt(fvp[0]+me)
        part4=-(fvp[1]-1j*fvp[2])/np.sqrt(fvp[0]+me)  
    if spin < 0:
        part1=0
        part2=np.sqrt(fvp[0]+me)
        part3=-(fvp[1]+1j*fvp[2])/np.sqrt(fvp[0]+me)
        part4=(fvp[3])/np.sqrt(fvp[0]+me)
    res=np.array([part1,part2,part3,part4],dtype=np.complex128)
    return res

@jit(nopython=True)
def Phi_cap(epsilon, theta, phi, l):
    p=np.sqrt(epsilon**2-me**2)
    two_vec_mod_=p*np.sin(theta)
    res= np.sqrt(2*epsilon)*(two_vec_mod_)**(abs(l))\
       *np.exp(-sigma_perp**2*two_vec_mod_**2/2-sigma_z**2*(p*np.cos(theta)-P_z)**2/2+1j*l*phi)
    return res

# @jit(nopython=True)
def M(epsilon,theta,phi,s,epsilon_f,theta_f,phi_f,s_f,omega,theta_k,phi_k,lamb):

    photon_polar_slash_=photon_polar_slash(lamb,theta_k,phi_k)
    four_vec_slash_k=four_vec_slash(omega,0,theta_k,phi_k)
    three_vec_i=three_vec(epsilon,me,theta,phi)
    three_vec_f=three_vec(epsilon_f,me,theta_f,phi_f)
    three_vec_k=three_vec(omega,0,theta_k,phi_k)

    res=1/((three_vec_f+three_vec_k-three_vec_i)@(three_vec_f+three_vec_k-three_vec_i))*\
                        u_f(epsilon_f,theta_f,phi_f,s_f)@\
                        (1/(2*(omega*epsilon_f-three_vec_k@three_vec_f))*photon_polar_slash_@\
                        (four_vec_slash(epsilon_f,me,theta_f,phi_f)+four_vec_slash_k+me*I_4)@gamma0\
                        +\
                        1/(-2*(omega*epsilon-three_vec_k@three_vec_i))*gamma0@\
                        (four_vec_slash(epsilon,me,theta,phi)-four_vec_slash_k+me*I_4)@photon_polar_slash_)@\
                        u(epsilon,theta,phi,s)
    return res

res=M(5.026,0.0007,5.5,0.5,3.026,np.pi/2,np.pi/3,0.5,2,np.pi/3,np.pi/3,1)
print(res)