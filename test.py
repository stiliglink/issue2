import numpy as np

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

print()