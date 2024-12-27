import numpy as np
from numba import jit
import math
from scipy.integrate import tplquad,quad,dblquad
from scipy.io import savemat
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

@jit(nopython=True,cache=True)
def Kro(i,j):
    if i==j : return 1
    else:return 0

@jit(nopython=True,cache=True)
def S(n,m,p,k,q):
    c=(p+q+k)/2
    if p+q>k and p+k>q and k+q>p:
        area=np.sqrt(c*(c-p)*(c-q)*(c-k))
        angle_pk=np.arccos((p**2+k**2-q**2)/(2*p*k))
        angle_kq=np.arccos((k**2+q**2-p**2)/(2*k*q))
        return np.cos(n*angle_pk-(m-n)*angle_kq)/(2*np.pi*area)
    else:
        return 0

tau=0.004
omega0=5#MeV
C=(4*omega0**2*tau**2+tau**4*(1-np.exp(-2*omega0**2/tau**2)))/(8*np.pi*omega0**2)
Z=29
alpha=(1/137)
coef=(Z**2*alpha**3)/(2*omega0)
me=0.511
pol=1#

up=np.array([1,0],dtype=np.complex128)
down=np.array([0,1],dtype=np.complex128)

@jit(nopython=True,cache=True)
def integrand(qp,the1,Ep2,the2,ele,posi,photon,l1,l2):
    Ep1=omega0-Ep2
    absp1=np.sqrt(Ep1**2-me**2)
    p1p=absp1*np.sin(the1)
    p1z=absp1*np.cos(the1)
    absp2=np.sqrt(Ep2**2-me**2)
    p2p=absp2*np.sin(the2)
    p2z=absp2*np.cos(the2)
    Ep=-Ep2
    Epp=Ep1
    pz=p1z-omega0
    ppz=omega0-p2z
    qz=omega0-p1z-p2z
    deno1=(-2*omega0*(Ep1-p1z)*(qz**2+qp**2))
    deno2=(-2*omega0*(Ep2-p2z)*(qz**2+qp**2))
    Sl2_0_0=S(l2,l2,p1p,p2p,qp)
    Sl2_p1_p1=S(l2+1,l2+1,p1p,p2p,qp)
    Sl2_m1_m1=S(l2-1,l2-1,p1p,p2p,qp)
    Sl1_0_0=S(l1,l1,p1p,p2p,qp)
    Sl1_p1_p1=S(l1+1,l1+1,p1p,p2p,qp)
    Sl1_m1_m1=S(l1-1,l1-1,p1p,p2p,qp)


    matd1=(p1z*p2z*(Ep+me)*qp)\
         *np.array([[0,1+photon],[1-photon,0]])*Sl2_0_0*Kro(l1,l2)

    matd2=(1j*p1p*p2z*(Ep+me)*qp)*np.array([[(1-photon)*Sl2_0_0*Kro(l2,l1+1),0],
                                           [0,(1+photon)*Sl2_0_0*Kro(l2,l1-1)]])

    matd3=(1j*p1z*p2p*(Ep+me)*qp)*np.array([[(1+photon)*Sl2_p1_p1*Kro(l1,l2+1),0],
                                           [0,(1-photon)*Sl2_m1_m1*Kro(l1,l2-1)]])

    matd4=(-p1p*p2p*(Ep+me)*qp)*np.array([[0,(1-photon)*Sl2_m1_m1*Kro(l1,l2-2)],
                                         [(1+photon)*Sl2_p1_p1*Kro(l1,l2+2),0]])

    matd5=(-(Ep1+me)*p2z*qp)*np.array([[1j*(1+photon)*p1p*Kro(l2,l1-1),(1+photon)*pz*Kro(l1,l2)],
                                      [(1-photon)*pz*Kro(l1,l2),1j*(1-photon)*p1p*Kro(l2,l1+1)]])*Sl2_0_0

    matd6=((Ep1+me)*p2p*qp)*np.array([[-1j*(1+photon)*pz*Sl2_p1_p1*Kro(l1,l2+1),(1+photon)*p1p*Sl2_m1_m1*Kro(l1,l2)],
                                     [(1-photon)*p1p*Sl2_p1_p1*Kro(l1,l2),-1j*(1-photon)*pz*Sl2_m1_m1*Kro(l1,l2-1)]])

    matd7=(p1z*(Ep2+me)*qp)*np.array([[-1j*(1+photon)*p1p*Kro(l2,l1-1),pz*(1+photon)*Kro(l1,l2)],
                                     [(1-photon)*pz*Kro(l1,l2),-1j*(1-photon)*p1p*Kro(l2,l1+1)]])*Sl2_0_0

    matd8=(p1p*(Ep2+me)*qp)*np.array([[1j*(1-photon)*pz*Kro(l2,l1+1),(1-photon)*p1p*Kro(l2,l1+2)],
                                     [(1+photon)*p1p*Kro(l2,l1-2),1j*(1+photon)*pz*Kro(l2,l1-1)]])*Sl2_0_0

    matd9=-((Ep1+me)*(Ep-me)*(Ep2+me)*qp)*np.array([[0,1+photon],
                                                   [1-photon,0]])*Sl2_0_0*Kro(l1,l2)

    dir_int=(1/deno1)*ele@(matd1+matd2+matd3+matd4+matd5+matd6+matd7+matd8+matd9)@posi

    mate1=((Ep1+me)*p2z*qp)*np.array([[1j*p2p*(1-photon)*Kro(l2,l1+1),ppz*(1+photon)*Kro(l1,l2)],
                                      [ppz*(1-photon)*Kro(l1,l2),1j*p2p*(1+photon)*Kro(l2,l1-1)]])*Sl1_0_0

    mate2=(p1z*(Epp-me)*p2z*qp)*np.array([[0,1+photon],
                                          [1-photon,0]])*Kro(l1,l2)*Sl1_0_0

    mate3=(p1p*(Epp-me)*p2z*qp)*np.array([[1j*Sl1_p1_p1*(1-photon)*Kro(l2,l1+1),0],
                                          [0,1j*Sl1_m1_m1*(1+photon)*Kro(l2,l1-1)]])

    mate4=(p2p*(Ep1+me)*qp)*np.array([[1j*ppz*(1+photon)*Kro(l1,l2+1),-p2p*(1-photon)*Kro(l1,l2-2)],
                                      [-p2p*(1+photon)*Kro(l1,l2+2),1j*ppz*(1-photon)*Kro(l1,l2-1)]])*Sl1_0_0

    mate5=(p2p*p1z*(Epp-me)*qp)*np.array([[1j*(1+photon)*Sl1_0_0*Kro(l1,l2+1),0],
                                          [0,1j*(1-photon)*Sl1_0_0*Kro(l1,l2-1)]])

    mate6=-(p1p*p2p*(Epp-me)*qp)*np.array([[0,(1-photon)*Sl1_p1_p1*Kro(l1,l2-2)],
                                           [(1+photon)*Sl1_m1_m1*Kro(l1,l2+2),0]])

    mate7=((Ep1+me)*(Epp+me)*(Ep2+me)*qp)*np.array([[0,-(1+photon)],
                                                    [-(1-photon),0]])*Sl1_0_0*Kro(l1,l2)

    mate8=(p1z*(Ep2+me)*qp)*np.array([[1j*(1-photon)*p2p*Kro(l1,l2-1),-(1+photon)*ppz*Kro(l1,l2)],
                                      [-(1-photon)*ppz*Kro(l1,l2),1j*(1+photon)*p2p*Kro(l1,l2+1)]])*Sl1_0_0

    mate9=(p1p*(Ep2+me)*qp)*np.array([[-1j*(1-photon)*ppz*Sl1_p1_p1*Kro(l2,l1+1),-(1+photon)*Sl1_p1_p1*p2p*Kro(l1,l2)],
                                      [-(1-photon)*Sl1_m1_m1*p2p*Kro(l2,l1),-1j*(1+photon)*Sl1_m1_m1*ppz*Kro(l2,l1-1)]])

    exc_int=(1/deno2)*ele@(mate1+mate2+mate3+mate4+mate5+mate6+mate7+mate8+mate9)@posi

    return dir_int+exc_int

@jit(nopython=True,cache=True)
def real_integrand(qp,the1,Ep2,the2,ele,posi,photon,l1,l2):
    return np.real(integrand(qp,the1,Ep2,the2,ele,posi,photon,l1,l2))

@jit(nopython=True,cache=True)
def imag_integrand(qp,the1,Ep2,the2,ele,posi,photon,l1,l2):
    return np.imag(integrand(qp,the1,Ep2,the2,ele,posi,photon,l1,l2))

def diff_pro(the1,Ep2,the2,ele,posi,photon,l1,l2):
    Ep1=omega0-Ep2
    absp1=np.sqrt(Ep1**2-me**2)
    p1p=absp1*np.sin(the1)
    absp2=np.sqrt(Ep2**2-me**2)
    p2p=absp2*np.sin(the2)
    real_IM=quad(real_integrand,np.abs(p1p-p2p),p1p+p2p,args=(the1,Ep2,the2,ele,posi,photon,l1,l2),epsabs = 1e-10)
    imag_IM=quad(imag_integrand,np.abs(p1p-p2p),p1p+p2p,args=(the1,Ep2,the2,ele,posi,photon,l1,l2),epsabs = 1e-10)
    InvariantM=real_IM[0]+1j*imag_IM[0]
    res=C*coef*(p1p*p2p)/((Ep1+me)*(Ep2+me))*np.abs(InvariantM)**2
    return res

Integrand=lambda the1,the2,Ep2,ele,posi,photon,l1,l2:diff_pro(the1,Ep2,the2,ele,posi,photon,l1,l2)

comm.Barrier()
sizel1=11
sizel2=11
TPP=sizel1*sizel2
l1 = np.linspace(-5, 5, sizel1)
l2 = np.linspace(-5, 5, sizel2)
L2, L1 = np.meshgrid(l2, l1)
if rank==0:
    linL1=L1.reshape(TPP)
    linL2=L2.reshape(TPP)
else:
    linL1 = None
    linL2 = None

quotient = math.floor(TPP / size)
remainder = TPP % size

local_begin = np.zeros(1)
local_end = np.zeros(1)
local_length = np.zeros(1)
comm.Barrier()
if rank >= remainder:
    local_begin[0] = remainder * (quotient + 1) + (rank - remainder) * quotient
    local_end[0] = remainder * (quotient + 1) + (rank - remainder) * quotient + (quotient - 1)
    local_length[0] = local_end-local_begin+1
elif rank < remainder:
    local_begin[0] = rank * (quotient + 1)
    local_end[0] = rank * (quotient + 1) + quotient
    local_length[0] = local_end - local_begin + 1
comm.Barrier()

if rank == 0:
    lengthcount = np.zeros(size)
    begincount = np.zeros(size)
else:
    lengthcount = None
    begincount = None

comm.Gatherv(local_length, [lengthcount, MPI.DOUBLE])
comm.Gatherv(local_begin, [begincount, MPI.DOUBLE])
comm.Barrier()

if rank == 0:
    begincount = begincount.astype(int)
    lengthcount = lengthcount.astype(int)
    #print(begincount,lengthcount)

locall1 = np.zeros(int(local_length[0]))
locall2 = np.zeros(int(local_length[0]))
comm.Scatterv([linL1, lengthcount, begincount, MPI.DOUBLE], locall1)
comm.Scatterv([linL2, lengthcount, begincount, MPI.DOUBLE], locall2)
localsp = np.zeros(int(local_length[0]))
print(rank,locall1,locall2,localsp)
comm.Barrier()

for i in np.arange(0, int(local_length[0])):
    for ele in [up, down]:
        for posi in [up, down]:
            localsp[i]=localsp[i]+tplquad(Integrand,me+0.01,omega0-me-0.1,
                                            lambda Ep2:np.pi/180,lambda Ep2:179*np.pi/180,
                                            lambda Ep2,the2:np.pi/180,lambda Ep2,the2:179*np.pi/180,
                                            args=(ele, posi, pol, locall1[i], locall2[i]))[0]

comm.Barrier()

if rank==0:
    linsp=np.zeros(TPP)
else:
    linsp=None

comm.Barrier()
comm.Gatherv(localsp, [linsp, lengthcount, begincount, MPI.DOUBLE])

if rank==0:
    sp=linsp.reshape((sizel1,sizel2))
    mdic = {"core": size, 'sp':sp,'l1':l1,'l2':l2}
    savemat('lam1_OAMspectrum.mat')
