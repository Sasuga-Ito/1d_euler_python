"""
1-dimensional euler equations
"""

__author__ = 'balshark(Twitter: @balsharkPhD)'
__version__ = '1.0.0'
__date__ = '03/17/2022'
__status__ = 'Development'

from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nx0 = 100       # grid sise
xg = 1          # ghost grid

nx = nx0 + xg*2 # total grid size
nt = 300        # total time step
dx = 10**(-2)   # size between grids
dt = 10**(-3)   # time step size

gam = 1.4       # specific heat ratio

x = [0.0]*nx
ql = [0.0]*3
qr = [0.0]*3

qf = [[0.0]*3 for i in range(nx)]
qc = [[0.0]*3 for i in range(nx)]
fl = [[0.0]*3 for i in range(nx+1)]
rhs = [[0.0]*3 for i in range(nx)]

fig = plt.figure()
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.set_ylabel('r [kg/m^3]')
ax2.set_ylabel('u [m/s]')
ax3.set_ylabel('p [Pa]')
ax3.set_xlabel('x [-]')
ims = []

class Euler():
    
    def main(self):
        print('start 1-D Euler simulation')
        self.setup()

        self.calmain()

        self.imsave()

        
        
    def setup(self):
        global ql,qr,qf,qc
        print('   setup...',end='')
        r = [0.0]*nx
        u = [0.0]*nx
        p = [0.0]*nx
        e = [0.0]*nx   

        for i in range(nx):
            u[i] = 0.0

            if i <= nx/2:
                r[i] = 1.0
                p[i] = 1.0
            else:
                r[i] = 0.125
                p[i] = 0.1
            
            e[i] = p[i]/(gam-1.0)+r[i]*(u[i]**2)/2
        
        for j in range(3):
            if j == 0:
                ql[j] = r[0]
                qr[j] = r[nx-1]
            elif j == 1:
                ql[j] = r[0]*u[0]
                qr[j] = r[nx-1]*u[nx-1]
            else:
                ql[j] = e[0]
                qr[j] = e[nx-1]
        
        for i in range(nx):
            x[i] = i*dx - dx/2
            for j in range(3):
                if j == 0:
                    qf[i][j] = r[i]
                    qc[i][j] = r[i]
                elif j == 1:
                    qf[i][j] = u[i]
                    qc[i][j] = r[i]*u[i]
                else:
                    qf[i][j] = p[i]
                    qc[i][j] = e[i]
        
        print('...done')
    
    def calmain(self):
        global qc,qf,rhs

        print('   calculation...',end='')

        for n in range(nt):

            self.boundary()
            self.calrhs()

            for i in range(1,nx-1):
                qc[i] = qc[i] - dt/dx*rhs[i]
            
            self.output(n)
        
        print('...done')
        
    def boundary(self):
        global qc, qf
        for i in range(3):
            qc[0][i] = 2.0*ql[i]-qc[1][i]
            qc[nx-1][i] = qc[nx-2][i]
        
        for i in range(nx):
            qf[i][0] = qc[i][0]
            qf[i][1] = qc[i][1]/qc[i][0]
            qf[i][2] = (gam-1.0)*(qc[i][2]-0.5*qf[i][0]*qf[i][1]**2)

    def calrhs(self):
        global rhs
        self.fvs()

        for i in range(1,nx-1):
            rhs[i] = fl[i]-fl[i-1]

    def fvs(self):
        global fl
        for i in range(0,nx-1):
            R,RI,GM,GA = self.Jacb(i)
            Ap = np.dot(np.dot(R,GM+GA),RI)

            R,RI,GM,GA = self.Jacb(i+1)
            Am = np.dot(np.dot(R,GM-GA),RI)

            fl[i] = 0.5*(np.dot(Ap,qc[i])+np.dot(Am,qc[i+1]))
        

    def Jacb(self,i):
        h = (qf[i][2]+qc[i][2])/qc[i][0]
        u = qf[i][1]
        c = np.sqrt((gam-1)*(h-0.5*u**2))
        b2 = (gam-1.0)/c**2
        b1 = 0.5*b2*u**2

        R = np.array([[1.0,1.0,1.0],[u-c,u,u+c],[h-u*c,0.5*u**2,h+u*c]])
        RI = np.array([[0.5*(b1+u/c),-0.5*(1.0/c+b2*u),0.5*b2],
                       [1.0-b1,b2*u,-b2],
                       [0.5*(b1-u/c),0.5*(1.0/c-b2*u),0.5*b2]])
        GM = np.array([[u-c,0.0,0.0],[0.0,u,0.0],[0.0,0.0,u+c]])
        GA = np.array([[abs(u-c),0.0,0.0],[0.0,abs(u),0.0],[0.0,0.0,abs(u+c)]])

        return R,RI,GM,GA

    def output(self,n):
        # print('   output...',end='')
        r = [0.0]*nx
        u = [0.0]*nx
        p = [0.0]*nx
        for i in range(nx):
            r[i] = qf[i][0]
            u[i] = qf[i][1]
            p[i] = qf[i][2]
        # plt.show()
        # print('...done')
        im1, = ax1.plot(x,r,color='red')
        im2, = ax2.plot(x,u,color='green')
        im3, = ax3.plot(x,p,color='blue')

        ims.append([im1,im2,im3])
    
    def imsave(self):
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        plt.show()
        ani.save("sample.gif", writer="pillow")

if __name__ == '__main__':
    proc = Euler()
    proc.main()
