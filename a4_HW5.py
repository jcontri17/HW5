###################################
###     IMPORTS & FUNCTIONS     ###
###################################
import numpy as np 
import matplotlib.pyplot as plt
import time
from scipy.fft import fft
from scipy.fft import ifft
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.integrate import solve_ivp
from scipy.linalg import lu
from scipy.linalg import solve_triangular
from scipy.linalg import solve
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import gmres
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import kron
from scipy.sparse import eye
from HW4_main import Lap2D
from HW4_main import partial_x
from HW4_main import partial_y


###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs(t, om_flat):
    kx = fftfreq(N, d=Lx / N) * 2 * np.pi   #Wavenumbers in x-direction
    ky = fftfreq(N, d=Ly / N) * 2 * np.pi   #Wavenumbers in y-direction
    kx[0] = 1e-6                            #avoids singularity
    ky[0] = 1e-6                            #avoids singularity
    kx, ky = np.meshgrid(kx, ky)            #Create 2D wavenumber grids
    K2 = kx**2 + ky**2                      #Laplacian in spectral space
    K2[0, 0] = 1e-6                         #Avoid division by zero for the zero mode
    #print(K2)
    om = om_flat.reshape((N,N))         #reshape omega in to (N, N) grid
    om_hat = fft2(om)                   #transform omega to spectral space
    psi_hat = -om_hat / K2              #solve for stream function in spectral space
    psi = ifft2(psi_hat).real

    psi_x_hat = 1j * kx * psi_hat       # ∂ψ/∂x in spectral space
    psi_y_hat = 1j * ky * psi_hat       # ∂ψ/∂y in spectral space
    om_x_hat = 1j * kx * om_hat         # ∂ω/∂x in spectral space
    om_y_hat = 1j * ky * om_hat         # ∂ω/∂y in spectral space

    psi_x = ifft2(psi_x_hat).real       #transform derivatives back to real space
    psi_y = ifft2(psi_y_hat).real
    om_x = ifft2(om_x_hat).real
    om_y = ifft2(om_y_hat).real

    Jac = psi_x*om_y - psi_y*om_x       #Compute [psi, omega]
    #om_x = ifft2(1j*kx*fft2(om)).real
    #Jac_hat = psi_x_hat * om_y_hat - psi_y_hat * om_x_hat
    #Jac = ifft2(Jac_hat).real

    LapOm_hat = -K2 * om_hat            #compute Laplacian of omega in spectral space
    LapOm = ifft2(LapOm_hat).real       #transform back to real space
    om_t = nu * LapOm - Jac             #compute time-derivative of omega
    
    idx = 1
    with open("debug_rhs.txt", "w") as f:
        f.write(f"{om_hat[idx,idx]}, {psi[idx,idx]}, {psi_y[idx,idx]}, {om_x[idx,idx]}, {LapOm[idx,idx]}, {Jac[idx,idx]}")
    print(psi_x.shape)
    print(psi_x)
    #print("___")
    #print("psi_hat, om_hat, Jac, LapOm, om_t")
    #print(psi_hat.shape, om_hat.shape, Jac.shape, LapOm.shape, om_t.shape)
    return om_t.flatten()



###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs_chat(t, om_flat):
    om = om_flat.reshape((N,N))
    om_hat = fft2(om)
    om_fft = ifft2(om_hat/(N*N)).real

    diff = om - om_fft
    psi = ifft2(-om_hat / K2).real     #!!--Solving for Stream Funciton using FFT--!!
    psi_x = np.gradient(psi, dx, axis=1)
    psi_y = np.gradient(psi, dy, axis=0)
    om_x = np.gradient(om, dx, axis=1) 
    om_y = np.gradient(om, dy, axis=0)
    Jac = psi_x*om_y - psi_y*om_x               #Compute [psi, omega]
    LapOm = np.gradient(np.gradient(om, dx, axis=1), dx, axis=1) + \
            np.gradient(np.gradient(om, dy, axis=0), dy, axis=0)
    
    print(om)
    idx = 1
    with open("debug_chat.txt", "w") as f:
        f.write(f"{om_hat[idx,idx]}, {psi[idx,idx]}, {psi_y[idx,idx]}, {om_x[idx,idx]}, {LapOm[idx,idx]}, {Jac[idx,idx]}")
    
    om_t = nu*LapOm - Jac   
    #print("psi, om, Jac, LapOm, om_t")
    #print(psi.shape, om.shape, Jac.shape, LapOm.shape, om_t.shape)
    return om_t.flatten()


def vorticity_new(t, om_flat):
    kx = fftfreq(N, d=Lx / N) * 2 * np.pi   #Wavenumbers in x-direction
    ky = fftfreq(N, d=Ly / N) * 2 * np.pi   #Wavenumbers in y-direction
    kx[0] = 1e-6                            #avoids singularity
    ky[0] = 1e-6                            #avoids singularity
    kx, ky = np.meshgrid(kx, ky)            #Create 2D wavenumber grids
    K2 = kx**2 + ky**2                      #Laplacian in spectral space
    K2[0, 0] = 1e-6                         #Avoid division by zero for the zero mode

    om = om_flat.reshape((N,N))         #reshape omega in to (N, N) grid
    om_hat = fft2(om)                   #transform omega to spectral space
    psi_hat = -om_hat / K2              #solve for stream function in spectral space
    psi = np.real(ifft2(psi_hat))

    psi_x_hat = 1j * kx * psi_hat       # ∂ψ/∂x in spectral space
    psi_y_hat = 1j * ky * psi_hat       # ∂ψ/∂y in spectral space
    om_x_hat = 1j * kx * om_hat         # ∂ω/∂x in spectral space
    om_y_hat = 1j * ky * om_hat         # ∂ω/∂y in spectral space

    psi_x = np.real(ifft2(psi_x_hat))      #transform derivatives back to real space
    psi_y = np.real(ifft2(psi_y_hat))
    om_x = np.real(ifft2(om_x_hat))
    om_y = np.real(ifft2(om_y_hat))

    Jac = psi_x*om_y - psi_y*om_x       #Compute [psi, omega]
    LapOm_hat = -K2 * om_hat            #compute Laplacian of omega in spectral space
    LapOm = ifft2(LapOm_hat).real       #transform back to real space

    om_t = nu * LapOm - Jac             #compute time-derivative of omega
    #print("new")
    idx = 1
    with open("debug_new.txt", "w") as f:
        f.write(f"{om_hat[idx,idx]}, {psi[idx,idx]}, {psi_y[idx,idx]}, {om_x[idx,idx]}, {LapOm[idx,idx]}, {Jac[idx,idx]}")
    #print("psi_hat, om_hat, Jac, LapOm, om_t")
    #print(psi_hat.shape, om_hat.shape, Jac.shape, LapOm.shape, om_t.shape)
    return om_t.flatten()

######################################
##########      PART A      ##########
######################################

###-----GRID SETUP
Lx = 20                             #domain size in x-dir
Ly = 20                             #domain size in y-dir
N = 64                              #number of grid points
x = np.linspace(-Lx/2, Lx/2, N)     #discretizes x-domain by N points
y = np.linspace(-Ly/2, Ly/2, N)     #discretizes y-domain by N points
X, Y = np.meshgrid(x,y)             #returns an array of coordinates of an xy gridded mesh
dx = Lx/N                           #differential x-space
dy = Ly/N                           #differential y-space
tol = 1e-8

#Define Wavenumbers
kx = fftfreq(N, d=dx)*2*np.pi       #wavenumbers in x-dir
ky = fftfreq(N, d=dy)*2*np.pi       #wavenumbers in y-dir
kx[0] = 1e-6                        #avoids singularity
ky[0] = 1e-6                        #avoids singularity
kx, ky = np.meshgrid(kx, ky)        #create 2D wavenumber grids
K2 = kx**2 + ky**2                  #Laplacian in Spectral Space
K2[0, 0] = 1e-6                     #avoids singularity


###-----INITIAL CONDITION: GAUSSIAN MOUND
xWeight = 1         #x-weight to play around with initial condition
yWeight = 1/20      #y-weight to play around with initial condition
om0 = np.exp(-1*(xWeight*X**2 + yWeight*Y**2))    #inital vorticity




###-----SOLVE THE ODE USING SOLVE_IVP
nu = 0.001                          #viscosity
tStart = 0                          #start time [s]
tEnd = 4                            #end time [s]
dt = 0.5                            #time step [s]
tSpan = (tStart, tEnd)              #time span [s]

tEval = np.arange(tStart, tEnd+dt, dt)
#tEval = np.linspace(tStart, tEnd, 9)

startTime = time.time()
om_sol = solve_ivp(vorticity_new, tSpan, om0.flatten(), t_eval=tEval, method='RK45', atol=tol, rtol=tol)
om_fin = om_sol.y[:,-1].reshape((N,N))  #reshape solution back to 2D
endTime = time.time()

print("")
print("Total Time:", endTime-startTime)


###-----ANSWERING THE QUESTION
A1 = om_fin
print(A1)
print(np.min(A1), np.max(A1))
np.save('A1.npy', A1)



###-----PLOTTING
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, om_fin, shading='auto', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final State of Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Part A - Vorticity FFT")
plt.show()





print("Part A Completed")
