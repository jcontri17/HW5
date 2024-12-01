####################################################################################
#########################         IMPORT & FUNCTIONS          ######################
####################################################################################
import numpy as np 
import matplotlib.pyplot as plt
import time
import sys
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
from matplotlib.animation import FuncAnimation

###-----CONSTRUCT THE 2D LAPLACIAN
def Lap2D(n,L):
    e = np.ones(n)
    offsets = [-1, 0, 1] 
    A = spdiags([e, -2*e, e], offsets, n, n)
    
    A = lil_matrix(A)   #Converts to a format that allows item assignment
    
    #Apply Period Boundary Conditions
    A[0,-1] = 1
    A[-1,0] = 1

    A[0,0] = 2
    #Construct the 2D Laplacian
    A = kron(eye(n), A) + kron(A, eye(n))
    A = A.toarray()
    dx = L / n
    A = (1/(dx**2)) * A
    
    return A


###-----CONSTRUCT THE ∂_x OPERATOR
def partial_x(n,L):
    e = np.ones(n)
    offsets = [-1, 1]
    B = spdiags([-e, e], offsets, n, n)
    B = lil_matrix(B)

    #Apply Period Boundary Conditions
    B[0,-1] = -1
    B[-1,0] = 1

    #Construct the partial-x matrix
    B = kron(eye(n), B.toarray())
    B = B.toarray()

    dx = L/n
    B = (1/(2*dx)) * B
    
    return B


###-----CONSTRUCT THE ∂_y OPERATOR
def partial_y(n,L):
    e = np.ones(n)
    offsets = [-1, 1]
    C = spdiags([-e, e], offsets, n, n)
    C = lil_matrix(C)

    #Apply Period Boundary Conditions
    C[0,-1] = -1
    C[-1,0] = 1

    #Construct the partial
    C = kron(C.toarray(), eye(n))
    C = C.toarray()
    dy = L/n
    C = (1/(2*dy)) * C
    
    return C


###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs_chat(t, om_flat):
    om = om_flat.reshape((N,N))
    psi = ifft2(fft2(om) / K2).real     #!!--Solving for Stream Funciton using FFT--!!
    psi_x = np.gradient(psi, dx, axis=1)
    psi_y = np.gradient(psi, dy, axis=0)
    om_x = np.gradient(om, dx, axis=1) 
    om_y = np.gradient(om, dy, axis=0)
    Jac = psi_x*om_y - psi_y*om_x               #Compute [psi, omega]
    LapOm = np.gradient(np.gradient(om, dx, axis=1), dx, axis=1) + \
            np.gradient(np.gradient(om, dy, axis=0), dy, axis=0)

    om_t = nu*LapOm - Jac   
    #print("chat")
    #print("psi, om, Jac, LapOm, om_t")
    #print(psi.shape, om.shape, Jac.shape, LapOm.shape, om_t.shape)
    return om_t.flatten()



###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs(t, om_flat):
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
    
    psi_x_hat = 1j * kx * psi_hat       # ∂ψ/∂x in spectral space
    psi_y_hat = 1j * ky * psi_hat       # ∂ψ/∂y in spectral space
    om_x_hat = 1j * kx * om_hat         # ∂ω/∂x in spectral space
    om_y_hat = 1j * ky * om_hat         # ∂ω/∂y in spectral space

    psi_x = ifft2(psi_x_hat).real       #transform derivatives back to real space
    psi_y = ifft2(psi_y_hat).real
    om_x = ifft2(om_x_hat).real
    om_y = ifft2(om_y_hat).real

    Jac = psi_x*om_y - psi_y*om_x       #Compute [psi, omega]
    LapOm_hat = -K2 * om_hat            #compute Laplacian of omega in spectral space
    LapOm = ifft2(LapOm_hat).real       #transform back to real space

    om_t = nu * LapOm - Jac             #compute time-derivative of omega
    #print("___")
    #print("psi_hat, om_hat, Jac, LapOm, om_t")
    #print(psi_hat.shape, om_hat.shape, Jac.shape, LapOm.shape, om_t.shape)
    return om_t.flatten()



####################################################################################
##############################         PART A          #############################
####################################################################################

###-----GRID SETUP
Lx = 20                             #domain size in x-dir
Ly = 20                             #domain size in y-dir
N = 64                              #number of grid points
x = np.linspace(-Lx/2, Lx/2, N)     #discretizes x-domain by N points
y = np.linspace(-Ly/2, Ly/2, N)     #discretizes y-domain by N points
X, Y = np.meshgrid(x,y)             #returns an array of coordinates of an xy gridded mesh
dx = Lx/N                           #differential x-space
dy = Ly/N                           #differential y-space


#Define Wavenumbers
kx = fftfreq(N, d=dx)*2*np.pi       #wavenumbers in x-dir
ky = fftfreq(N, d=dy)*2*np.pi       #wavenumbers in y-dir
kx[0] = 1e-6                        #avoids singularity
ky[0] = 1e-6                        #avoids singularity
KX, KY = np.meshgrid(kx, ky)        #create 2D wavenumber grids
K2 = KX**2 + KY**2                  #Laplacian in Spectral Space
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

om_sol = solve_ivp(vorticity_rhs, tSpan, om0.flatten(), t_eval=tEval, method='RK45')
om_fin = om_sol.y[:,-1].reshape((N,N))  #reshape solution back to 2D



###-----ANSWERING THE QUESTION
A1 = om_fin
#print(A1)
np.save('A1.npy', A1)



###-----PLOTTING
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, om_fin, shading='auto', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final State of Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Part A - Vorticity FFT")
#plt.show()





print("Part A Completed")
####################################################################################
##############################         PART B          #############################
####################################################################################

#######################################
###     DIRECT SOLUTION (A/B)       ###
#######################################
start_time = time.time()

print(len(om0.flatten()))
###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs_direct(t, om_flat):
    om = om_flat
    LAP2D = Lap2D(N,Lx)
    #print(LAP2D)
    #print("LAP2D contains NaN:", np.isnan(LAP2D).any())
    #print("LAP2D contains Inf:", np.isinf(LAP2D).any())
    #print("om contains NaN:", np.isnan(om).any())
    #print("om contains Inf:", np.isinf(om).any())
    psi = solve(LAP2D,om)
    psi_x = np.dot(partial_x(N,Lx), psi)        
    psi_y = np.dot(partial_y(N,Ly), psi)
    om_x = np.dot(partial_x(N,Lx), om)
    om_y = np.dot(partial_y(N,Lx), om)
    Jac = psi_x*om_y - psi_y*om_x               #Compute [psi, omega]
    LapOm = np.dot(LAP2D, om)
    om_t = (nu*LapOm - Jac).flatten()
    
    print("Direct:", t)
    return om_t


om_sol = solve_ivp(vorticity_rhs_direct, tSpan, om0.flatten(), t_eval=tEval, method='RK45')
om_fin_direct = om_sol.y[:,-1].reshape((N,N))  #reshape solution back to 2D


end_time = time.time()
t_direct = end_time - start_time
print("Direct Completed after: ", t_direct)



###-----PLOTTING
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, om_fin_direct, shading='auto', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final State of Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Part B - Direct")
#plt.show()



###-----ANSWERING THE QUESTION
A2 = om_fin_direct
#print(A2)
np.save('A2.npy', A2)



#######################################
###     LU Decomposition            ###
#######################################
start_time = time.time()

###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs_LU(t, om_flat):
    om = om_flat
    LAP2D = Lap2D(N,Lx)
    psi = solve_lu(LAP2D, om)
    psi_x = np.dot(partial_x(N,Lx), psi)        
    psi_y = np.dot(partial_y(N,Ly), psi)
    om_x = np.dot(partial_x(N,Lx), om)
    om_y = np.dot(partial_y(N,Lx), om)
    Jac = psi_x*om_y - psi_y*om_x               #Compute [psi, omega]
    LapOm = np.dot(LAP2D, om)
    om_t = (nu*LapOm - Jac).flatten()

    print("LU Decomp:", t)

    return om_t


###-----LU DECOMPOSITION FUNCTION
def solve_lu(A, b):
    P, L, U = lu(A)
    Pb = np.dot(P,b)
    y = solve_triangular(L, Pb, lower=True)
    x = solve_triangular(U, y)

    return x


om_sol = solve_ivp(vorticity_rhs_LU, tSpan, om0.flatten(), t_eval=tEval, method='RK45')
om_fin = om_sol.y[:,-1].reshape((N,N))  #reshape solution back to 2D

end_time = time.time()
t_LU = end_time - start_time
print("LU Decomposition Completed after: ", t_LU)


###-----PLOTTING
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, om_fin, shading='auto', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final State of Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Part B - LU Decomp")
#plt.show()



###-----ANSWERING THE QUESTION
A3 = om_fin
#print(A3)
np.save('A3.npy', A3)

#######################################
###         BICGSTAB                ###
#######################################
start_time = time.time()
tol_bicgstab = 1e-6

###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs_bicgstab(t, om_flat):
    om = om_flat
    LAP2D = Lap2D(N,Lx)

    print("BICGSTAB:", t)
    psi, info_bicgstab = bicgstab(LAP2D, om, tol=tol_bicgstab)
    psi_x = np.dot(partial_x(N,Lx), psi)        
    psi_y = np.dot(partial_y(N,Ly), psi)
    om_x = np.dot(partial_x(N,Lx), om)
    om_y = np.dot(partial_y(N,Ly), om)
    Jac = psi_x*om_y - psi_y*om_x               #Compute [psi, omega]
    LapOm = np.dot(LAP2D, om)
    om_t = (nu*LapOm - Jac).flatten()

    print("BICGSTAB:", t)

    return om_t

om_sol = solve_ivp(vorticity_rhs_bicgstab, tSpan, om0.flatten(), t_eval=tEval, method='RK45')
om_fin = om_sol.y[:,-1].reshape((N,N))  #reshape solution back to 2D

end_time = time.time()
t_bicgstab = end_time - start_time
print("BICGSTAB Completed after: ", t_bicgstab)


###-----PLOTTING
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, om_fin, shading='auto', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final State of Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Part B - BICGSTAB")
#plt.show()



#######################################
###             GMRES               ###
#######################################
start_time = time.time()
tol_gmres = 1e-6

###-----FUNCTION TO SOLVE VORTICITY EQUATION USING SOLVE_IVP
def vorticity_rhs_gmres(t, om_flat):
    om = om_flat
    LAP2D = Lap2D(N,Lx)
    psi, info_gmres = gmres(LAP2D, om, tol=tol_gmres)
    psi_x = np.dot(partial_x(N,Lx), psi)        
    psi_y = np.dot(partial_y(N,Ly), psi)
    om_x = np.dot(partial_x(N,Lx), om)
    om_y = np.dot(partial_y(N,Ly), om)
    Jac = psi_x*om_y - psi_y*om_x               #Compute [psi, omega]
    LapOm = np.dot(LAP2D, om)
    om_t = (nu*LapOm - Jac).flatten()

    print("GMRES:", t)
    return om_t


om_sol = solve_ivp(vorticity_rhs_gmres, tSpan, om0.flatten(), t_eval=tEval, method='RK45')
om_fin = om_sol.y[:,-1].reshape((N,N))  #reshape solution back to 2D

end_time = time.time()
t_gmres = end_time - start_time
print("GMRES Completed after: ", t_gmres)


###-----PLOTTING
plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, om_fin, shading='auto', cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final State of Vorticity')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("Part B - GMRES")
#plt.show()

print(t_direct, t_LU, t_bicgstab, t_gmres)



print("Part B Completed")
#sys.exit()
####################################################################################
############################         PART C & D          ###########################
####################################################################################

###-----GRID SETUP
Lx = 50                             #domain size in x-dir
Ly = 50                             #domain size in y-dir
N = 100                              #number of grid points
x = np.linspace(-Lx/2, Lx/2, N)     #discretizes x-domain by N points
y = np.linspace(-Ly/2, Ly/2, N)     #discretizes y-domain by N points
X, Y = np.meshgrid(x,y)             #returns an array of coordinates of an xy gridded mesh
dx = Lx/N                           #differential x-space
dy = Ly/N                           #differential y-space

#Define Wavenumbers
kx = fftfreq(N, d=dx)*2*np.pi       #wavenumbers in x-dir
ky = fftfreq(N, d=dy)*2*np.pi       #wavenumbers in y-dir
kx[0] = 1e-6                        #avoids singularity
ky[0] = 1e-6                        #avoids singularity
KX, KY = np.meshgrid(kx, ky)        #create 2D wavenumber grids
K2 = KX**2 + KY**2                  #Laplacian in Spectral Space
K2[0, 0] = 1e-6                     #avoids singularity



###-----SOLVE THE ODE USING SOLVE_IVP
nu = 0.001                          #viscosity
tStart = 0                          #start time [s]
tEnd = 50                            #end time [s]
dt = 0.05                            #time step [s]
tSpan = (tStart, tEnd)              #time span [s]
tEval = np.arange(tStart, tEnd+dt, dt)
tol = 1e-12



###-----FUNCTION FOR GAUSSIAN VORTEX INITIALIZATION
def gaussian_vortex(x0, y0, amp, xWeight, yWeight):
    return amp * np.exp(-1*(xWeight*(X-x0)**2 + yWeight*(Y-y0)**2))



###-----FUNCTION FOR INITIALE VORTICITY IN SPECIAL CASES
def initialize_vorticity(case):
    if case == "opposite_charged":
        return gaussian_vortex(-Lx/4, 0, 1, 1, 1/20) - gaussian_vortex(Lx/4, 0, 1, 1, 1/20)
    if case == "same_charged":
        return gaussian_vortex(-Lx/4, 0, 1, 1, 1/20) + gaussian_vortex(Lx/4, 0, 1, 1, 1/20)
    if case == "colliding_pairs":
        return gaussian_vortex(-Lx/4, -Ly/4, 1, 1, 1/20) + gaussian_vortex(Lx/4, Ly/4, 1, 1, 1/20) - \
                gaussian_vortex(-Lx/4, Ly/4, -1, 1/20, 1) - gaussian_vortex(Lx/4, -Ly/4, -1, 1/20, 1)
    elif case == "random_vortices":
        om0 = np.zeros_like(X)
        numVorts = round(np.random.uniform(10,15))
        print("Number of vorticies: {}".format(numVorts))
        for _ in range(numVorts):
            x0 = np.random.uniform(-Lx/3, Lx/3)
            y0 = np.random.uniform(-Ly/3, Ly/3)
            amp = np.random.uniform(-1,1)
            xWeight = np.random.uniform(1/50, 1/2)
            yWeight = np.random.uniform(1/50, 1/2)
            om0 += gaussian_vortex(x0, y0, amp, xWeight, yWeight)
            #print(x0, y0, amp, xWeight, yWeight)
        return om0
    else:
        raise ValueError("Unknown Case.")



###-----ANIMATION FUNCITON
def animate_solution(case, fileName):
    om0 = initialize_vorticity(case)
    om_sol = solve_ivp(vorticity_rhs, tSpan, om0.flatten(), t_eval=tEval, method="RK45", rtol=tol, atol=tol)
    om_frames = om_sol.y.T.reshape(-1, N, N)
    
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.pcolormesh(X, Y, om_frames[0], shading="auto", cmap="viridis")
    fig.colorbar(cax, ax=ax, label="Vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    timeText = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    def update(frame):
        cax.set_array(om_frames[frame].flatten())
        time_Text = "Time: {:.2f} s".format(tEval[frame])
        timeText.set_text(time_Text)
        return cax, timeText
    
    print(case, len(om_frames), "/", len(tEval))

    frameInterval = int((len(tEval)-1)/100)
    ani = FuncAnimation(fig, update, frames=len(om_frames), interval=frameInterval, blit=True)

    ani.save(fileName, fps=20)
    #plt.show()
    plt.close(fig)



cases = ["opposite_charged", "same_charged", "colliding_pairs", "random_vortices"]

for case in cases:
    fileName = "{}_vorticity.mp4".format(case)
    animate_solution(case, fileName)
#case = "opposite_charged"
#case = "same_charged"
#case = "colliding_pairs"
#case = "random_vortices"
#fileName = "{}_vorticity.mp4".format(case)
#animate_solution(case, fileName)




print("Part C/D Completed")















