import numpy as n
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib
import cmath
import scipy.integrate as integ
import sympy as sym
import copy
from IPython.display import display
kpath = ("Z", r"$\Gamma$", "Z")

#printing style choice
sym.init_printing()


#%% Infinite chain energy, 1 atom cell

#Hamiltonian Parameters
a=1
tss = 1
tpp = -tss
tsp = tss
delta = 0



#Energy Eigenvalues
def bonding_energy(k,a):
    return (tpp-tss) * n.cos(a*k) - 0.5*cmath.sqrt( (2*(tss+tpp)*n.cos(a*k) + delta)**2 + (4*tsp*n.sin(a*k))**2 )

def antibonding_energy(k,a):
    return (tpp-tss) * n.cos(a*k) + 0.5*cmath.sqrt( (2*(tss+tpp)*n.cos(a*k) + delta)**2 + (4*tsp*n.sin(a*k))**2 )

def bonding_s(k,a):
    return -2*tss*n.cos(a*k) - delta/2

def antibonding_p(k,a):
    return 2*tpp*n.cos(a*k) + delta/2

#Calculating and plotting the energy bands
k_array = n.linspace(-n.pi,n.pi,50)
bonding_energy_array = [(bonding_energy(i,a)) for i in k_array]
antibonding_energy_array = [(antibonding_energy(i,a)) for i in k_array]
bonding_s_array = [(bonding_s(i,a)) for i in k_array]
antibonding_p_array = [(antibonding_p(i,a)) for i in k_array]


#Plotting the energy eigenvalues
plt.plot(k_array, bonding_energy_array, linewidth = 5, label = r"Bonding")
plt.plot(k_array, antibonding_energy_array,linewidth = 5, label = r"Anti-Bonding")
plt.plot(k_array, bonding_s_array, linewidth = 5, color = "black",linestyle = "dashed", label = r"$Pure \; d_{xy} \; (t' = 0)$")
plt.plot(k_array, antibonding_p_array, linewidth = 5, color = "black", label = r"$Pure \; d_{x^2-y^2} \; (t' = 0)$")

plt.xlabel(r"$k/a$", fontsize=15)
plt.ylabel(r"E (eV)", fontsize=15)
plt.legend(loc="upper right")
plt.grid()
plt.show()
plt.close()






#%% Symbolic No Mixing Infinite chain, 2 atom cell, 4-dimensional basis (dxy1, dx^2_1, dxy2, dx^2_2 order)

chosen_gamma = 0
#Declaring my variables and the Hamiltonian, and getting the evals
k = sym.Symbol('k', real=True)
t = sym.Symbol('t', real=True)
T = sym.Symbol('T', real=True)
A = sym.Symbol('A', real=True)
B = sym.Symbol('B', real=True)
C = sym.Symbol('C', real=True)
#H = sym.Matrix([ [0,0,-2*t*sym.cos(k/2) - T*sym.exp(sym.I*k/2),0], [0,0,0,-2*t*sym.cos(k/2) - T*sym.exp(-sym.I*k/2)], [-2*t*sym.cos(k/2) - T*sym.exp(-sym.I*k/2), 0, 0, 0], [0, -2*t*sym.cos(k/2) - T*sym.exp(sym.I*k/2), 0, 0] ])

#H = sym.Matrix([ [0,0,-t*(1+sym.exp(-sym.I*k))-T,0] , [0,0,0,-t*(1+sym.exp(-sym.I*k)) - T*sym.exp(-sym.I*k)] , [-t*(1+sym.exp(sym.I*k))-T,0,0,0], [0,-t*(1+sym.exp(sym.I*k)) - T*sym.exp(sym.I*k), 0 , 0] ])
H = sym.Matrix([ [0,-t*(1+sym.exp(-sym.I*k))-T*(1-sym.exp(-sym.I*k)),0,0] , [-t*(1+sym.exp(sym.I*k))-T*(1-sym.exp(sym.I*k)),0,0,0] , [0,0,0,-t*(1+sym.exp(-sym.I*k))+T*(1-sym.exp(-sym.I*k))], [0, 0, -t*(1+sym.exp(sym.I*k))+T*(1-sym.exp(sym.I*k)) , 0] ])

diag_H = H.eigenvects()


#Showing H, the evals and evecs as images
display(H)


display(diag_H[0][0])
display(diag_H[0][2])
display(diag_H[1][0])
display(diag_H[1][2])

#%% Symbolic With Mixing Infinite chain, 2 atom cell, 4-dimensional basis (dxy1, dx^2_1, dxy2, dx^2_2 order)

chosen_gamma = sym.pi/8

#Declaring my variables and the Hamiltonian, and getting the evals
k = sym.Symbol('k', real=True)
t = sym.Symbol('t', real=True)
T = sym.Symbol('T', real=True)
gamma = sym.Symbol('gamma', real=True)
H = sym.Matrix([ [0,-t*(1+sym.exp(-sym.I*k))-T*(1-sym.exp(-sym.I*k)*(sym.cos(4*gamma)+sym.sin(4*gamma))),0,T*sym.sin(4*gamma)*sym.exp(-sym.I*k)] , [-t*(1+sym.exp(sym.I*k))-T*(1-sym.exp(sym.I*k)*(sym.cos(4*gamma)+sym.sin(4*gamma))),0,T*sym.sin(4*gamma)*sym.exp(sym.I*k),0] , [0,T*sym.sin(4*gamma)*sym.exp(-sym.I*k),0,-t*(1+sym.exp(-sym.I*k))+T*(1-sym.exp(-sym.I*k)*(sym.cos(4*gamma)-sym.sin(4*gamma)))], [T*sym.sin(4*gamma)*sym.exp(sym.I*k), 0, -t*(1+sym.exp(sym.I*k))+T*(1-sym.exp(sym.I*k)*(sym.cos(4*gamma)-sym.sin(4*gamma))) , 0] ])

H = H.subs(gamma, chosen_gamma)
display(H)
evals_symbolic = H.eigenvals(simplify = True)

diag_H = H.eigenvects()


#Printing the Evals
counter = 0
for eigenval in evals_symbolic:
    
    if counter <= 1:
        
        ## Simplification for gamma = 0
        if chosen_gamma == 0:
            
            eigenval = eigenval.expand()
            eigenval = eigenval.rewrite(sym.cos)
            eigenval = eigenval**2
            eigenval = eigenval.simplify()
            eigenval = eigenval.factor(sym.cos(k))
            
        
        if chosen_gamma == sym.pi/8:
            pass 
        
            eigenval = eigenval.expand()
            #eigenval = eigenval.expand(complex=True)
            eigenval = eigenval**2
            #eigenval = eigenval.rewrite(sym.cos)


            eigenval = sym.collect(eigenval, t)
            eigenval = sym.collect(eigenval, T)
            
            
            #eigenval = sym.trigsimp(eigenval)
            #eigenval = eigenval.rewrite(sym.cos)
            #eigenval = sym.simplify(eigenval)
            
            ###Simplifying all arguments separately
            eigenval_args = eigenval.args
            args = [i for i in eigenval_args]

            #Simple part outside square root
            args[2] = args[2].rewrite(sym.cos).factor()
            
            #Complicated square root part
            args_sq = [i for i in args[1].expand().args]
            args_sq[3] = args_sq[3]**2
            args_sq[3] = args_sq[3].expand()

            
            args_sq[3] = sym.re(args_sq[3])
            args_sq[3] = args_sq[3].factor()
            args_sq[3] = args_sq[3].trigsimp()
            args_sq[3] = args_sq[3].collect(sym.cos(k))
            #args_sq[3] = args_sq[3].collect(-16*T**2*t*(T-t))
            #args_sq[3] = args_sq[3].simplify()
            #args_sq[3] = sym.sqrt(args_sq[3])

            display(args_sq[3]) 
            


            display(args[1])
            

        
        #display(eigenval)
        
        
        print(eigenval.evalf(subs={k:1.2, t:1, T:-2}))
        
    counter += 1


#%% Calculating and plotting the bands and evecs (from symbolic Hamiltonian)

#Creating Figures
k_array = n.linspace(-n.pi,n.pi,1000)
multiplier_res = 7
dpi = 300
fig_all, ax_all = plt.subplots(nrows=1, ncols=3, figsize=(3*1.6*multiplier_res, 1*multiplier_res), dpi=dpi)
fig_band, ax_band = plt.subplots(figsize=(1.6*multiplier_res, 1*multiplier_res), dpi=dpi)
fig_xy, ax_xy = plt.subplots(figsize=(1.6*multiplier_res, 1*multiplier_res), dpi=dpi)
fig_x2y2, ax_x2y2 = plt.subplots(figsize=(1.6*multiplier_res, 1*multiplier_res), dpi=dpi)

#chosen values of hopping
t_mn= -2

T_value = -1
phase_factor = cmath.exp(1j * 0)
dxy_evec_plot_index = 0
dx2y2_evec_plot_index = 1



    
    
# Band indices based on if there is degeneracy or not
if chosen_gamma == 0:
    number_of_bands = 2
    b_eigenvector_1 = diag_H[0][2][0].normalized()
    b_eigenvector_2 = diag_H[0][2][1].normalized()
    ab_eigenvector_1 = diag_H[1][2][0].normalized()
    ab_eigenvector_2 = diag_H[1][2][1].normalized()
    evecs = [[sym.lambdify([k,t,T],b_eigenvector_1[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)], [sym.lambdify([k,t,T],b_eigenvector_2[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)], [sym.lambdify([k,t,T],ab_eigenvector_1[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)], [sym.lambdify([k,t,T],ab_eigenvector_2[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)]]
    evecs[0][2] = n.zeros(len(k_array))
    evecs[0][3] = n.zeros(len(k_array))
    evecs[1][0] = n.zeros(len(k_array))
    evecs[1][1] = n.zeros(len(k_array))
    evecs[2][2] = n.zeros(len(k_array))
    evecs[2][3] = n.zeros(len(k_array))
    evecs[3][0] = n.zeros(len(k_array))
    evecs[3][1] = n.zeros(len(k_array))
else:
    number_of_bands = 4
    evecs = n.array([ sym.lambdify([k,t,T],diag_H[band_index][2][0].normalized(), "numpy")(k_array,t_mn,T_value)[:] for band_index in range(number_of_bands)])


#Unmixing the bands/evecs if necessary!!! evec indexing: [band][orbital][idk what this is but put 0 lol][k]
bands = n.array([sym.lambdify([k,t,T],diag_H[i][0], "numpy")(k_array,t_mn,T_value).real for i in range(number_of_bands)])
evecs_unmixed = copy.deepcopy(evecs)
bands_unmixed = copy.deepcopy(bands)


#Unmixing the bands and evecs!
for band_index in [0,1]:
    
    previous_diff = 0.3
    for k_idx in range(len(k_array)-1):
        
        if n.abs(bands[band_index][k_idx+1] - bands[band_index][k_idx]) > 2*previous_diff:
            
            # Swapping the bands
            lower_band_segment = n.copy(bands[band_index][k_idx+1:])
            bands[band_index][k_idx+1:] = n.copy(bands[band_index+2][k_idx+1:])
            bands[band_index+2][k_idx+1:] = lower_band_segment

            #swapping the evecs
            for orb in range(4):
                lower_evec_segment = n.copy(evecs[band_index][orb][0][k_idx+1:])
                evecs[band_index][orb][0][k_idx+1:] = n.copy(evecs[band_index+2][orb][0][k_idx+1:])
                evecs[band_index+2][orb][0][k_idx+1:] = lower_evec_segment
            
            #print(f"Swapped bands {band_index} and {band_index + 2}, from k={k_idx+1} onwards")


#### Plotting Bands
for ax in [ax_band, ax_all[0]]:
    for i in range(number_of_bands):
        ax.plot(k_array, bands[i], linewidth = 8)
    ax.set_ylabel(r"E (eV)", fontsize=50)
    ax.set_xticks([min(k_array), 0, max(k_array)])
    ax.set_xlim(min(k_array),max(k_array))
    ax.set_xticklabels(kpath)
    ax.tick_params(axis="both", labelsize = 40)
    ax.grid()

#### Plotting Eigenstates

#dxy
xy_label_array= [r"$Re(\alpha_{xy,1})$", r"$Im(\alpha_{xy,1})$",r"$Re(\alpha_{xy,2})$", r"$Im(\alpha_{xy,2})$"]
for ax in [ax_xy, ax_all[1]]:
    for i in range(2):
        
        if number_of_bands == 2:
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][:].real, linewidth = 8, label = xy_label_array[2*i])
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][:].imag, linewidth = 8, label = xy_label_array[2*i+1])
        else:
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][0][:].real, linewidth = 8, label = xy_label_array[2*i])
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][0][:].imag, linewidth = 8, label = xy_label_array[2*i+1])

    ax.set_xticks([min(k_array), 0, max(k_array)])
    ax.set_yticks([-0.7,0,0.7])
    ax.set_xlim(min(k_array),max(k_array))
    ax.set_ylim(-0.8,0.8)
    ax.set_xticklabels(kpath)
    ax.tick_params(axis="both", labelsize = 40)
    ax.legend(loc="lower left", fontsize = 25)
    ax.grid()


#dx^2-y^2
x2y2_label_array= [r"$Re(\alpha_{x^2-y^2,1})$", r"$Im(\alpha_{x^2-y^2,1})$",r"$Re(\alpha_{x^2-y^2,2})$", r"$Im(\alpha_{x^2-y^2,2})$"]
for ax in [ax_x2y2, ax_all[2]]:
    for i in range(2):
        
        if number_of_bands == 2:
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][:].real, linewidth = 8, label = x2y2_label_array[2*i])
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][:].imag, linewidth = 8, label = x2y2_label_array[2*i+1])
        else:
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][0][:].real, linewidth = 8, label = x2y2_label_array[2*i])
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][0][:].imag, linewidth = 8, label = x2y2_label_array[2*i+1])

    ax.set_xticks([min(k_array), 0, max(k_array)])
    ax.set_yticks([-0.7,0,0.7])
    ax.set_xlim(min(k_array),max(k_array))
    ax.set_ylim(-0.8,0.8)
    ax.set_xticklabels(kpath)
    ax.tick_params(axis="both", labelsize = 40)
    ax.legend(loc="lower left", fontsize = 25)
    ax.grid()


# Winding Number Calculation
chosen_bands = [0,2]
for chosen_band in chosen_bands:
    wind_evec = evecs[chosen_band]
    if number_of_bands == 2:
        wind_evec_diff = [n.gradient(wind_evec[orb], k_array)for orb in range(4)]
        wind_integrand = [ n.sum([1j*n.conjugate(wind_evec[orb][k_idx]) * wind_evec_diff[orb][k_idx] for orb in range(4)]).real for k_idx in range(len(k_array))]
    
    else:
        wind_evec_diff = [n.gradient(wind_evec[orb][0], k_array)for orb in range(4)]
        wind_integrand = [ n.sum([1j*n.conjugate(wind_evec[orb][0][k_idx]) * wind_evec_diff[orb][k_idx] for orb in range(4)]).real for k_idx in range(len(k_array))]
    
        
    winding_number = n.trapz(wind_integrand, k_array)/(n.pi)
    
    print(f"Winding Number Band {chosen_band}: {winding_number:.3f}")

#Chossing which figs to plot
#plt.close(fig_all)
plt.close(fig_band)
plt.close(fig_xy)
plt.close(fig_x2y2)




#%% Finite chain, 4-dim basis  

#number of unit cells
N = 200

#Parameter values
t_value = 1
T_value = 1
gamma = 0.74 * n.pi/180 * 0
gamma = n.pi/16 #maximum mixing
chemical_potential = 0

t_value_array = n.linspace(-5,5, 101)
#t_value_array = [-1.4]

#Looping over parameters
for t_value in t_value_array:

    #Definining the Real Space Hamiltonian Array (d1, d2, D2, D2 basis order)
    H = n.zeros((4*N,4*N))
    
    
    for i in range(0,4*N):
        #chemical potential in diagonal entries
        H[i,i] = -chemical_potential
        
        #xy-xy in-cell hopping
        if 4*i < 4*N-1:
            H[4*i,4*i + 1] = -t_value - T_value
            H[4*i + 1,4*i] = -t_value - T_value
        
        #x^2-y^2 - x^2-y^2 in-cell hopping
        if 4*i + 2 < 4*N-1:  
            H[4*i + 2,4*i + 3] = -t_value + T_value
            H[4*i + 3,4*i + 2] = -t_value + T_value
            
        #xy-xy inter-cell hopping
        if 4*i +4 < 4*N-1:
            H[4*i + 1,4*i + 4] = -t_value + T_value*(n.cos(4*gamma) + n.sin(4*gamma))
            H[4*i + 4,4*i + 1] = -t_value + T_value*(n.cos(4*gamma) + n.sin(4*gamma))
            
        #x^2-y^2-x^2-y^2 inter-cell hopping
        if 4*i +6 < 4*N-1:
            H[4*i + 3,4*i + 6] = -t_value - T_value*(n.cos(4*gamma) - n.sin(4*gamma))
            H[4*i + 6,4*i + 3] = -t_value - T_value*(n.cos(4*gamma) - n.sin(4*gamma))
            
        #xy-x^2-y^2 inter-cell hopping
        if 4*i +3 < 4*N-1:
            H[4*i + 3,4*i + 4] = T_value*n.sin(4*gamma)
            H[4*i + 4,4*i + 3] = T_value*n.sin(4*gamma)
        if 4*i +6 < 4*N-1:
            H[4*i + 1,4*i + 6] = T_value*n.sin(4*gamma)
            H[4*i + 6,4*i + 1] = T_value*n.sin(4*gamma) 
        
        
        
    #diagonalizing the madeltax (evec indexing: [orbital][band])
    evals,evecs = la.eigh(H)
    evals=evals.real
    
    
    #finding the zero energy states, and plotting the associated eigenstates
    zero_energy_indices = [idx for idx,i in enumerate(evals) if n.abs(i)<0.01]
    
    #Finding the edge states
    edge_tol = 0.3
    edge_states = [i for i in range(len(evecs[:,0])) if edge_tol < n.sum([n.abs(evecs[orb][i])**2 for orb in range(8)]) or edge_tol < n.sum([n.abs(evecs[-(orb+1)][i])**2 for orb in range(8)])  ]
    edge_state_energies = sorted( (evals[i], i) for i in edge_states)
    
    #Plotting energy eigenvalues
    for idx, i in enumerate(evals):
        if idx in edge_states:
            plt.axhline(i, color = "red", zorder = 2)
        else:
            plt.axhline(i, zorder = 1)
    plt.ylabel("Energy (eV)", fontsize=15)
    plt.tick_params(axis = 'y', labelsize = 15)
    plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    plt.grid(True)
    #plt.ylim(1.8,2.1)
    plt.annotate(f"t = {t_value:.2f}, t'={T_value:.2f}, nu = {len(edge_states)//2}", (0.23, 0.65), xycoords = "figure fraction", fontsize = 20, color = "black")
    plt.show()
    plt.close()

    
    '''
    #Plotting the chosen eigenstates
    eigenstate_index_array = zero_energy_indices
    #eigenstate_index_array = edge_states
    label_array = ("Left Edge State","Right Edge State")
    for i in eigenstate_index_array:
        evec_d_array = n.array([evecs[j][i] for j in range(len(evecs[:,0])) if j%4==0 or (j-1)%4 ==0])
        evec_D_array = n.array([evecs[j][i] for j in range(len(evecs[:,0])) if (j-2)%4==0 or (j-3)%4 ==0])
        plt.plot(n.linspace(1,2*N,2*N),evec_d_array**2 + evec_D_array**2, label=label_array[(i+1)%2])
    plt.xlabel("Chain Site", fontsize=15)
    plt.ylabel(r"$|\Psi|^2$", fontsize=15)
    plt.tick_params(axis = 'y', labelsize = 15)
    plt.tick_params(axis="x", labelsize=15)
    #plt.xticks(n.arange(1,max(plt.xticks()[0])-1,step=1))
    #plt.xlim(0,10)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
    '''
    
    '''
    #Plotting the chosen eigenstates' orbital contributions
    eigenstate_index_array = zero_energy_indices
    label_array = ("Left Edge State (dxy)", "Left Edge State (dx^2-y^2)","Right Edge State (dxy)" ,"Right Edge State (dx^2-y^2)")
    a = 0
    for i in eigenstate_index_array:
        evec_d_array = n.array([evecs[i][j] for j in range(len(evecs[0])) if j%4==0 or (j-1)%4 ==0])
        evec_D_array = n.array([evecs[i][j] for j in range(len(evecs[0])) if (j-2)%4==0 or (j-3)%4 ==0])
        plt.plot(n.linspace(1,2*N,2*N),evec_d_array**2, label=label_array[2*a])
        plt.plot(n.linspace(1,2*N,2*N),evec_D_array**2, label=label_array[2*a+1])
        a += 1
    plt.xlabel("Chain Site", fontsize=15)
    plt.ylabel(r"$|\Psi|^2$", fontsize=15)
    plt.tick_params(axis = 'y', labelsize = 15)
    plt.tick_params(axis="x", labelsize=15)
    #plt.xticks(n.arange(1,max(plt.xticks()[0])-1,step=1))
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
    
    #Printing the evals and their indices
    for i in eigenstate_index_array:
        print("E = %.4f eV, index = %d" % (evals[i], i))
    
    print("GS Energy: %.2f eV" % min(evals))
    
    '''
    
    
    #Printing the topological edge states and their energy
    print(f"!!!There are {len(edge_states)} Edge States (nu = {len(edge_states)//2})!!!")
    for i in edge_state_energies: print(f"E = {i[0]:.2f} eV, index = {i[1]}")
    


#%% Calculating the Overlap Integral scaling with angle gamma

def x2y2_func(theta, gamma, b, r):
    return n.cos(2*theta) * (r**2 * n.cos(2*theta) + b**2 * n.cos(2*gamma) - 2*b*r*n.cos(theta+gamma))/(r**2 + b**2 - 2*r*b*n.cos(theta-gamma))

def xy_func(theta, gamma, b, r):
   return 0.25*n.sin(2*theta) * (r**2 * n.sin(2*theta) + b**2 * n.sin(2*gamma) - 4*b*r*n.sin(theta+gamma))/(r**2 + b**2 - 2*r*b*n.cos(theta-gamma))

def x2y2_xy_func(theta, gamma, b, r):
    return 0.5*n.cos(2*theta) * (r**2 * n.sin(2*theta) + b**2 * n.sin(2*gamma) - 2*b*r*n.sin(theta+gamma))/(r**2 + b**2 - 2*r*b*n.cos(theta-gamma))

def xy_x2y2_func(theta, gamma, b, r):
    return 0.5*n.sin(2*theta) * (r**2 * n.cos(2*theta) + b**2 * n.cos(2*gamma) - 2*b*r*n.cos(theta+gamma))/(r**2 + b**2 - 2*r*b*n.cos(theta-gamma))


r = 1
b = 2
theta_array = n.linspace(0,2*n.pi,100)
gamma_array = n.linspace(0,n.pi/2,100)


x2y2_gamma_array = [integ.trapezoid([x2y2_func(i,j,b,r) for i in theta_array], dx=n.abs(theta_array[1]-theta_array[0])) for j in gamma_array]
xy_gamma_array = [integ.trapezoid([xy_func(i,j,b,r) for i in theta_array], dx=n.abs(theta_array[1]-theta_array[0])) for j in gamma_array]
x2y2_xy_gamma_array = [integ.trapezoid([x2y2_xy_func(i,j,b,r) for i in theta_array], dx=n.abs(theta_array[1]-theta_array[0])) for j in gamma_array]
xy_x2y2_gamma_array = [integ.trapezoid([xy_x2y2_func(i,j,b,r) for i in theta_array], dx=n.abs(theta_array[1]-theta_array[0])) for j in gamma_array]

'''
#making the functions have the right 0s and right scaling
xy_gamma_array = (xy_gamma_array - xy_gamma_array[0])
xy_gamma_array = xy_gamma_array/max(xy_gamma_array)
x2y2_gamma_array = x2y2_gamma_array - min(x2y2_gamma_array)
x2y2_gamma_array = x2y2_gamma_array/max(x2y2_gamma_array)

xy_x2y2_gamma_array = xy_x2y2_gamma_array/n.abs(max(xy_x2y2_gamma_array))
x2y2_xy_gamma_array = x2y2_xy_gamma_array/n.abs(max(x2y2_xy_gamma_array))
'''

#Plotting 
x_tick_array = [0,90/4,90/2]
x_tick_labels = ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]
#plt.plot(gamma_array*180/n.pi, x2y2_gamma_array, linewidth = 5, label = r"$x^2-y^2 \leftrightarrow x^2-y^2 \; (cos(2\gamma))$")
plt.plot(gamma_array*180/n.pi, xy_gamma_array, linewidth = 5, label = r"$xy \leftrightarrow xy \; (sin(2\gamma))$")
#plt.plot(gamma_array*180/n.pi, x2y2_xy_gamma_array, linewidth = 5, label = r"$x^2-y^2 \leftrightarrow xy \; (sin(4\gamma))$")
#plt.plot(gamma_array*180/n.pi, xy_x2y2_gamma_array, linewidth = 5, linestyle = "dotted", zorder = 2,  label = r"$xy \leftrightarrow x^2-y^2 \; (sin(4\gamma))$")
plt.xlabel(r"$\gamma$", fontsize=20)
plt.ylabel("Overlap Integral", fontsize=20)
plt.legend()
plt.xlim(0,n.pi)
plt.xticks(x_tick_array, labels=x_tick_labels, fontsize = 20)
plt.yticks(fontsize = 20)
plt.grid()
plt.show()
plt.close()





#%% 6 dimensional model (including Ti dz^2 states combined in a ZR singlet-likie state) with angle gamma

#chosen values of hopping
t_mn=0.5
t_mixing=0
t_ti = 0
delta = 0
#gamma = 0.74 * n.pi/180
gamma = 0

'''
t_mn=0.01
t_mixing=0.83
t_ti = 0.35
delta = 1.2
'''
#initializing the hamiltonian and filling it for all k-points
k_array = n.linspace(-(n.pi),n.pi,800)
H = n.zeros((6,6),dtype=n.cdouble)

#eigenvalues and eigenvectors arrya initizliation (evec indexing: [k][orbital][band])
evals = n.zeros((len(k_array),6), dtype=n.cdouble)
evecs = n.zeros( (len(k_array),6,6), dtype=n.cdouble)

H_storage = n.zeros((len(k_array),6,6),dtype=n.cdouble)


for i in range(len(k_array)):
    
    k = k_array[i]
    
    #Mn xy
    H[0,0] = -delta/3
    H[1,1] = -delta/3
    H[0,1] = -t_mn*(1+cmath.exp(-1j*k)) 
    H[1,0] = -t_mn*(1+cmath.exp(1j*k))

    #Mn x^2-y^2
    H[2,2] = -delta/3
    H[3,3] = -delta/3
    H[2,3] = -t_mn*(1+cmath.exp(-1j*k))
    H[3,2] = -t_mn*(1+cmath.exp(1j*k))
    
    #Ti
    H[4,4] = 2*delta/3
    H[5,5] = 2*delta/3
    H[4,5] = -t_ti * (1+cmath.exp(-1j*k)) * (n.sin(2*gamma) + n.cos(2*gamma))
    H[5,4] = -t_ti * (1+cmath.exp(1j*k)) * (n.sin(2*gamma) + n.cos(2*gamma))
    
    #Mn xy <-> Ti
    H[0,4] = -t_mixing
    H[1,4] = -t_mixing
    H[0,5] = t_mixing * cmath.exp(-1j*k) * n.sin(2*gamma)
    H[1,5] = t_mixing * n.sin(2*gamma)
    
    H[4,0] = -t_mixing
    H[4,1] = -t_mixing
    H[5,0] = t_mixing * cmath.exp(1j*k) * n.sin(2*gamma)
    H[5,1] = t_mixing * n.sin(2*gamma)
    
    #Mn x^2-y^2 <-> Ti
    H[2,4] = 0
    H[3,4] = 0
    H[2,5] = -t_mixing
    H[3,5] = -t_mixing * n.cos(2*gamma)
    
    H[4,2] = 0
    H[4,3] = 0
    H[5,2] = -t_mixing
    H[5,3] = -t_mixing * n.cos(2*gamma)
    
    '''
    #Uncomment to have assymetry between xy and x2-y2 due to the unit cell choice (x2-y2 has out of cell hopping)
    H[2,5] = -t_mixing * cmath.exp(-1j*k) * n.cos(2*gamma)
    H[5,2] = -t_mixing * cmath.exp(1j*k) * n.cos(2*gamma)
    '''
    ####
    
    H_original = H.copy()
    H_storage[i] = H_original
    
    #Chaning the Hamiltonian basis to even/odd Mn states (Ti basis stantes unchanged)

    T_matrix = n.array([[1/n.sqrt(2),1/n.sqrt(2),0,0,0,0], [1/n.sqrt(2),-1/n.sqrt(2),0,0,0,0], [0,0,1/n.sqrt(2),1/n.sqrt(2),0,0], [0,0,1/n.sqrt(2),-1/n.sqrt(2),0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    #T_matrix = n.array([[1/n.sqrt(2),(n.cos(k/2) + 1j*n.sin(k/2))/n.sqrt(2),0,0,0,0], [1/n.sqrt(2),-(n.cos(k/2) + 1j*n.sin(k/2))/n.sqrt(2),0,0,0,0], [0,0,1/n.sqrt(2),(n.cos(k/2) + 1j*n.sin(k/2))/n.sqrt(2),0,0], [0,0,1/n.sqrt(2),-(n.cos(k/2) + 1j*n.sin(k/2))/n.sqrt(2),0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    H = n.linalg.inv(T_matrix) @ H @ T_matrix
    
    #diagonalizing the Hamiltonian and populating the evecs and evals arrays
    evals_temp,evecs_temp = n.linalg.eigh(H)
    evals_temp=evals_temp.real

    
    evals[i] = evals_temp
    evecs[i] = evecs_temp

evecs_original = evecs.copy()


H = n.linalg.inv(T_matrix)@H@T_matrix

sympoints = [k_array[0], k_array[len(k_array)//2], k_array[-1]]

#Plotting the energy bands
multiplier_res = 3
matplotlib.pyplot.figure(figsize=(1.6*multiplier_res, 1*multiplier_res), dpi=300)
chosen_bands = [0,1,2,3,4,5]

band_index_text_positions_array = [0 for i in range(len(H))]
for i in chosen_bands: 
    plt.plot(k_array, evals[:,i].real, linewidth = 2)
    
    #Adding band indices to the plot
    mid_point_value = round(evals[3*len(k_array)//4,i].real,3)
    number_of_degeneracies = band_index_text_positions_array.count(mid_point_value)
    plt.text(k_array[3*len(k_array)//4] + 0.2*number_of_degeneracies, mid_point_value + 0.1, "%d"%i)  
    band_index_text_positions_array[i] = mid_point_value
        
#plt.annotate("t' = %.2ft" % t_mixing, xy=(-2.7,0), fontsize = 20)
#plt.legend(loc="upper right")
plt.ylabel(r"E (eV)", fontsize=17)
plt.xticks(n.sort(sympoints),kpath)
plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelsize = 17)
plt.tick_params(axis = 'y', labelsize = 10)
plt.xlim(0,max(k_array))
plt.grid()
plt.show()
plt.close()


#### Evecs Probability Amplitude
#evec indexing: [k][orbital][band_index]
#orbital ordering: [0 - xy Mn 1] [1 - xy Mn 2] [2 - x2y2 Mn 1] [3 - x2y2 Mn 2]  [4 - xy Ti (in cell)] [5 - x2y2 Ti (between cells)] 


#Uncomment to plot original, unmodified evecs
#evecs = evecs_original.copy()            

chosen_bands = [0,1,2,3,4,5]

for band_number in chosen_bands:
    
    #Creating one big plot with 6 subplots (for each orbital)
    multiplier_res = 3
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
    axes = axes.flatten(order="F")
    #legend_labels = [r"$d_{xy} \; (Mn \; 1, real)$", r"$d_{xy} \; (Mn \; 1, imag)$", r"$d_{xy} \; (Mn \; 2, real)$", r"$d_{xy} \; (Mn \; 2, imag)$", r"$d_{x^2-y^2} \; (Mn \; 1, real)$", r"$d_{x^2-y^2} \; (Mn \; 1, imag)$", r"$d_{x^2-y^2} \; (Mn \; 2, real)$", r"$d_{x^2-y^2} \; (Mn \; 2, imag)$", r"$Ti_{xy} \; (Ti \; 1,real)$", r"$Ti_{xy} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,imag)$"]
    legend_labels = [r"$d_{xy} \; (Mn \; Even, real)$", r"$d_{xy} \; (Mn \; Even, imag)$", r"$d_{xy} \; (Mn \; Odd, real)$", r"$d_{xy} \; (Mn \; Odd, imag)$", r"$d_{x^2-y^2} \; (Mn \; Even, real)$", r"$d_{x^2-y^2} \; (Mn \; Even, imag)$", r"$d_{x^2-y^2} \; (Mn \; Odd, real)$", r"$d_{x^2-y^2} \; (Mn \; Odd, imag)$", r"$Ti_{xy} \; (Ti \; 1,real)$", r"$Ti_{xy} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,imag)$"]

    
    #Adding the band index in the plot
    axes[2].text(0,-1.3, "Band Index: %d" % band_number, fontsize = 8, horizontalalignment="center")
    
    for plot_index in range(len(axes)):
        
        axes[plot_index].grid()
        
        markersize = 5
        axes[plot_index].set_xlim([-n.pi,n.pi])
        axes[plot_index].set_ylim([-1,1])
        
        
        #### All orbtials separated in their own subplots
        
        axes[plot_index].scatter(k_array, evecs[:,plot_index,band_number].real, linewidth = 0.1, s = markersize, linestyle = "solid", marker = "o", color = "red", label = legend_labels[2*plot_index], zorder = 2)
        axes[plot_index].scatter(k_array, evecs[:,plot_index,band_number].imag, linewidth = 0.1, s = markersize, linestyle = "dashed", marker = "^", color = "green", label = legend_labels[2*plot_index+1], zorder = 2)
        
        ###
        
        
        
        '''
        ####Adding up some evecs to analyze mixing of the degenerate states
        for other_band_number in [0,1]:
            if other_band_number == 0:
                axes[plot_index].scatter(k_array, evecs[:,plot_index,band_number + other_band_number].real, linewidth = 0.1, s = markersize, linestyle = "solid", marker = "o", color = "red", label = legend_labels[2*plot_index], zorder = 3)
                axes[plot_index].scatter(k_array, evecs[:,plot_index,band_number + other_band_number].imag, linewidth = 0.1, s = markersize, linestyle = "dashed", marker = "^", color = "green", label = legend_labels[2*plot_index+1], zorder = 3)
            else:
                axes[plot_index].scatter(k_array, evecs[:,plot_index,band_number + other_band_number].real, linewidth = 0.1, s = markersize, linestyle = "solid", marker = "o", color = "blue", zorder = 1)
                axes[plot_index].scatter(k_array, evecs[:,plot_index,band_number + other_band_number].imag, linewidth = 0.1, s = markersize, linestyle = "dashed", marker = "^", color = "blue", zorder = 1)
        
        '''
        
        
            
        
        axes[plot_index].legend(fontsize = 5)
        
        #Removing ticks when unecessary
        if plot_index > 1:
            axes[plot_index].set_yticklabels([])
        else:
            axes[plot_index].set_yticks([-1,0,1])
        
        if plot_index%2 == 0:
            #axes[plot_index].set_xticks([0,n.pi])
            axes[plot_index].set_xticklabels([])
        else:
            axes[plot_index].set_xticks([-n.pi,0,n.pi])
            axes[plot_index].set_xticklabels(["Z", r"$\Gamma$", "Z"])
        


#### Evecs Probability Squared

#chosen_bands = [0,1,2,3,4,5]
chosen_bands = [0,2,4]

for band_number in chosen_bands:
    
    #Creating one big plot with 6 subplots (for each orbital)
    multiplier_res = 3
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
    axes = axes.flatten(order="F")
    #legend_labels = [r"$d_{xy} \; (Mn \; 1, real)$", r"$d_{xy} \; (Mn \; 1, imag)$", r"$d_{xy} \; (Mn \; 2, real)$", r"$d_{xy} \; (Mn \; 2, imag)$", r"$d_{x^2-y^2} \; (Mn \; 1, real)$", r"$d_{x^2-y^2} \; (Mn \; 1, imag)$", r"$d_{x^2-y^2} \; (Mn \; 2, real)$", r"$d_{x^2-y^2} \; (Mn \; 2, imag)$", r"$Ti_{xy} \; (Ti \; 1,real)$", r"$Ti_{xy} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,imag)$"]
    legend_labels = [r"$d_{xy} \; (Mn \; Even, real)$", r"$d_{xy} \; (Mn \; Even, imag)$", r"$d_{xy} \; (Mn \; Odd, real)$", r"$d_{xy} \; (Mn \; Odd, imag)$", r"$d_{x^2-y^2} \; (Mn \; Even, real)$", r"$d_{x^2-y^2} \; (Mn \; Even, imag)$", r"$d_{x^2-y^2} \; (Mn \; Odd, real)$", r"$d_{x^2-y^2} \; (Mn \; Odd, imag)$", r"$Ti_{xy} \; (Ti \; 1,real)$", r"$Ti_{xy} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,real)$", r"$Ti_{x^2-y^2} \; (Ti \; 2,imag)$"]

    
    #Adding the band index in the plot
    axes[2].text(0,-1.3, "Band Index: %d" % band_number, fontsize = 8, horizontalalignment="center")
    
    for plot_index in range(len(axes)):
        
        axes[plot_index].grid()
        
        markersize = 5
        axes[plot_index].set_xlim([-n.pi,n.pi])
        axes[plot_index].set_ylim([-1,1])
        
    
     
        
        ####All orbtials separated in their own subplots
        
        axes[plot_index].scatter(k_array, abs(evecs[:,plot_index,band_number])**2, linewidth = 0.1, s = markersize, linestyle = "solid", marker = "o", color = "red", label = legend_labels[2*plot_index][0:len(legend_labels[2*plot_index])-8] + ")" + legend_labels[2*plot_index][-1], zorder = 2 )
        
        ###
        
        
        '''
        ####Adding up some evecs to analyze mixing of the degenerate states
        
        summed_band_numbers = [0,1]
        summed_degen_evec = (evecs[:,plot_index,band_number] + evecs[:,plot_index,band_number + 1])/n.sqrt(2)
        #axes[plot_index].scatter(k_array, abs(summed_degen_evec)**2, linewidth = 0.1, s = markersize, linestyle = "solid", marker = "o", color = "red", label = legend_labels[2*plot_index], zorder = 3)
        axes[plot_index].scatter(k_array, abs(evecs[:,plot_index,band_number])**2 + abs(evecs[:,plot_index,band_number+1])**2 , linewidth = 0.1, s = markersize, linestyle = "solid", marker = "o", color = "red", label = legend_labels[2*plot_index], zorder = 3)

        ###
        '''
        
        axes[plot_index].legend(fontsize = 5)
        
        #Removing ticks when unecessary
        if plot_index > 1:
            axes[plot_index].set_yticklabels([])
        else:
            axes[plot_index].set_yticks([-1,0,1])
        
        if plot_index%2 == 0:
            #axes[plot_index].set_xticks([0,n.pi])
            axes[plot_index].set_xticklabels([])
        else:
            axes[plot_index].set_xticks([-n.pi,0,n.pi])
            axes[plot_index].set_xticklabels(["Z", r"$\Gamma$", "Z"])















#%% Symbolic Checking


#Declaring my variables and the Hamiltonian, and getting the evals
k = sym.Symbol('k')
delta = sym.Symbol('Delta')
t_mn = sym.Symbol('t')
t_mix = sym.Symbol('t\'')
t_ti = sym.Symbol('t\'\'')

a = sym.Symbol('a')
b = sym.Symbol('b')
c = sym.Symbol('c')
d = sym.Symbol('d')

#H = sym.Matrix([ [0,2,0,0,0,0], [2,0,0,0,0,0], [0,0,0,2,0,0], [0,0,2,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]  ])
H = sym.Matrix([ [0,a,0,0], [b,0,0,0], [0,0,0,c], [0,0,d,0]  ])

T_matrix = sym.Matrix([[1/n.sqrt(2),1/n.sqrt(2),0,0], [1/n.sqrt(2),-1/n.sqrt(2),0,0], [0,0,1/n.sqrt(2),1/n.sqrt(2)], [0,0,1/n.sqrt(2),-1/n.sqrt(2)]])
 
H_T = T_matrix.inv() @ (H @ T_matrix)


#Showing H, the evals and evecs as images
display(H)
display(T_matrix)
display(H_T)










#%% Mixed Bands (k and real space) Plotting from symbolic Hamiltonian


#Declaring my variables and the Hamiltonian, and getting the evals
k = sym.Symbol('k')
delta = sym.Symbol('Delta')
t_mn = sym.Symbol('t')
t_mix = sym.Symbol('t\'')
t_ti = sym.Symbol('t\'\'')

a = sym.Symbol('a')
b = sym.Symbol('b')
c = sym.Symbol('c')
d = sym.Symbol('d')

#H = sym.Matrix([ [0,2,0,0,0,0], [2,0,0,0,0,0], [0,0,0,2,0,0], [0,0,2,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]  ])
H = sym.Matrix([ [0,a,0,0], [b,0,0,0], [0,0,0,c], [0,0,d,0]  ])

T_matrix = sym.Matrix([[1/n.sqrt(2),1/n.sqrt(2),0,0], [1/n.sqrt(2),-1/n.sqrt(2),0,0], [0,0,1/n.sqrt(2),1/n.sqrt(2)], [0,0,1/n.sqrt(2),-1/n.sqrt(2)]])
 
H_T = T_matrix.inv() @ (H @ T_matrix)


#Showing H, the evals and evecs as images
display(H)
display(T_matrix)
display(H_T)




#Creating Figures
k_array = n.linspace(-n.pi,n.pi,1000)
multiplier_res = 7
dpi = 300
fig_bands, ax_bands = plt.subplots(nrows=1, ncols=2, figsize=(1.9*multiplier_res, 1*multiplier_res), sharey=True, dpi=dpi)

#chosen values of hopping
t_mn= 1.3
T_value = 1
phase_factor = cmath.exp(1j * 0)
dxy_evec_plot_index = 0
dx2y2_evec_plot_index = 1




#### K-Space Code
  
# Band indices based on if there is degeneracy or not
if chosen_gamma == 0:
    number_of_bands = 2
    b_eigenvector_1 = diag_H[0][2][0].normalized()
    b_eigenvector_2 = diag_H[0][2][1].normalized()
    ab_eigenvector_1 = diag_H[1][2][0].normalized()
    ab_eigenvector_2 = diag_H[1][2][1].normalized()
    evecs = [[sym.lambdify([k,t,T],b_eigenvector_1[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)], [sym.lambdify([k,t,T],b_eigenvector_2[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)], [sym.lambdify([k,t,T],ab_eigenvector_1[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)], [sym.lambdify([k,t,T],ab_eigenvector_2[orb], "numpy")(k_array,t_mn,T_value) for orb in range(4)]]
    evecs[0][2] = n.zeros(len(k_array))
    evecs[0][3] = n.zeros(len(k_array))
    evecs[1][0] = n.zeros(len(k_array))
    evecs[1][1] = n.zeros(len(k_array))
    evecs[2][2] = n.zeros(len(k_array))
    evecs[2][3] = n.zeros(len(k_array))
    evecs[3][0] = n.zeros(len(k_array))
    evecs[3][1] = n.zeros(len(k_array))
else:
    number_of_bands = 4
    evecs = n.array([ sym.lambdify([k,t,T],diag_H[band_index][2][0].normalized(), "numpy")(k_array,t_mn,T_value)[:] for band_index in range(number_of_bands)])


#Unmixing the bands/evecs if necessary!!! evec indexing: [band][orbital][idk what this is but put 0 lol][k]
bands = n.array([sym.lambdify([k,t,T],diag_H[i][0], "numpy")(k_array,t_mn,T_value).real for i in range(number_of_bands)])
evecs_unmixed = copy.deepcopy(evecs)
bands_unmixed = copy.deepcopy(bands)


#Unmixing the bands and evecs!
for band_index in [0,1]:
    
    previous_diff = 0.3
    for k_idx in range(len(k_array)-1):
        
        if n.abs(bands[band_index][k_idx+1] - bands[band_index][k_idx]) > 2*previous_diff:
            
            # Swapping the bands
            lower_band_segment = n.copy(bands[band_index][k_idx+1:])
            bands[band_index][k_idx+1:] = n.copy(bands[band_index+2][k_idx+1:])
            bands[band_index+2][k_idx+1:] = lower_band_segment

            #swapping the evecs
            for orb in range(4):
                lower_evec_segment = n.copy(evecs[band_index][orb][0][k_idx+1:])
                evecs[band_index][orb][0][k_idx+1:] = n.copy(evecs[band_index+2][orb][0][k_idx+1:])
                evecs[band_index+2][orb][0][k_idx+1:] = lower_evec_segment
            
            #print(f"Swapped bands {band_index} and {band_index + 2}, from k={k_idx+1} onwards")


#### Plotting Bands
for i in range(4):
    ax_bands[0].plot(k_array, bands[i], linewidth = 8)
    ax_bands[0].set_ylabel(r"E (eV)", fontsize=50)
    ax_bands[0].set_xticks([min(k_array), 0, max(k_array)])
    ax_bands[0].set_xlim(min(k_array),max(k_array))
    ax_bands[0].set_xticklabels(kpath)
    ax_bands[0].tick_params(axis="both", labelsize = 40)
ax_bands[0].grid()
    
    
    


#### Plotting Eigenstates

#dxy
xy_label_array= [r"$Re(\alpha_{xy,1})$", r"$Im(\alpha_{xy,1})$",r"$Re(\alpha_{xy,2})$", r"$Im(\alpha_{xy,2})$"]
for ax in [ax_xy, ax_all[1]]:
    for i in range(2):
        
        if number_of_bands == 2:
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][:].real, linewidth = 8, label = xy_label_array[2*i])
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][:].imag, linewidth = 8, label = xy_label_array[2*i+1])
        else:
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][0][:].real, linewidth = 8, label = xy_label_array[2*i])
            ax.plot(k_array, evecs[dxy_evec_plot_index][i][0][:].imag, linewidth = 8, label = xy_label_array[2*i+1])

    ax.set_xticks([min(k_array), 0, max(k_array)])
    ax.set_yticks([-0.7,0,0.7])
    ax.set_xlim(min(k_array),max(k_array))
    ax.set_ylim(-0.8,0.8)
    ax.set_xticklabels(kpath)
    ax.tick_params(axis="both", labelsize = 40)
    ax.legend(loc="lower left", fontsize = 25)
    ax.grid()


#dx^2-y^2
x2y2_label_array= [r"$Re(\alpha_{x^2-y^2,1})$", r"$Im(\alpha_{x^2-y^2,1})$",r"$Re(\alpha_{x^2-y^2,2})$", r"$Im(\alpha_{x^2-y^2,2})$"]
for ax in [ax_x2y2, ax_all[2]]:
    for i in range(2):
        
        if number_of_bands == 2:
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][:].real, linewidth = 8, label = x2y2_label_array[2*i])
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][:].imag, linewidth = 8, label = x2y2_label_array[2*i+1])
        else:
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][0][:].real, linewidth = 8, label = x2y2_label_array[2*i])
            ax.plot(k_array, evecs[dx2y2_evec_plot_index][i+2][0][:].imag, linewidth = 8, label = x2y2_label_array[2*i+1])

    ax.set_xticks([min(k_array), 0, max(k_array)])
    ax.set_yticks([-0.7,0,0.7])
    ax.set_xlim(min(k_array),max(k_array))
    ax.set_ylim(-0.8,0.8)
    ax.set_xticklabels(kpath)
    ax.tick_params(axis="both", labelsize = 40)
    ax.legend(loc="lower left", fontsize = 25)
    ax.grid()


# Winding Number Calculation
chosen_bands = [0,2]
for chosen_band in chosen_bands:
    wind_evec = evecs[chosen_band]
    if number_of_bands == 2:
        wind_evec_diff = [n.gradient(wind_evec[orb], k_array)for orb in range(4)]
        wind_integrand = [ n.sum([1j*n.conjugate(wind_evec[orb][k_idx]) * wind_evec_diff[orb][k_idx] for orb in range(4)]).real for k_idx in range(len(k_array))]
    
    else:
        wind_evec_diff = [n.gradient(wind_evec[orb][0], k_array)for orb in range(4)]
        wind_integrand = [ n.sum([1j*n.conjugate(wind_evec[orb][0][k_idx]) * wind_evec_diff[orb][k_idx] for orb in range(4)]).real for k_idx in range(len(k_array))]
    
        
    winding_number = n.trapz(wind_integrand, k_array)/(n.pi)
    
    print(f"Winding Number Band {chosen_band}: {winding_number:.3f}")
    

#### Real Space Calculations

#number of unit cells
N = 200

#Parameter values
t_value = t_mn
T_value = T_value

gamma = n.pi/8 #maximum mixing
chemical_potential = 0

#Definining the Real Space Hamiltonian Array (d1, d2, D2, D2 basis order)
H = n.zeros((4*N,4*N))


for i in range(0,4*N):
    #chemical potential in diagonal entries
    H[i,i] = -chemical_potential
    
    #xy-xy in-cell hopping
    if 4*i < 4*N-1:
        H[4*i,4*i + 1] = -t_value - T_value
        H[4*i + 1,4*i] = -t_value - T_value
    
    #x^2-y^2 - x^2-y^2 in-cell hopping
    if 4*i + 2 < 4*N-1:  
        H[4*i + 2,4*i + 3] = -t_value + T_value
        H[4*i + 3,4*i + 2] = -t_value + T_value
        
    #xy-xy inter-cell hopping
    if 4*i +4 < 4*N-1:
        H[4*i + 1,4*i + 4] = -t_value + T_value*(n.cos(4*gamma) + n.sin(4*gamma))
        H[4*i + 4,4*i + 1] = -t_value + T_value*(n.cos(4*gamma) + n.sin(4*gamma))
        
    #x^2-y^2-x^2-y^2 inter-cell hopping
    if 4*i +6 < 4*N-1:
        H[4*i + 3,4*i + 6] = -t_value - T_value*(n.cos(4*gamma) - n.sin(4*gamma))
        H[4*i + 6,4*i + 3] = -t_value - T_value*(n.cos(4*gamma) - n.sin(4*gamma))
        
    #xy-x^2-y^2 inter-cell hopping
    if 4*i +3 < 4*N-1:
        H[4*i + 3,4*i + 4] = T_value*n.sin(4*gamma)
        H[4*i + 4,4*i + 3] = T_value*n.sin(4*gamma)
    if 4*i +6 < 4*N-1:
        H[4*i + 1,4*i + 6] = T_value*n.sin(4*gamma)
        H[4*i + 6,4*i + 1] = T_value*n.sin(4*gamma) 
    
    
    
#diagonalizing the madeltax (evec indexing: [orbital][band])
evals,evecs = la.eigh(H)
evals=evals.real


#finding the zero energy states, and plotting the associated eigenstates
zero_energy_indices = [idx for idx,i in enumerate(evals) if n.abs(i)<0.01]

#Finding the edge states
edge_tol = 0.3
edge_states = [i for i in range(len(evecs[:,0])) if edge_tol < n.sum([n.abs(evecs[orb][i])**2 for orb in range(8)]) or edge_tol < n.sum([n.abs(evecs[-(orb+1)][i])**2 for orb in range(8)])  ]
edge_state_energies = sorted( (evals[i], i) for i in edge_states)

#Plotting Finite Chain energy eigenvalues
for idx, i in enumerate(evals):
    if idx in edge_states:
        ax_bands[1].axhline(i, color = "red", zorder = 2, linewidth=4)
    else:
        ax_bands[1].axhline(i, zorder = 1)
    #arranging ticks
ax_bands[1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    


#Chossing which figs to plot
fig_bands.tight_layout()
#plt.close(fig_bands)
#plt.close(fig_evecs)

