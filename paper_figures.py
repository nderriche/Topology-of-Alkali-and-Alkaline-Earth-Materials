import numpy as n
import cmath as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib as mpl

#Hamiltonian parameters (order: Li, Fr) indexing: [element, [delta, tss, tpp, tsp]]
element_string_array = ["Be", "Ra"]
params = n.zeros((2,4))
params[0] = [3.73, 2.25, 2.23, 2.35]
params[1] = [1.94, 0.37, 0.07, 0.32]


#%% Figure 1): 1D Chain Results

#Initial Parameters  (For nice DOS use N=2000 and width=0.05)
N = 500
dos_energy_width = 0.05
dos_edge_state_multiplier = 10
kdens = 201
kpoints = n.linspace(-n.pi, n.pi, kdens)
kpath = ("Z", r"$\Gamma$", "Z")
a_array = [2.15, 5.11]
dos_xlim_array = [1, 6]
dos_ticks_array = [[0.0, 0.5, 1.0], [0, 2, 4, 6]]
dos_ticks_labels = [["0.00", "0.01", "0.02"], ["0.0", "0.1", "0.2"]]


#Creating the Figure
multiplier_res = 11
dpi = 300
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(1.7*multiplier_res, 1*multiplier_res), dpi=dpi)
fig.supylabel("E (eV)", x = 0.02, fontsize = 45)
ax[1,1].set_xlabel("DOS $[eV \\cdot a]^{-1}$", fontsize = 45)


#Filling the Hamiltonian and diagonalizing for all k-points
for element_idx, element in enumerate(element_string_array[0:2]):
    
    #Hamiltonian energy parameters
    delta = params[element_idx][0]
    tss = params[element_idx][1] * 20
    tpp = params[element_idx][2] * 100
    tsp = params[element_idx][3] * 100
    
    
    #### k-space Band Structure Plotting ### ~~

    #Eigenvalues and Eigenvectors array initialization (eval indexing: [k][band] evec indexing: [k][orbital][band])
    H = n.zeros((2,2),dtype=n.cdouble)
    H_no_sp = n.zeros((2,2),dtype=n.cdouble)
    evals = n.zeros((kdens,2), dtype=n.double)
    evecs = n.zeros((kdens,2,2), dtype=n.cdouble)
    evals_no_sp = n.zeros((kdens,2), dtype=n.double)
    evecs_no_sp = n.zeros((kdens,2,2), dtype=n.cdouble)

    
    for i,k in enumerate(kpoints):
        
        #ss part
        H[0,0] = -2*tss*n.cos(k) - delta/2 
        H_no_sp[0,0] = -2*tss*n.cos(k) - delta/2
        
        #p-p part
        H[1,1] = 2*tpp*n.cos(k) + delta/2
        H_no_sp[1,1] = 2*tpp*n.cos(k) + delta/2
        
        #s-p part
        H[0,1] = -2*tsp*1j*n.sin(k)
        H[1,0] = 2*tsp*1j*n.sin(k)
        
        #Diagonalizing the Hamiltonian and populating the evecs and evals arrays
        evals[i], evecs[i] = n.linalg.eigh(H)
        evals_no_sp[i], evecs_no_sp[i] = n.linalg.eigh(H_no_sp)

    #Arranging the B, AB and unmixed band (if they cross) arrays
    crossing_points = n.where(n.abs(evals_no_sp[:,0] - evals_no_sp[:,1])<0.05)[0]
    b_band = evals[:,0]
    ab_band = evals[:,1]
    pure_s_band = evals_no_sp[:, 0]
    pure_p_band = evals_no_sp[:, 1]
    if len(crossing_points) == 2:
        pure_s_band = n.concatenate((evals_no_sp[0:crossing_points[0], 1], evals_no_sp[crossing_points[0]:crossing_points[1],0], evals_no_sp[crossing_points[1]:,1]))
        pure_p_band = n.concatenate((evals_no_sp[0:crossing_points[0], 0], evals_no_sp[crossing_points[0]:crossing_points[1],1], evals_no_sp[crossing_points[1]:,0]))
    
    #Fermi energy determination
    #fermi_energy = (max(b_band) + min(b_band))/2 #Real half-filling Alkali Fermi Energy
    fermi_energy = (max(evals[:,0]) + min(evals[:,1]))/2  #Gap Center  Alkaline Earth Fermi Energy 
    
    #Plotting the B and AB bands
    ax[element_idx,0].plot(kpoints, b_band - fermi_energy, linewidth = 10, color = "C0", label="Bonding")
    ax[element_idx,0].plot(kpoints, ab_band - fermi_energy, linewidth = 10, color = "C1", label="Antibonding")
    ax[element_idx,0].plot(kpoints, pure_s_band - fermi_energy, linewidth = 8, linestyle = "dashed", color = "black", label="Pure $s$")
    ax[element_idx,0].plot(kpoints, pure_p_band - fermi_energy, linewidth = 8, linestyle = "solid", color = "black", label="Pure $p_z$")
    
    #Adding the element name to the plot
    ax[element_idx,0].text(0.87,0.1, "%s" % (element), fontsize = 40, transform = ax[element_idx,0].transAxes, bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    #Plot Parameters
    ax[element_idx,0].set_xlim(-n.pi,n.pi)
    ax[element_idx,0].set_xticks([-n.pi, 0.0, n.pi])
    ax[element_idx,0].tick_params(axis = 'both', labelsize = 38)
    ax[element_idx,0].set_xticklabels(kpath) 
    ax[element_idx,0].grid()
    legend = ax[element_idx,0].legend(fontsize = 22, facecolor = "white", framealpha = 0.5)
    for line in legend.get_lines():
        line.set_linewidth(5)




    #### DOS Real-Space Plotting  ### ~~
    block_left_hopping = n.array([[-tss,-tsp],[tsp,tpp]])
    block_onsite = n.array([[-delta/2,0],[0,delta/2]])
    block_right_hopping = n.array([[-tss,tsp],[-tsp,tpp]])
    H_real = n.zeros((2*N,2*N))
    
    for i in range(0,N):
         H_real[2*i:2*(i+1),2*i:2*(i+1)] = block_onsite
         if i<N-1:
             H_real[2*i:2*(i+1),2*(i+1):2*(i+2)] = block_right_hopping
             H_real[2*(i+1):2*(i+2),2*i:2*(i+1)] = block_left_hopping
    
    #Diagonalizing the Hamiltonian
    evals_real,evecs_real = n.linalg.eigh(H_real)
    evals_real = n.real(evals_real) - fermi_energy
    
    
    #Finding the edge states
    edge_tolerance = 0.2
    edge_states = [i for i in range(len(evals_real)) if edge_tolerance < n.abs(evecs_real[0][i]) or edge_tolerance < n.abs(evecs_real[1][i]) or edge_tolerance < n.abs(evecs_real[-1][i]) or edge_tolerance < n.abs(evecs_real[-2][i]) ]
    edge_states.reverse()
    
    #Determining the Energy gap
    energy_gap = evals_real[len(evals_real)//2+1] - evals_real[len(evals_real)//2-2]
    
    
    #DOS Plotting
    dos_array = []
    dos_energy_array = []
    dos_energy_window_min = min(evals_real)
    while dos_energy_window_min <= max(evals_real):
        dos_array += [ len([i for i in evals_real if dos_energy_window_min <= i < dos_energy_window_min + dos_energy_width])/(N*dos_energy_width)  ]
        dos_energy_array += [dos_energy_window_min + dos_energy_width/2]
        dos_energy_window_min += dos_energy_width
    gap_dos_idx = [idx for (idx,dos) in enumerate(dos_array) if dos == 0]
    
    ax[element_idx,1].tick_params(axis = 'both', labelsize = 38)
    ax[element_idx,1].set_xlim(0, dos_xlim_array[element_idx])
    ax[element_idx,1].fill_betweenx(dos_energy_array[0:gap_dos_idx[0]], dos_array[0:gap_dos_idx[0]], color='C0')
    ax[element_idx,1].fill_betweenx(dos_energy_array[gap_dos_idx[-1]:-1], dos_array[gap_dos_idx[-1]:-1], color='C1')
    ax[element_idx,1].set_xticks(dos_ticks_array[element_idx])
    #ax[element_idx,1].set_xticklabels(dos_ticks_labels[element_idx]) 
    ax[element_idx,1].grid()
    
    #Removing duplicate x-axis and y-axis
    ax[element_idx,1].set_yticklabels([])
    
    #Adding the element name to the plot
    ax[element_idx,1].text(0.87,0.1, "%s" % (element), fontsize = 40, transform = ax[element_idx,1].transAxes, bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    #Wavefunction squared inset and edge state DOS
    if len(edge_states) == 2:
        edge_state_dos_idx = [idx_forward for (idx,idx_forward) in zip(gap_dos_idx, gap_dos_idx[1:]) if idx_forward != idx + 1]
        ax[element_idx,1].plot([0, dos_edge_state_multiplier * n.array(dos_array)[edge_state_dos_idx[0]-1]], [dos_energy_array[edge_state_dos_idx[0]-1], dos_energy_array[edge_state_dos_idx[0]-1]], color = "red", linewidth=3)
        ax[element_idx,1].fill_betweenx(dos_energy_array[edge_state_dos_idx[0]-1:edge_state_dos_idx[-1]], dos_array[edge_state_dos_idx[0]-1:edge_state_dos_idx[-1]], color='red')
        ax_evecs = inset_axes(ax[element_idx,1], width = "100%", height = "100%", bbox_to_anchor=(0.55, 0.45, 2 * 0.2, 1.5 * 0.3), bbox_transform=ax[element_idx,1].transAxes, loc = "lower left")
        ax[element_idx,1].add_patch(plt.Rectangle((0, 0), 1, 1, ls="solid", ec=mpl.colors.to_rgba("black",1), fc=mpl.colors.to_rgba("darkgoldenrod", 0.25), linewidth = 4, transform=ax_evecs.transAxes) )
        label_array= ("Left Edge State", "Right Edge State")
        linestyle_array = ("solid", "dashed")
        color_array = ("darkgreen", "red")
        for edge_idx, label, linestyle, color in zip(edge_states,label_array, linestyle_array, color_array):
            ax_evecs.plot(n.linspace(0,N,N), (evecs_real[0::2,edge_idx])**2 + (evecs_real[1::2,edge_idx])**2, label = label, linestyle = linestyle, color = color, linewidth = 5)
        ax_evecs.set_xlabel("$\\leftarrow$Chain Site$\\rightarrow$", fontsize = 20)
        ax_evecs.xaxis.set_label_coords(0.5,0.2)
        ax_evecs.set_xticks([])
        ax_evecs.text(0.09, 0.4, "$|\Psi_{left}|^2$", fontsize = 25, transform = ax_evecs.transAxes, rotation = "vertical", color = color_array[0], weight = "black")
        ax_evecs.text(0.75, 0.38, "$|\Psi_{right}|^2$", fontsize = 25, transform = ax_evecs.transAxes, rotation = "vertical", color = color_array[1], weight = "black")
        ax_evecs.set_yticks([])
        ax_evecs.tick_params(axis = 'both', labelsize = 15)
        ax_evecs.patch.set_facecolor(color = "lightgrey")
        ax_evecs.patch.set_alpha(0.8)
         

#Space betweetn subfigs and aligning y-labels
#fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.30, wspace = 0.30 )
#fig.align_ylabels(ax[:, 1])
fig.tight_layout()




#%% Figure 2) Bands and Eigenstates at different topological phases

#Initial Parameters 
kdens = 201
kpoints = n.linspace(-n.pi, n.pi, kdens)
kpath = ("Z", r"$\Gamma$", "Z")
bands_x_tick_array = [-5,0,5]
phase_label_array = [dict(text_pos = [1.05, 1.17], label = r"$\Delta < 4|t_+|$", color = "lightcoral"), dict(text_pos = [1.05, 1.17], label = r"$\Delta = 4|t_+|$", color = "yellow"), dict(text_pos = [1.05, 1.17], label = r"$\Delta > 4|t_+|$", color = "limegreen")]


#Hamiltonian energy parameters
tss_array = [params[0][1], params[0][1], params[0][1]/2]
tpp_array = [params[0][2], params[0][2], params[0][2]/2]
tsp_array = [params[0][3], params[0][3], params[0][3]/2]
delta_array = [params[0][0], 2*abs(tss_array[1] + tpp_array[1]), 2*abs(tss_array[2] + tpp_array[2]) + 3]

#Creating the Figure
multiplier_res = 10
dpi = 300
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(1*multiplier_res, 1*multiplier_res), dpi=dpi)

#Filling the Hamiltonian and diagonalizing for all k-points
for idx, (delta, tss, tpp, tsp) in enumerate(zip(delta_array, tss_array, tpp_array, tsp_array)):
    
    #Eigenvalues and Eigenvectors array initialization (eval indexing: [k][band] evec indexing: [k][orbital][band])
    H = n.zeros((2,2),dtype=n.cdouble)
    evals = n.zeros((kdens,2), dtype=n.double)
    evecs = n.zeros((kdens,2,2), dtype=n.cdouble)

    for i,k in enumerate(kpoints):
        
        #ss part
        H[0,0] = -2*tss*n.cos(k) - delta/2 
        
        #p-p part
        H[1,1] = 2*tpp*n.cos(k) + delta/2
        
        #s-p part
        H[0,1] = -2*tsp*1j*n.sin(k)
        H[1,0] = 2*tsp*1j*n.sin(k)
        
        #Diagonalizing the Hamiltonian and populating the evecs and evals arrays
        evals[i], evecs[i] = n.linalg.eigh(H)



    #Fermi energy determination
    #fermi_energy = (max(b_band) + min(b_band))/2 #Real half-filling alklai Fermi Energy
    fermi_energy = (max(evals[:,0]) + min(evals[:,1]))/2  #Gap Center Fermi Energy

    #### Plotting the Bands
    b_band = evals[:,0]
    ab_band = evals[:,1]
    ax[idx,0].plot(kpoints, b_band - fermi_energy, linewidth = 10, label="Bonding")
    ax[idx,0].plot(kpoints, ab_band - fermi_energy, linewidth = 10, label="Anti-Bonding") 
    ax[idx,0].set_xlim(-n.pi,n.pi)
    ax[idx,0].set_xticks([-n.pi, 0.0, n.pi])
    ax[idx,0].set_xticklabels(kpath)
    ax[idx,0].set_yticks(bands_x_tick_array)
    ax[idx,0].set_ylabel("E (eV)", fontsize = 40)
    ax[idx,0].tick_params(axis = 'both', labelsize = 30)
    ax[idx,0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    ax[idx,0].grid()
    legend = ax[idx,0].legend(fontsize = 21, facecolor = "white", framealpha = 0.5)
    for line in legend.get_lines():
        line.set_linewidth(5)
    
    #### Plotting the Bonding Eigenstate
    evec_s = n.abs(evecs[:,0,0].real)
    evec_p = n.concatenate( (n.abs(evecs[0:kdens//2,1,0].imag), -n.abs(evecs[kdens//2:kdens,1,0].imag) ) )
    ax[idx,1].plot(kpoints[1:kdens-1], evec_s[1:kdens-1], linewidth = 10, color = "red", label="Re($\\alpha_s$)")
    ax[idx,1].plot(kpoints[1:kdens-1], evec_p[1:kdens-1], linewidth = 10, color = "green", label="Im($\\alpha_p$)")   
    ax[idx,1].set_xlim(-n.pi,n.pi)
    ax[idx,1].set_xticks([-n.pi, 0.0, n.pi])
    ax[idx,1].set_xticklabels(kpath)
    ax[idx,1].set_ylim(-1.2,1.2)
    ax[idx,1].set_yticks([-1, 0, 1])
    ax[idx,1].set_yticklabels(["-1", "0", "1"])
    ax[idx,1].yaxis.tick_right()
    ax[idx,1].tick_params(axis = 'both', labelsize = 30)
    ax[idx,1].grid()
    legend = ax[idx,1].legend(fontsize = 21, facecolor = "white", framealpha = 0.5)
    for line in legend.get_lines():
        line.set_linewidth(5)
        

    #Adding text about the topological phase and boxes around the data for each phase
    fig.text(*phase_label_array[idx]["text_pos"], phase_label_array[idx]["label"], fontsize = 30, transform = ax[idx,0].transAxes, bbox = dict(boxstyle='round', facecolor=phase_label_array[idx]["color"], alpha=0.5), horizontalalignment = "center")
    
    fig.patches.extend([plt.Rectangle((-0.28, -0.25), 2.5, 1.7, linewidth = 5, ec="black", fc=mpl.colors.to_rgba("black", 0.0),  transform=ax[idx,0].transAxes)])


#Adjusting space between subplots and adding an overal border
fig.subplots_adjust(left=0, bottom=0, right=0.9, top=1, hspace=0.7, wspace = 0.1 )
fig.patch.set_linewidth(10)
fig.patch.set_edgecolor('black')



#%% Figure 4) 2D Hexagonal Results

sym_points = dict(G = dict(pos = [0,0], label = r"$\Gamma$"), K = dict(pos = [4*n.pi/(3*n.sqrt(3)), 0], label = "K"), M = dict(pos = [n.pi/n.sqrt(3), n.pi/3], label = "M"))

#Initial Parameters and Chosen Path
kdens = 100
chosen_path = ("G", "K", "M", "G")
chosen_path_labels = [sym_points[point]["label"] for point in chosen_path]
chosen_path_pos = [sym_points[point]["pos"] for point in chosen_path]
kpath = [chosen_path_pos[0]]
sympoint_idx = [0]
yticks_array = [ [-10, -5, 0], [-10, -5, 0, 5], [-5, 0, 5], [-5, 0, 5], [-2, -1, 0, 1] ]
phase_label_array = [dict(text_pos = [0.5, 0.70], label = r"$\Delta < \frac{3}{2}|3t_+ - t_-|$", color = "lightcoral"), dict(text_pos = [0.5, 0.48], label = r"$\Delta = \frac{3}{2}|3t_+ - t_-|$", color = "yellow"), dict(text_pos = [0.5, 0.48], label = r"$\Delta > \frac{3}{2}|3t_+ - t_-|$", color = "limegreen")]

#Calculating the k-points along the path
for segment_idx, (start_point, end_point) in enumerate(zip(chosen_path_pos, chosen_path_pos[1:])):
    segment_vec = n.subtract(end_point, start_point)
    segment_kpoints = [ n.add(i * segment_vec , start_point) for i in n.linspace(0, 1, int(n.linalg.norm(segment_vec)*kdens), endpoint = True)[1:] ]
    kpath += segment_kpoints
    sympoint_idx += [sympoint_idx[-1] + len(segment_kpoints)]
kpath = n.array(kpath) 
num_kpoints = len(kpath)
plotting_kpath = n.linspace(0, 1, num_kpoints)
plotting_sympoints = [plotting_kpath[idx] for idx in sympoint_idx]

#Hamiltonian energy parameters
hopping_scaling = 2
tss_array = [params[0][1], params[0][1], params[0][1]/2]
tpp_sigma_array = [params[0][2], params[0][2], params[0][2]/2]
tpp_pi_array = [0,0,0]
tsp_array = [params[0][3], params[0][3], params[0][3]/2]
delta_array = [params[0][0], 3*abs(tss_array[1] + tpp_sigma_array[1]/2), 3*abs(tss_array[2] + tpp_sigma_array[2]/2) + 3]

#Creating the Figure
k_subplot_indices = ((0,1), (1,0), (1,1))
multiplier_res = 10
dpi = 300
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(1.122*multiplier_res, 1*multiplier_res), dpi=dpi)


#### k-space Band Structure Plotting ### ~~
for idx, (delta, tss, tpp_sigma, tpp_pi, tsp, subplot_idx) in enumerate(zip(delta_array, tss_array, tpp_sigma_array, tpp_pi_array, tsp_array, k_subplot_indices)):

    #Eigenvalues and Eigenvectors array initialization (eval indexing: [kx][ky][band] evec indexing: [kx][ky][orbital][band])
    H = n.zeros((6,6),dtype=n.cdouble)
    evals = n.zeros((num_kpoints, 6), dtype=n.cdouble)
    evecs = n.zeros((num_kpoints, 6, 6), dtype=n.cdouble)   
    
    #on-site energy part
    H[0,0] = -2*delta/3
    H[1,1] = delta/3
    H[2,2] = delta/3
    H[3,3] = -2*delta/3
    H[4,4] = delta/3
    H[5,5] = delta/3
    
    for k_idx, (kx, ky) in enumerate(kpath):

        #complex exponents
        kgam1 = 1j * (kx * cm.sqrt(3)/2 + ky/2)
        kgam2 = 1j * (-kx * cm.sqrt(3)/2 + ky/2)
        kgam3 = 1j * (-ky)
    
        #ss part
        H[0,3] = -tss*(cm.exp(kgam1) + cm.exp(kgam2) + cm.exp(kgam3))
        H[3,0] = -tss*(cm.exp(-kgam1) + cm.exp(-kgam2) + cm.exp(-kgam3))
        
        #px-px part
        H[1,4] = (3*tpp_sigma/4)*(cm.exp(kgam1) + cm.exp(kgam2)) - (tpp_pi/4)*(cm.exp(kgam1) + cm.exp(kgam2) + 4*cm.exp(kgam3))
        H[4,1] = (3*tpp_sigma/4)*(cm.exp(-kgam1) + cm.exp(-kgam2)) - (tpp_pi/4)*(cm.exp(-kgam1) + cm.exp(-kgam2) + 4*cm.exp(-kgam3))
        
        #py-py part
        H[2,5] = (3*tpp_pi/4)*(cm.exp(kgam1) + cm.exp(kgam2)) + (tpp_sigma/4)*(cm.exp(kgam1) + cm.exp(kgam2) + 4*cm.exp(kgam3))
        H[5,2] = (3*tpp_pi/4)*(cm.exp(-kgam1) + cm.exp(-kgam2)) + (tpp_sigma/4)*(cm.exp(-kgam1) + cm.exp(-kgam2) + 4*cm.exp(-kgam3))
        
        #px-py part
        H[1,5] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(kgam1) - cm.exp(kgam2))
        H[5,1] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(-kgam1) - cm.exp(-kgam2))
        H[2,4] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(kgam1) - cm.exp(kgam2))
        H[4,2] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(-kgam1) - cm.exp(-kgam2))
        
        #s-px part
        H[0,4] = (tsp * (cm.sqrt(3)/2))*(cm.exp(kgam1) - cm.exp(kgam2))
        H[4,0] = (tsp * (cm.sqrt(3)/2))*(cm.exp(-kgam1) - cm.exp(-kgam2))
        H[1,3] = (-tsp * (cm.sqrt(3)/2))*(cm.exp(kgam1) - cm.exp(kgam2))
        H[3,1] = (-tsp * (cm.sqrt(3)/2))*(cm.exp(-kgam1) - cm.exp(-kgam2))
        
        #s-py part
        H[0,5] = (tsp/2)*(cm.exp(kgam1) + cm.exp(kgam2) - 2*cm.exp(kgam3))
        H[5,0] = (tsp/2)*(cm.exp(-kgam1) + cm.exp(-kgam2) - 2*cm.exp(-kgam3))
        H[2,3] = (-tsp/2)*(cm.exp(kgam1) + cm.exp(kgam2) - 2*cm.exp(kgam3))
        H[3,2] = (-tsp/2)*(cm.exp(-kgam1) + cm.exp(-kgam2) - 2*cm.exp(-kgam3))
         
        #Diagonalizing the Hamiltonian and populating the evecs and evals arrays
        evals[k_idx],evecs[k_idx] = n.linalg.eigh(H)


    #Plotting the energy bands  
    #fermi_energy = max( max(evals[:,0]), max(evals[:,1])  ) #Alkali center of s bands Fermi Energy
    fermi_energy = ( max(evals[:,1]) + min(evals[:,2]) )/2#Alakaline Earth between top of s bands and first flat p band
    evals = evals - fermi_energy
    for band_idx in range(len(evals[0])):
        ax[subplot_idx].plot(plotting_kpath, evals[:,band_idx].real, linewidth = 6, color= "C0" if band_idx <=1 else "C1" )
    
    
    
    #Adding the phase text to the plot
    ax[subplot_idx].text(*phase_label_array[idx]["text_pos"], phase_label_array[idx]["label"], fontsize = 20, transform = ax[subplot_idx].transAxes, bbox = dict(boxstyle='round', facecolor=phase_label_array[idx]["color"], alpha=0.8), horizontalalignment = "center")
    
    #Plot Parameters
    ax[subplot_idx].set_xlim(min(plotting_kpath),max(plotting_kpath))
    ax[subplot_idx].set_xticks(plotting_sympoints)
    ax[subplot_idx].set_xticklabels(chosen_path_labels)
    ax[subplot_idx].set_ylabel("E (eV)", fontsize = 35)
    ax[subplot_idx].set_yticks(yticks_array[idx])
    ax[subplot_idx].tick_params(axis = 'both', labelsize = 30)
    ax[subplot_idx].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    ax[subplot_idx].grid()
    

# ~~~



#### DOS Real Space Band Structure Plotting ### ~~

#Initial Parameters and Chosen Path (Nl = 100, kdens_strip = 201 for nice DOS plots)
Nl = 100
kdens_strip = 201
dos_energy_width = 0.05
dos_edge_state_multiplier = 10
dos_ticks_array = [[0.0, 0.5, 1.0], [0, 1, 2]]
y_labelpad_array = [8,-10]
chosen_path_strip_labels = ("K", r"$\Gamma$", "K")
kpath_strip = n.linspace(-n.pi/n.sqrt(3), n.pi/n.sqrt(3), kdens_strip, endpoint=True)
sympoints_strip = [kpath_strip[0], kpath_strip[kdens_strip//2], kpath_strip[-1]]

for element_idx, element in enumerate(element_string_array):
    
    #Hamiltonian energy parameters
    delta = params[element_idx][0]
    tss = params[element_idx][1]
    tpp_sigma = params[element_idx][2]
    tpp_pi = 0
    tsp = params[element_idx][3]
    
    #Eigenvalues and Eigenvectors array initialization (eval indexing: [kx][band] evec indexing: [kx][orbital][band])
    evals = n.zeros((kdens_strip,6*Nl), dtype=n.cdouble)
    evecs = n.zeros((kdens_strip,6*Nl,6*Nl), dtype=n.cdouble)
  
    #Filling the Hamiltonian and diagonalizing for all k-points
    #Eval Indexing is l=0  [k][band]
    for k_idx, kx, in enumerate(kpath_strip):
        H = n.zeros((6*Nl,6*Nl),dtype=n.cdouble)
        
        for l in range(Nl):
            
            #position indices (lm = l-1)
            pos_l = 6*l
            pos_lm = 6*(l-1)
            
            #complex exponents (only first (x) lattice direction this time)
            kgam1 = 1j * (kx * cm.sqrt(3)/2)
            kgam2 = 1j * (-kx * cm.sqrt(3)/2)
            
            #complex exponents
            kgam1 = 1j * (kx * cm.sqrt(3)/2)
            kgam2 = 1j * (-kx * cm.sqrt(3)/2)
    
            
            ###Onsite energy part
            H[pos_l,pos_l] = -2*delta/3
            H[pos_l+1,pos_l+1] = delta/3
            H[pos_l+2,pos_l+2] = delta/3
            H[pos_l+3,pos_l+3] = -2*delta/3
            H[pos_l+4,pos_l+4] = delta/3
            H[pos_l+5,pos_l+5] = delta/3
            

             
            ###s-s part
            H[pos_l, pos_l + 3] += -tss*(cm.exp(kgam1) + cm.exp(kgam2))
            H[pos_l + 3, pos_l] += -tss*(cm.exp(-kgam1) + cm.exp(-kgam2))
            
            if pos_lm >= 0:
                H[pos_l, pos_lm + 3] += -tss
                H[pos_lm + 3, pos_l] += -tss
    
    
            ###px-px part  
            H[pos_l+1, pos_l+4] = (3*tpp_sigma/4)*(cm.exp(kgam1) + cm.exp(kgam2)) - (tpp_pi/4)*(cm.exp(kgam1) + cm.exp(kgam2))
            H[pos_l+4, pos_l+1] = (3*tpp_sigma/4)*(cm.exp(-kgam1) + cm.exp(-kgam2)) - (tpp_pi/4)*(cm.exp(-kgam1) + cm.exp(-kgam2))
            
            if pos_lm >= 0:
                H[pos_l+1,pos_lm+4] = -tpp_pi
                H[pos_lm+4,pos_l+1] = -tpp_pi
                
            
            ###py-py part
            H[pos_l+2, pos_l+5] = (3*tpp_pi/4)*(cm.exp(kgam1) + cm.exp(kgam2)) + (tpp_sigma/4)*(cm.exp(kgam1) + cm.exp(kgam2))
            H[pos_l+5, pos_l+2] = (3*tpp_pi/4)*(cm.exp(-kgam1) + cm.exp(-kgam2)) + (tpp_sigma/4)*(cm.exp(-kgam1) + cm.exp(-kgam2))
    
            if pos_lm >= 0:
                H[pos_l+2,pos_lm+5] = tpp_sigma
                H[pos_lm+5,pos_l+2] = tpp_sigma
                
            
            ###px-py part
            H[pos_l+1, pos_l+5] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(kgam1) - cm.exp(kgam2))
            H[pos_l+5, pos_l+1] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(-kgam1) - cm.exp(-kgam2))
            H[pos_l+2, pos_l+4] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(kgam1) - cm.exp(kgam2))
            H[pos_l+4, pos_l+2] = ( (tpp_sigma + tpp_pi) * (cm.sqrt(3)/4) ) * (cm.exp(-kgam1) - cm.exp(-kgam2))
            
            
            ###s-p part
            H[pos_l, pos_l+4] = (tsp * (cm.sqrt(3)/2))*(cm.exp(kgam1) - cm.exp(kgam2))
            H[pos_l+4, pos_l] = (tsp * (cm.sqrt(3)/2))*(cm.exp(-kgam1) - cm.exp(-kgam2))
            H[pos_l+1, pos_l+3] = (-tsp * (cm.sqrt(3)/2))*(cm.exp(kgam1) - cm.exp(kgam2))
            H[pos_l+3, pos_l+1] = (-tsp * (cm.sqrt(3)/2))*(cm.exp(-kgam1) - cm.exp(-kgam2))
            
            H[pos_l, pos_l+5] = (tsp/2)*(cm.exp(kgam1) + cm.exp(kgam2))
            H[pos_l+5, pos_l] = (tsp/2)*(cm.exp(-kgam1) + cm.exp(-kgam2))
            H[pos_l+2, pos_l+3] = (-tsp/2)*(cm.exp(kgam1) + cm.exp(kgam2))
            H[pos_l+3, pos_l+2] = (-tsp/2)*(cm.exp(-kgam1) + cm.exp(-kgam2))
                
            if pos_lm >= 0:
                H[pos_l, pos_lm+5] = -tsp
                H[pos_lm+5, pos_l] = -tsp
                H[pos_lm+3, pos_l+2] = tsp
                H[pos_l+2, pos_lm+3] = tsp
                  
        
         
    
        #Diagonalizing the Hamiltonian
        evals[k_idx], evecs[k_idx] = n.linalg.eigh(H)

 

    
    #Checking if Element is in Topological Phase based on Parameters
    is_topo = True
    if delta > 3*abs(tss + tpp_sigma/2):
        is_topo = False
        
    #Fermi energy determination (Alkali up to and including s bands, Alkali-Earth up to and including the first denerate flat p band) 
    #fermi_energy = ( evals[len(evals)//2,len(evals[0])//6-1] + evals[len(evals)//2,len(evals[0])//6-]  )/2 #Alakali center of s bands Fermi Energy
    fermi_energy = ( ( max(evals[:,len(evals[0])//3-1]) +  min(evals[:,len(evals[0])//3]) )/2 ).real #Alakaline Earth top of s bands Fermi energy
    
    #Determining edge states while excluding Flat Bands { Evec Indexing: [k][site 0 6 orbitals, site 1 6 orbitals, ...][band] }
    edge_tolerance = 0.5
    edge_states = [ band_index for band_index in [len(evals[0])//2-1,len(evals[0])//2] if any ([edge_tolerance < abs(evecs[0][orb][band_index])   for orb in [0,1,2,3,4,5,-1,-2,-3,-4,-5,-6]])  ]

    
    ### Uncomment to plot DOS
    #DOS Plotting
    evals_dos = evals.real.reshape(-1)
    evals_dos = [i - fermi_energy for i in evals_dos]
    evals_dos.sort()
    dos_array = []
    dos_energy_array = []
    dos_energy_window_min = min(evals_dos)
    while dos_energy_window_min <= max(evals_dos):
        dos_array += [ len([i for i in evals_dos if dos_energy_window_min <= i < dos_energy_window_min + dos_energy_width])/(Nl*kdens_strip*dos_energy_width * (3*n.sqrt(3))/2)  ]
        dos_energy_array += [dos_energy_window_min + dos_energy_width/2]
        dos_energy_window_min += dos_energy_width
    gap_dos_idx = [idx for (idx,dos) in enumerate(dos_array) if dos == 0]
    
    ax[2,element_idx].tick_params(axis = 'both', labelsize = 30)
    ax[2,element_idx].set_xlabel("DOS $[eV \\cdot a^2]^{-1}$", fontsize = 35)
    ax[2,element_idx].set_ylabel("E (eV)", fontsize = 35, labelpad = y_labelpad_array[element_idx])
    ax[2,element_idx].set_xlim(0, dos_ticks_array[element_idx][-1])
    ax[2,element_idx].set_xticks(dos_ticks_array[element_idx])
    ax[2,element_idx].fill_betweenx(dos_energy_array[0:gap_dos_idx[0]], dos_array[0:gap_dos_idx[0]], color='C0')
    ax[2,element_idx].fill_betweenx(dos_energy_array[gap_dos_idx[-1]:-1], dos_array[gap_dos_idx[-1]:-1], color='C1')
    ax[2,element_idx].grid()
    
    #Adding the element name to the plot
    ax[2,element_idx].text(0.82,0.15, "%s" % (element), fontsize = 35, transform = ax[2,element_idx].transAxes, bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Wavefunction squared inset, edge state DOS and lower flat p band DOS if topological
    if len(edge_states) == 2 and element_idx == 0:
        edge_state_dos_idx = [idx_forward for (idx,idx_forward) in zip(gap_dos_idx, gap_dos_idx[1:]) if idx_forward != idx + 1]
        ax[2,element_idx].fill_betweenx(dos_energy_array[gap_dos_idx[0]:gap_dos_idx[-1]], dos_array[gap_dos_idx[0]:gap_dos_idx[-1]], color='red')
        ax[2,element_idx].fill_betweenx(dos_energy_array[gap_dos_idx[0]-2:gap_dos_idx[0]+1], dos_array[gap_dos_idx[0]-2:gap_dos_idx[0]+1], color='C1')

        ax_evecs = inset_axes(ax[2,element_idx], width = "100%", height = "100%", bbox_to_anchor=(0.55, 0.40, 2 * 0.2, 1.5 * 0.3), bbox_transform=ax[2,element_idx].transAxes, loc = "lower left")
        ax[2,element_idx].add_patch(plt.Rectangle((0, 0), 1, 1, ls="solid", ec=mpl.colors.to_rgba("black",1), fc=mpl.colors.to_rgba("darkgoldenrod", 0.0000), linewidth = 4, transform=ax_evecs.transAxes) )
        label_array= ("Left Edge State", "Right Edge State")
        linestyle_array = ("solid", "solid")
        color_array = ("red", "red")
        for edge_idx, label, linestyle, color in zip(edge_states,label_array, linestyle_array, color_array):
            
            #Average real-space density of the edge states over k
            edge_state_density = ( (evecs[0][0::6,edge_idx])**2 + (evecs[0][1::6,edge_idx])**2 + (evecs[0][2::6,edge_idx])**2  + (evecs[0][3::6,edge_idx])**2 + (evecs[0][4::6,edge_idx])**2 + (evecs[0][5::6,edge_idx])**2 )/kdens_strip
            for k in range(1,kdens_strip):
                edge_state_density = edge_state_density + ( (evecs[k][0::6,edge_idx])**2 + (evecs[k][1::6,edge_idx])**2 + (evecs[k][2::6,edge_idx])**2  + (evecs[k][3::6,edge_idx])**2 + (evecs[k][4::6,edge_idx])**2 + (evecs[k][5::6,edge_idx])**2 )/kdens_strip 
            
            ax_evecs.plot(n.linspace(0,Nl,Nl), edge_state_density, label = label, linestyle = linestyle, color = color, linewidth = 3)
        
        ax_evecs.set_xlabel("$\\vec{e}_2$$\\rightarrow$", fontsize = 20)
        ax_evecs.xaxis.set_label_coords(0.3,0.32)
        ax_evecs.set_xticks([])
        ax_evecs.text(0.25, 0.6, "$|\Psi_{Edge}|^2$", fontsize = 20, transform = ax_evecs.transAxes, color = color_array[0], weight = "black")
        ax_evecs.set_yticks([])
        ax_evecs.tick_params(axis = 'both', labelsize = 15)
        ax_evecs.patch.set_facecolor(color = "lightgrey")
        ax_evecs.patch.set_alpha(0.8)
             
            
    ''' ### Uncomment to plot strip bands
    #Plotting the bands
    for band_index in range(len((evals[0]))):
        if band_index in edge_states and is_topo == True:
            ax[2,element_idx].plot(kpath_strip, evals[:,band_index].real - fermi_energy, color = "red", zorder = 2, linewidth = 5)
        else:
            color = "C0" if band_index <= len(evals[0])//3 else "C1"
            ax[2,element_idx].plot(kpath_strip, evals[:,band_index].real - fermi_energy, color = color, zorder = 1,  linewidth = 4)
        
        
    #Adding the element name to the plot
    ax[2,element_idx].text(0.82,0.15, "%s" % (element), fontsize = 35, transform = ax[2,element_idx].transAxes, bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    #Plot Parameters
    ax[2,element_idx].tick_params(axis = 'y', labelsize = 8)
    ax[2,element_idx].set_xticklabels([])
    ax[2,element_idx].set_xticks([])
    ax[2,element_idx].set_xlim(min(kpath_strip),max(kpath_strip))
    ax[2,element_idx].set_xticks(sympoints_strip)
    ax[2,element_idx].set_xticklabels(chosen_path_strip_labels)
    ax[2,element_idx].set_yticks(yticks_array[element_idx + 3])
    ax[2,element_idx].set_ylabel("E (eV)", fontsize = 35)
    ax[2,element_idx].tick_params(axis = 'both', labelsize = 30)
    ax[2,element_idx].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    ax[2,element_idx].grid()
    '''
    
#Space between subfigs and Deleting first subplot
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.30, wspace = 0.30 )
fig.delaxes(ax[0,0])




