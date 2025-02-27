import numpy as n
import scipy.linalg as la
import matplotlib.pyplot as plt
import cmath
import matplotlib as mpl
import sympy as sym
from scipy import optimize
import cmath as cm
kpath = ("Z", r"$\Gamma$", "Z")
kpath_2D = (r"$\Gamma$", "K", "M", r"$\Gamma$")

k_dens = 200
kpoints = n.linspace(0,1,k_dens)
cm_to_meV = 0.124
THz_to_meV = 4.136



#%% Creating Bonding and Antibonding Bands

#Binding band function for fitting

def eval_B(x,charge_transfer,tss,tpp,tsp):
    return charge_transfer/2 + n.cos((x-0.5)*2*n.pi)*(tpp-tss) - 0.5*n.sqrt(4 * ((n.cos((x-0.5)*2*n.pi))**2) * (tpp**2 + tss**2 + 2*tpp*tss - 4*(tsp**2)) + 4*charge_transfer*n.cos((x-0.5)*2*n.pi)*(tss+tpp) + charge_transfer**2 + 16*(tsp**2) )

def eval_AB(x,charge_transfer,tss,tpp,tsp):
    return charge_transfer/2 + n.cos((x-0.5)*2*n.pi)*(tpp-tss) + 0.5*n.sqrt(4 * ((n.cos((x-0.5)*2*n.pi))**2) * (tpp**2 + tss**2 + 2*tpp*tss - 4*(tsp**2)) + 4*charge_transfer*n.cos((x-0.5)*2*n.pi)*(tss+tpp) + charge_transfer**2 + 16*(tsp**2) )

def eval_B_fermi(x,charge_transfer,tss,tpp,tsp, fermi):
    return charge_transfer/2 + n.cos((x-0.5)*2*n.pi)*(tpp-tss) - 0.5*n.sqrt(4 * ((n.cos((x-0.5)*2*n.pi))**2) * (tpp**2 + tss**2 + 2*tpp*tss - 4*(tsp**2)) + 4*charge_transfer*n.cos((x-0.5)*2*n.pi)*(tss+tpp) + charge_transfer**2 + 16*(tsp**2) ) - fermi

def eval_AB_fermi(x,charge_transfer,tss,tpp,tsp, fermi):
    return charge_transfer/2 + n.cos((x-0.5)*2*n.pi)*(tpp-tss) + 0.5*n.sqrt(4 * ((n.cos((x-0.5)*2*n.pi))**2) * (tpp**2 + tss**2 + 2*tpp*tss - 4*(tsp**2)) + 4*charge_transfer*n.cos((x-0.5)*2*n.pi)*(tss+tpp) + charge_transfer**2 + 16*(tsp**2) ) - fermi

def combined_evals(x_combined,charge_transfer,tss,tpp,tsp):
    x_b = x_combined[0:k_dens]
    x_ab = x_combined[k_dens:len(x_combined)]
    
    B_result = eval_B(x_b,charge_transfer,tss,tpp,tsp)
    AB_result = eval_AB(x_ab,charge_transfer,tss,tpp,tsp)
    
    #Centering the bands:
    center = (AB_result[0] + B_result[0])/2
    full_results = n.append(B_result, AB_result) - center
    return full_results



element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]

num_band_array = [14, 14,17, 17,19,19, 19,19, 19,14,19,21]
num_band_SO_array = [28, 28, 34, 34, 38, 38, 38, 38, 38, 28, 38, 42]
lat_const_array = [3.041,2.146,3.282,3.073,4.095,3.99,4.339,4.481,4.759,3.897,4.688,5.109]
s_p_band_indices = [[1,4,2], [1,4,2], [4, 8,7], [4, 8,7], [4,15,13],[4,10,13], [4,15,5],[4,15,5], [4,15,5], [4,13,5], [4,15,5],[4,15,5]]
s_p_band_indices_SO = [ [ 1 + 2*(i[0] - 1), 1 + 2*(i[1] - 1), 1 + 2*(i[2] - 1)] for i in s_p_band_indices]
isolated_s_p_band_indices = [ [1,4], [1,4], [4,7], [4,7],  [4,12],  [4,12], [4,12],  [4,12], [4,12], [4,12],  [4,7], [4,12]] 
weights_index_array = [[3,7], [3,7], [6,10], [6,10], [6,10], [6,10], [6,10], [6,10], [6,20], [6,15], [6,20], [6,15]]
weights_index_SO_array = [ [ 1 + 2*(i[0] - 1), 1 + 2*(i[1] - 1)] for i in weights_index_array]

#### Getting the band energy for each k where the s and p weights are maximised (not too far from the fermi band)

#Looping over elements
constructed_bands = n.zeros( (2,len(element_string_array), k_dens ))
constructed_bands_SO = n.zeros( (2,len(element_string_array), k_dens ))
mask_bands = n.full(((2,len(element_string_array), k_dens )), False)
mask_bands_SO = n.full(((2,len(element_string_array), k_dens )), False)
energy_min = -10
energy_max = 10
for i in range(len(element_string_array)):
    
    band_data = n.loadtxt("other_elements//%s//+bweights" % element_string_array[i])
    band_data_SO = n.loadtxt("other_elements//%s//+bweights_so" % element_string_array[i])
    
    
    #Looping through kpoints
    for k_index in range(k_dens):

        #Looping through bands (only in the energy range) to get the maximum s and p weights value
        included_bands = [l for l in range(num_band_array[i]) if energy_min < band_data[l+ k_index*num_band_array[i]][1] < energy_max]
        included_bands_SO = [l for l in range(num_band_SO_array[i]) if energy_min < band_data_SO[l+ k_index*num_band_SO_array[i]][1] < energy_max]
        
        max_weights = [-1,-1]
        max_weights_SO = [-1,-1]
        max_weight_band_indices = [-1,-1]
        max_weight_band_indices_SO = [-1,-1]
        
        for band_index in included_bands:
            current_weights = [band_data[band_index + k_index*num_band_array[i]][weights_index_array[i][0]], band_data[band_index + k_index*num_band_array[i]][weights_index_array[i][1]]]
            for orb in range(2):
                if current_weights[orb] > max_weights[orb]:
                    max_weights[orb] = current_weights[orb]
                    max_weight_band_indices[orb] = band_index
                             
        for band_index_SO in included_bands_SO:
            current_weights_SO = [band_data_SO[band_index_SO + k_index*num_band_SO_array[i]][weights_index_SO_array[i][0]], band_data_SO[band_index_SO + k_index*num_band_SO_array[i]][weights_index_SO_array[i][1]]]
            for orb in range(2):
                if current_weights_SO[orb] > max_weights_SO[orb]:
                    max_weights_SO[orb] = current_weights_SO[orb]
                    max_weight_band_indices_SO[orb] = band_index_SO
        
        
     
        
        #Getting the energy of the band with max s and p weights for that k-point, but masking it if it is from the mainly s band of mainly p band
        for orb in range(2):
            constructed_bands[orb,i,k_index] = band_data[max_weight_band_indices[orb] + k_index*num_band_array[i]][1]
            constructed_bands_SO[orb,i,k_index] = band_data_SO[max_weight_band_indices_SO[orb] + k_index*num_band_SO_array[i]][1]
            if max_weight_band_indices[orb] == s_p_band_indices[i][orb]:
                mask_bands[orb,i,k_index] = True
            if max_weight_band_indices_SO[orb] == s_p_band_indices_SO[i][orb]:
                mask_bands_SO[orb,i,k_index] = True
        




# Constructing the bonding and anti-bonding bands from the data I have
b_bands= n.zeros((len(element_string_array), k_dens))
b_bands_SO = n.zeros((len(element_string_array), k_dens))
ab_bands = n.zeros((len(element_string_array), k_dens))
ab_bands_SO = n.zeros((len(element_string_array), k_dens))
ab_mask_array = n.full(((len(element_string_array)),k_dens), False)
ab_mask_array_SO = n.full(((len(element_string_array)),k_dens), False)

crossed_element_limit = 5
for i in range(len(element_string_array)):
    orb_switch_b = 0
    orb_switch_b_SO = 0
    ab_band_done = False
    ab_band_done_SO = False
    
    #Looping over half the k-range
    for k in range(k_dens//2, k_dens):
        
        #Keeping on the same band for the Bonding Band if no discontinuity
        if abs(constructed_bands[orb_switch_b%2][i][k] - constructed_bands[orb_switch_b%2][i][k-1]) < 0.1*abs(max(constructed_bands[orb_switch_b%2][i]) - min(constructed_bands[orb_switch_b%2][i])):
            b_bands[i][k] = constructed_bands[orb_switch_b%2][i][k]
        else:
            orb_switch_b += 1
            b_bands[i][k] = constructed_bands[orb_switch_b%2][i][k]
            
        if abs(constructed_bands_SO[orb_switch_b_SO%2][i][k] - constructed_bands_SO[orb_switch_b_SO%2][i][k-1]) < 0.1*abs(max(constructed_bands_SO[orb_switch_b_SO%2][i]) - min(constructed_bands_SO[orb_switch_b_SO%2][i])):
            b_bands_SO[i][k] = constructed_bands_SO[orb_switch_b_SO%2][i][k]
        else:
            orb_switch_b_SO += 1
            b_bands_SO[i][k] = constructed_bands_SO[orb_switch_b_SO%2][i][k]
        
     
        #Using the discontinuous s part only if there is crosing in the band:
        if i < crossed_element_limit:
            
            #Values for the AB band Will get masked when there is band discontinuity
            if ab_band_done == False:
                if abs(constructed_bands[0][i][k-k_dens//2] - constructed_bands[0][i][k-k_dens//2 - 1]) < 0.1*abs(max(constructed_bands[orb_switch_b%2][i]) - min(constructed_bands[orb_switch_b%2][i])):
                        ab_bands[i][k-k_dens//2] = constructed_bands[0][i][k-k_dens//2]
                else:
                    ab_band_done = True
                    ab_mask_array[i][k-k_dens//2] = True
            else:
                ab_mask_array[i][k-k_dens//2] = True
                
            if ab_band_done_SO == False:
                if abs(constructed_bands_SO[0][i][k-k_dens//2] - constructed_bands_SO[0][i][k-k_dens//2 - 1]) < 0.1*abs(max(constructed_bands_SO[orb_switch_b_SO%2][i]) - min(constructed_bands_SO[orb_switch_b_SO%2][i])):
                        ab_bands_SO[i][k-k_dens//2] = constructed_bands_SO[0][i][k-k_dens//2]
                else:
                    ab_band_done_SO = True
                    ab_mask_array_SO[i][k-k_dens//2] = True
            else:
                ab_mask_array_SO[i][k-k_dens//2] = True
        
        #Using the start of the p band instead as the antibonding here
        else:
            '''
            ##Consider This if only adding the first point of the p band to the AB constructed band
            if ab_band_done == False:
                ab_bands[i][k-k_dens//2] = constructed_bands[1][i][k-k_dens//2]
                ab_band_done = True
            else:
                ab_mask_array[i][k-k_dens//2] = True
            ##
            '''
            
            '''
            ##Consider this only if the uncrossed bands will also be fitted with the full p part 
            if ab_band_done == False:
                if abs(constructed_bands[1][i][k-k_dens//2] - constructed_bands[1][i][k-k_dens//2 - 1]) < 0.1*abs(max(constructed_bands[0][i]) - min(constructed_bands[0][i])):
                    ab_bands[i][k-k_dens//2] = constructed_bands[1][i][k-k_dens//2]
                else:
                    ab_band_done = True
                    ab_mask_array[i][k-k_dens//2] = True
            
            
            else:
                ab_mask_array[i][k-k_dens//2] = True
            '''
            
            
            ##Consider this only if the uncrossed bands will also be fitted with the p-part expect the downwards part that can lead to negative tpp
            k_offset = 0
            k_offset_SO = 0
            #Fixing negative hopping parameters
            if i == 5 or i == 11:
                k_offset = 15
                k_offset_SO = 15
                
            if ab_band_done == False:
                if abs(constructed_bands[1][i][k-k_dens//2 + k_offset] - constructed_bands[1][i][k-k_dens//2 - 1 + k_offset]) < 0.1*abs(max(constructed_bands[0][i]) - min(constructed_bands[0][i])) and constructed_bands[1][i][k-k_dens//2 + 1 + k_offset] > constructed_bands[1][i][k-k_dens//2 + k_offset]:
                    ab_bands[i][k-k_dens//2] = constructed_bands[1][i][k-k_dens//2]
                else:
                    ab_band_done = True
                    ab_mask_array[i][k-k_dens//2] = True
            else:
                ab_mask_array[i][k-k_dens//2] = True
            
            if ab_band_done_SO == False:
                if abs(constructed_bands_SO[1][i][k-k_dens//2 + k_offset_SO] - constructed_bands_SO[1][i][k-k_dens//2 - 1 + k_offset_SO]) < 0.1*abs(max(constructed_bands_SO[0][i]) - min(constructed_bands_SO[0][i])) and constructed_bands_SO[1][i][k-k_dens//2 + 1 + k_offset_SO] > constructed_bands_SO[1][i][k-k_dens//2 + k_offset_SO]:
                    ab_bands_SO[i][k-k_dens//2] = constructed_bands_SO[1][i][k-k_dens//2]
                else:
                    ab_band_done_SO = True
                    ab_mask_array_SO[i][k-k_dens//2] = True
            else:
                ab_mask_array_SO[i][k-k_dens//2] = True
                
            
            
            ##
            

#Adding the mirrored band arrays to fill up the full Brillouin Zone
b_bands[:,0:k_dens//2] = n.flip(b_bands[:,k_dens//2:k_dens], axis=1)
b_bands_SO[:,0:k_dens//2] = n.flip(b_bands_SO[:,k_dens//2:k_dens], axis=1)

ab_bands = n.ma.masked_array(ab_bands, mask = ab_mask_array)
ab_bands[:,k_dens//2:k_dens] = n.flip(ab_bands[:,0:k_dens//2], axis=1)
ab_bands_compressed = [i.compressed() for i in ab_bands]
ab_bands_SO = n.ma.masked_array(ab_bands_SO, mask = ab_mask_array_SO)
ab_bands_SO[:,k_dens//2:k_dens] = n.flip(ab_bands_SO[:,0:k_dens//2], axis=1)
ab_bands_compressed_SO = [i.compressed() for i in ab_bands_SO]

k_ab_masked = n.ma.masked_array(n.array([kpoints for i in range(len(element_string_array))]), mask = ab_mask_array)
k_ab_compressed = [i.compressed() for i in k_ab_masked]
k_ab_masked_SO = n.ma.masked_array(n.array([kpoints for i in range(len(element_string_array))]), mask = ab_mask_array_SO)
k_ab_compressed_SO= [i.compressed() for i in k_ab_masked_SO]


#Plotting The construbted bonding and antibonding bands
multiplier_res = 4
fig_constructed, axes_constructed = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_constructed_SO, axes_constructed_SO = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
axes_constructed = axes_constructed.flatten(order="F")
axes_constructed_SO = axes_constructed_SO.flatten(order="F")


for i in range(len(element_string_array)):
    center = (constructed_bands[1,i,0]+constructed_bands[0,i,0])/2
    axes_constructed[i].scatter(kpoints,constructed_bands[0,i,:] - center, s = 3, color = "red", zorder = 1)
    axes_constructed[i].scatter(kpoints,constructed_bands[1,i,:] - center, s = 3, color = "green", zorder = 2)
    #axes_constructed[i].scatter(kpoints,b_bands[i] - center, s = 1, color = "blue", zorder = 4)
    #axes_constructed[i].scatter(kpoints,ab_bands[i] - center, s = 1, color = "orange", zorder = 5)
    
    center_SO = (constructed_bands_SO[1,i,0]+constructed_bands_SO[0,i,0])/2
    axes_constructed_SO[i].scatter(kpoints,constructed_bands_SO[0,i,:] - center_SO, s = 3, color = "red", zorder = 1)
    axes_constructed_SO[i].scatter(kpoints,constructed_bands_SO[1,i,:] - center_SO, s = 3, color = "green", zorder = 2)
    #axes_constructed_SO[i].scatter(kpoints,b_bands_SO[i] - center_SO, s = 1, color = "blue", zorder = 4)
    #axes_constructed_SO[i].scatter(kpoints,ab_bands_SO[i] - center_SO, s = 1, color = "orange", zorder = 5)
    
    #Arranging tick Labels for subplots
    axes_constructed[i].set_xlim([0,1])
    axes_constructed[i].grid()
    axes_constructed_SO[i].set_xlim([0,1])
    axes_constructed_SO[i].grid()
    if (i+1)%4 == 0:
        axes_constructed[i].set_xticks([0, 0.5,1])
        axes_constructed[i].set_xticklabels(["Z",r"$\Gamma$", "Z"])
        axes_constructed_SO[i].set_xticks([0, 0.5,1])
        axes_constructed_SO[i].set_xticklabels(["Z",r"$\Gamma$", "Z"])
    
    else:
        axes_constructed[i].set_xticks([0, 0.5,1])
        axes_constructed[i].set_xticklabels([])
        axes_constructed_SO[i].set_xticks([0, 0.5,1])
        axes_constructed_SO[i].set_xticklabels([])
        
#Stopping a figure from being shown
#plt.close(fig_constructed)
#plt.close(fig_constructed_SO)


#%% Fitting the B and AB bands and Plotting results

#Creating the 4x3 Figures (Full band structure and isolated structure)
multiplier_res = 5
fig_dft, axes_dft = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_dft_SO, axes_dft_SO = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_isolated, axes_isolated = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_b_ab, axes_b_ab = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_params, axes_params = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
axes_dft = axes_dft.flatten(order="F")
axes_dft_SO = axes_dft_SO.flatten(order="F")
axes_isolated = axes_isolated.flatten(order="F")
axes_b_ab = axes_b_ab.flatten(order="F")
axes_params = axes_params.flatten(order="F")


#Fitting and Plotting for all elements
initial_param_guesses = n.zeros((len(element_string_array),4))
fitted_params = n.zeros((len(element_string_array),4))
fitted_params_SO = n.zeros((len(element_string_array),4))
fermi_energy_array = n.zeros(len(element_string_array))
fermi_energy_array_SO = n.zeros(len(element_string_array))
for i, element in enumerate(element_string_array):
    
    #Importing Band Data
    band_data = n.loadtxt("other_elements//%s//+bweights" % element_string_array[i])
    band_data_SO = n.loadtxt("other_elements//%s//+bweights_so" % element_string_array[i])
    band_data_isolated = n.loadtxt("other_elements//isolated//%s//+bweights.txt" % element_string_array[i])
    dft_bands = [ [ band_data[band_index + j * num_band_array[i] ][1] for j in range(k_dens)] for band_index in range(num_band_array[i])]
    dft_bands_SO = [ [ band_data_SO[band_index + j * num_band_SO_array[i] ][1] for j in range(k_dens)] for band_index in range(num_band_SO_array[i])]
    s_band_isolated = [ band_data_isolated[isolated_s_p_band_indices[i][0] + j * num_band_array[i] ][1] for j in range(k_dens)]
    p_band_isolated = [ band_data_isolated[isolated_s_p_band_indices[i][1] + j * num_band_array[i] ][1] for j in range(k_dens)]

    #Determination of Initial Hamiltonian Parameters (Using Isolated Bands and Harrison Scaling, and increasing Delta for when there is no crossing to help fit)
    atomic_delta = p_band_isolated[len(kpoints)//2] - s_band_isolated[len(kpoints)//2]
    initial_param_guesses[i][0] =  atomic_delta
    if i >= 5:
        initial_param_guesses[i][0] = 3 * atomic_delta
    initial_param_guesses[i][1] = 1.40 * 7.62 * (1/lat_const_array[i]**2)
    initial_param_guesses[i][2] = 3.24 * 7.62 * (1/lat_const_array[i]**2)
    initial_param_guesses[i][3] = 1.84 * 7.62 * (1/lat_const_array[i]**2) 
    
    
    #Centering the DFT bands so they can be fitted to the Eval Functions
    center_dft_bands = (ab_bands[i][0] + b_bands[i][0])/2
    center_dft_bands_SO = (ab_bands_SO[i][0] + b_bands_SO[i][0])/2

    #Fitting for parameters(Scipy otimize)
    combined_k_array = n.ma.append(kpoints, k_ab_compressed[i])
    combined_k_array_SO = n.ma.append(kpoints, k_ab_compressed_SO[i])
    combined_band_data = n.ma.append(b_bands[i], ab_bands_compressed[i]) - center_dft_bands
    combined_band_data_SO = n.ma.append(b_bands_SO[i], ab_bands_compressed_SO[i]) - center_dft_bands_SO
    fitted_params[i] = optimize.curve_fit(combined_evals, combined_k_array, combined_band_data, p0 = initial_param_guesses[i])[0]
    #fitted_params_SO[i] = optimize.curve_fit(combined_evals, combined_k_array_SO, combined_band_data_SO, p0 = initial_param_guesses[i])[0]

    
    #Adding the DFT bands to subplots:
    for band in dft_bands:
        axes_dft[i].plot(kpoints, band, color="black", linewidth = 1, zorder=2)
    for band_SO in dft_bands_SO:
        axes_dft_SO[i].plot(kpoints, band_SO, color="black", linewidth = 1, zorder=2)
    axes_isolated[i].plot(kpoints, s_band_isolated)
    axes_isolated[i].plot(kpoints, p_band_isolated)
    axes_b_ab[i].plot(kpoints, b_bands[i], linewidth=1, color="orange")
    axes_b_ab[i].plot(kpoints, ab_bands[i], linewidth=1, color="orange")
    
    #Adding the Fitted Bands to subplots
    #fitted_params[i][3] = 0     # Uncomment to plot unmixed bands
    fitted_b_band = n.array(eval_B(kpoints,fitted_params[i][0], fitted_params[i][1],fitted_params[i][2], fitted_params[i][3]))
    fitted_ab_band = n.array(eval_AB(kpoints,fitted_params[i][0], fitted_params[i][1],fitted_params[i][2], fitted_params[i][3]))
    #fitted_b_band_SO = n.array(eval_B(kpoints,fitted_params_SO[i][0], fitted_params_SO[i][1],fitted_params_SO[i][2], fitted_params_SO[i][3]))
    #fitted_ab_band_SO = n.array(eval_AB(kpoints,fitted_params_SO[i][0], fitted_params_SO[i][1],fitted_params_SO[i][2], fitted_params_SO[i][3]))

    
    #Aligning the fitted bands with the dft bands
    fermi_energy_array[i] = (fitted_ab_band[0] + fitted_b_band[0])/2 - center_dft_bands
    #fermi_energy_array_SO[i] = (fitted_ab_band_SO[0] + fitted_b_band_SO[0])/2 - center_dft_bands_SO
    fitted_b_band = fitted_b_band - fermi_energy_array[i]
    fitted_ab_band = fitted_ab_band - fermi_energy_array[i]
    #fitted_b_band_SO = fitted_b_band_SO - fermi_energy_array_SO[i]
    #fitted_ab_band_SO = fitted_ab_band_SO - fermi_energy_array_SO[i]
  
    axes_dft[i].plot(kpoints, fitted_b_band, linestyle="solid", linewidth = 3, zorder = 1)
    axes_dft[i].plot(kpoints, fitted_ab_band, linestyle="dashed", linewidth = 3, zorder = 1)
    #axes_dft_SO[i].plot(kpoints, fitted_b_band_SO, linestyle="solid", color = "red", linewidth = 3, zorder = 1)
    #axes_dft_SO[i].plot(kpoints, fitted_ab_band_SO, linestyle="dashed", color = "green", linewidth = 3, zorder = 1)
    
    axes_b_ab[i].plot(kpoints, fitted_b_band , linestyle="solid", color = "red", linewidth = 3, zorder = 1)
    axes_b_ab[i].plot(kpoints, fitted_ab_band, linestyle="dashed", color = "green", linewidth = 3, zorder = 1)
    
    #Adding Text in the Subplots
    axes_dft[i].text(0.5,0.3, "%s" % (element_string_array[i]), fontsize = 15, transform = axes_dft[i].transAxes, horizontalalignment='center')
    #axes_dft_SO[i].text(0.5,0.3, "%s" % (element_string_array[i]), fontsize = 15, transform = axes_dft_SO[i].transAxes, horizontalalignment='center')
    axes_b_ab[i].text(0.5,0.3, "%s" % (element_string_array[i]), fontsize = 15, transform = axes_b_ab[i].transAxes, horizontalalignment='center')
    axes_isolated[i].text(0.5,0.3, "%s\n $\Delta=%.2f$ eV" % (element_string_array[i], atomic_delta), fontsize = 12, transform = axes_isolated[i].transAxes, horizontalalignment='center')
    axes_params[i].text(0.5,0.5, "%s\n$\Delta$ = %.2f eV\n$t_{ss}$ = %.2f eV\n$t_{pp}$ = %.2f eV\n$t_{sp}$ = %.2f eV" % (element_string_array[i], fitted_params[i][0], fitted_params[i][1], fitted_params[i][2], fitted_params[i][3] ), fontsize = 9, transform = axes_params[i].transAxes, horizontalalignment='center', verticalalignment='center')

    
    #Arranging tick Labels for subplots
    fig_dft.supylabel('E (eV)', fontsize = 13)
    fig_b_ab.supylabel('E (eV)', fontsize = 13)
    fig_isolated.supylabel('E (eV)', fontsize = 13)
    axes_dft[i].set_xlim([0,1])
    axes_dft[i].set_ylim(min(fitted_b_band) - 1,max(fitted_ab_band) + 2)
    axes_dft_SO[i].set_xlim([0,1])
    axes_dft_SO[i].set_ylim(min(fitted_b_band) - 1,max(fitted_ab_band) + 2)
    axes_b_ab[i].set_ylim(min(b_bands[i]) - 1,max(fitted_ab_band) + 2)
    axes_isolated[i].set_xlim([0,1])
    axes_dft[i].grid()
    axes_dft_SO[i].grid()
    axes_isolated[i].grid()
    axes_params[i].set_xticks([])
    axes_params[i].set_xticklabels([])
    axes_params[i].set_yticks([])
    axes_params[i].set_yticklabels([])
    if (i+1)%4 == 0:
        axes_dft[i].set_xticks([0, 0.5,1])
        axes_dft[i].set_xticklabels(["Z",r"$\Gamma$", "Z"])
        axes_dft_SO[i].set_xticks([0, 0.5,1])
        axes_dft_SO[i].set_xticklabels(["Z",r"$\Gamma$", "Z"])
        axes_b_ab[i].set_xticks([0, 0.5,1])
        axes_b_ab[i].set_xticklabels(["Z",r"$\Gamma$", "Z"])
        axes_isolated[i].set_xticks([0, 0.5,1])
        axes_isolated[i].set_xticklabels(["Z",r"$\Gamma$", "Z"])

    else:
        axes_dft[i].set_xticks([0, 0.5,1])
        axes_dft[i].set_xticklabels([])
        axes_dft_SO[i].set_xticks([0, 0.5,1])
        axes_dft_SO[i].set_xticklabels([])
        axes_b_ab[i].set_xticks([0, 0.5,1])
        axes_b_ab[i].set_xticklabels([])
        axes_isolated[i].set_xticks([0, 0.5,1])
        axes_isolated[i].set_xticklabels([])
        
    #Checking if Topological based on Parameters and Printing
    topo_1D = "trivial"
    topo_2D = "trivial"
    if fitted_params[i][0] < 2*abs(fitted_params[i][1] + fitted_params[i][2]):
        topo_1D = "TOPOLOGICAL"
    if fitted_params[i][0] < 3*abs(fitted_params[i][1] + fitted_params[i][2]/2):
        topo_2D = "TOPOLOGICAL"
    print("%s: %s, %s" % (element, topo_1D, topo_2D))
    

#Stopping a figure from being shown
#plt.close(fig_dft)
plt.close(fig_dft_SO)
plt.close(fig_isolated)
plt.close(fig_b_ab)
plt.close(fig_params)

fig_dft.tight_layout()




#%% 1D, DFT Lattice Constant Determination
# Indexing (only Alkali): Li(0), Be(1), Na(2), Mg(3), K(4), Ca(5), Rb(6), Sr(7), Cs(8), Ba(9), Fr(10), Ra(11)
element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]
a_eq_guess_array = [3.00, 2.1, 3.25, 2.9, 4.05, 4.00, 4.30, 4.45, 4.70, 3.80, 4.65, 5.10]
lat_limits_array = [[2,4], [2,4], [2,4], [2,4], [3,8],[3,8], [3,8], [3,8], [3,8], [3,8], [3,8], [3,8]]
fit_interval_array = [[11,11],[2,11],[11,11],[11,11],[11,11],[11,11],[11,11],[11,11],[11,11],[11,11],[11,11],[11,11]]


#Creating the 4x3 Figures (Energy states and edge state localization Eigenvectors)
dpi = 300
multiplier_res = 5
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(1.4 * multiplier_res, 1 * multiplier_res), dpi=dpi)
ax = ax.flatten(order="F")
fig.supylabel("Total Energy (eV)", fontsize = 13)
fig.supxlabel("Lattice Constant (Å)", fontsize = 13)



#Quadratic function to fit for minimum energy value
def quad(x, a, b, h, k):
    return a*(b*(x-h))**2 + k

for element_idx in range(len(element_string_array)):

    lat = n.linspace(lat_limits_array[element_idx][0], lat_limits_array[element_idx][1], int((lat_limits_array[element_idx][1]-lat_limits_array[element_idx][0])/0.05)+1)
    
    eq_index_guess = n.where(lat==a_eq_guess_array[element_idx])[0][0]
    

    
    #Parsing the combined energy files for total energy
    energy = []
    energy_file = open("other_elements//%s//energies.txt" % element_string_array[element_idx] , "r")
    for line in energy_file:
        energy += [float(line.split()[1])]
    energy_file.close()
    
    
    #Fitting quadratic function to potential surface
    lat_fit = lat[eq_index_guess - fit_interval_array[element_idx][0]: eq_index_guess + fit_interval_array[element_idx][1]]
    energy_fit = energy[eq_index_guess - fit_interval_array[element_idx][0]: eq_index_guess + fit_interval_array[element_idx][1]]
    par, par_covariance = optimize.curve_fit(quad,lat_fit, energy_fit, p0 = [1,1,a_eq_guess_array[element_idx],min(energy)])

    #Plotting 
    ax[element_idx].plot(lat, energy, marker='o', label="FPLO data")
    ax[element_idx].plot(n.linspace(lat_fit[0],lat_fit[-1], 200), quad(n.linspace(lat_fit[0],lat_fit[-1], 200),par[0], par[1], par[2], par[3]), label = "Quadratic fit")
    ax[element_idx].text(0.2,0.5, "%s \n $a_{eq} = %.3f \; Å$" % (element_string_array[element_idx], par[2]),transform = ax[element_idx].transAxes, fontsize = 12)
    ax[element_idx].tick_params(axis = 'x', labelsize = 10)
    ax[element_idx].tick_params(axis = 'y', labelsize = 5)
    ax[element_idx].ticklabel_format(useOffset=False)
    #plt.xlim(lat_fit[0]-0.5, lat_fit[-1]+0.5)
    ax[element_idx].grid(True)



fig.tight_layout()



#%% 1D, Real Space, Finite Hamiltonian

element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]
a_array = [3.00, 2.1, 3.25, 2.9, 4.05, 4.00, 4.30, 4.45, 4.70, 3.80, 4.65, 5.10]


N = 2000
dos_energy_width = 0.05
dos_edge_state_multiplier = 10

#Creating the 4x3 Figures (Energy states and edge state localization Eigenvectors)
multiplier_res = 5
fig_gap, axes_gap = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_edge, axes_edge = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_dos, axes_dos = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
axes_gap = axes_gap.flatten(order="F")
axes_edge = axes_edge.flatten(order="F")
axes_dos = axes_dos.flatten(order="F")


for element_idx, element in enumerate(element_string_array):

    delta = fitted_params[element_idx][0]
    tss = fitted_params[element_idx][1]
    tpp = fitted_params[element_idx][2]
    tsp = fitted_params[element_idx][3]
    
    
    #Definining the Real Space Hamiltonian Array 
    
    block_left_hopping = n.array([[-tss,-tsp],[tsp,tpp]])
    block_onsite = n.array([[-delta/2,0],[0,delta/2]])
    block_right_hopping = n.array([[-tss,tsp],[-tsp,tpp]])
    H = n.zeros((2*N,2*N))
    
    for i in range(0,N):
         H[2*i:2*(i+1),2*i:2*(i+1)] = block_onsite
         if i<N-1:
             H[2*i:2*(i+1),2*(i+1):2*(i+2)] = block_right_hopping
             H[2*(i+1):2*(i+2),2*i:2*(i+1)] = block_left_hopping
    
    #Diagonalizing the Hamiltonian
    evals,evecs = la.eigh(H)
    evals=evals.real
    
    
    #Finding the edge states
    edge_tolerance = 0.2
    edge_states = [i for i in range(len(evals)) if edge_tolerance < n.abs(evecs[0][i]) or edge_tolerance < n.abs(evecs[1][i]) or edge_tolerance < n.abs(evecs[-1][i]) or edge_tolerance < n.abs(evecs[-2][i]) ]
    
    #Determining the Energy gap
    energy_gap = evals[len(evals)//2+1] - evals[len(evals)//2-2]
    
    #Fermi energy determination 
    fermi_energy = (evals[len(evals)//2-2] + evals[0])/2 #Alkali, Center of Bonding Band
    if element_idx%2 == 1:
        fermi_energy = (evals[len(evals)//2 - 2] + evals[len(evals)//2+1])/2  #Alkaline Earth (Trivial), Center of Gap
        if len(edge_states) == 2:
            fermi_energy = evals[edge_states[0]] #Alkaline Earth (Topological), at the degenerate edge states

    evals = evals - fermi_energy
    
    
    #### Eigenvalues Plotting
    #Plotting energy eigenvalues (red color if hey are edge states)
    for i, j in enumerate(evals):
        if i in edge_states:
            axes_gap[element_idx].axhline(j, color = "red", linestyle = "dotted")
        else:
            axes_gap[element_idx].axhline(j, color = "royalblue")
            
    axes_gap[element_idx].text(0.5,0.7, "%s" % (element), fontsize = 15, transform = axes_gap[element_idx].transAxes, horizontalalignment='center')
    

    
    
    #### Eigenstate Plotting
    # Edge Eigenstates (adding diferent text depending on if there are edge states or not)
    eigenstate_index_array = edge_states
    label_array= ("Left Edge State", "Right Edge State")
    if len(edge_states) == 2:
        for i,color,style in list(zip(edge_states, ["Blue", "Orange"], ["solid", "dashed"])):
            axes_edge[element_idx].plot(n.linspace(1,N,N),(evecs[0::2,i])**2 + (evecs[1::2,i])**2, label=label_array[(i+1)%2], color = color, linestyle = style)
        axes_edge[element_idx].text(0.5, 0.5, "%s\n(E = %.2f, %.2f)\nGap: %.2f eV" % (element, evals[edge_states[0]], evals[edge_states[1]], abs(evals[max(edge_states)+1] - evals[min(edge_states)-1])), fontsize = 12, transform = axes_edge[element_idx].transAxes, horizontalalignment='center', verticalalignment='center')
        
    else:
        gap = abs(evals[len(evals)//2] - evals[len(evals)//2-1])
        axes_edge[element_idx].text(0.5, 0.5, "%s\nNo Gap State\nGap: %.2f eV" % (element, gap), fontsize = 12, transform = axes_edge[element_idx].transAxes, horizontalalignment='center', verticalalignment='center')
    
    
    #### DOS Plotting
    dos_array = []
    dos_energy_array = []
    dos_energy_window_min = min(evals)
    while dos_energy_window_min <= max(evals):
        dos_array += [ len([i for i in evals if dos_energy_window_min <= i < dos_energy_window_min + dos_energy_width])/(N*dos_energy_width)  ]
        dos_energy_array += [dos_energy_window_min + dos_energy_width/2]
        dos_energy_window_min += dos_energy_width
    gap_dos_idx = [idx for (idx,dos) in enumerate(dos_array) if dos == 0]
    

    axes_dos[element_idx].text(0.7, 0.5, "%s" % element, fontsize = 14, transform = axes_dos[element_idx].transAxes, horizontalalignment='center', verticalalignment='center')
    axes_dos[element_idx].tick_params(axis = 'both', labelsize = 10)
    axes_dos[element_idx].set_xlim(0, max(dos_array))
    axes_dos[element_idx].fill_betweenx(dos_energy_array[0:gap_dos_idx[0]], dos_array[0:gap_dos_idx[0]], color='C0')
    axes_dos[element_idx].fill_betweenx(dos_energy_array[gap_dos_idx[-1]:-1], dos_array[gap_dos_idx[-1]:-1], color='C1')
    #axes_dos[element_idx].grid()
    if len(edge_states) == 2:
        edge_state_dos_idx = [idx_forward for (idx,idx_forward) in zip(gap_dos_idx, gap_dos_idx[1:]) if idx_forward != idx + 1]
        axes_dos[element_idx].plot([0, dos_edge_state_multiplier * n.array(dos_array)[edge_state_dos_idx[0]-1]], [dos_energy_array[edge_state_dos_idx[0]-1], dos_energy_array[edge_state_dos_idx[0]-1]], color = "red", linewidth=1)
        axes_dos[element_idx].fill_betweenx(dos_energy_array[edge_state_dos_idx[0]-1:edge_state_dos_idx[-1]], dos_array[edge_state_dos_idx[0]-1:edge_state_dos_idx[-1]], color='red')
    
    #Arranging tick Labels for subplots
    fig_gap.supylabel('E (eV)', fontsize = 13)
    fig_edge.supxlabel('Chain Site', fontsize = 13)
    fig_edge.supylabel(r'$|\Psi|^2$', fontsize = 13)
    fig_dos.supxlabel("DOS $[eV \\cdot a]^{-1}$", fontsize = 13)
    fig_dos.supylabel("$E$ (eV)", fontsize = 13)
    axes_gap[element_idx].tick_params(axis = 'y', labelsize = 8)
    axes_gap[element_idx].set_xticklabels([])
    axes_gap[element_idx].set_xticks([])
    axes_edge[element_idx].set_xlim([0 - 1, N+1 + 1])
    if (element_idx+1)%4 != 0:
        axes_edge[element_idx].set_xticklabels([])
    
    

#Stopping a figure from being shown
fig_gap.tight_layout()
fig_edge.tight_layout()
fig_dos.tight_layout()
#plt.close(fig_gap)
#plt.close(fig_edge)
#plt.close(fig_dos)





#%% 1D K-Space Hamiltonian


#### Hamiltonian Parameters
# Indexing: Li(0), Be(1), Na(2), Mg(3), K(4), Ca(5), Rb(6), Sr(7), Cs(8), Ba(9), Fr(10), Ra(11)
element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]
kpath = ("K", r"$\Gamma$", "K")



chosen_dim = 2
#Initializing the hamiltonian and filling it for all k-points
k_array = n.linspace(-n.pi, n.pi, 201)
H = n.zeros((chosen_dim,chosen_dim),dtype=n.cdouble)
H_no_sp = n.zeros((chosen_dim,chosen_dim),dtype=n.cdouble)

#eigenvalues and eigenvectors array initialization (eval indexing: [kx][ky][band] evec indexing: [kx][ky][orbital][band])
evals = n.zeros((len(k_array),chosen_dim), dtype=n.cdouble)
evecs = n.zeros((len(k_array),chosen_dim,chosen_dim), dtype=n.cdouble)
evals_no_sp = n.zeros((len(k_array),chosen_dim), dtype=n.cdouble)
evecs_no_sp = n.zeros((len(k_array),chosen_dim,chosen_dim), dtype=n.cdouble)

#for chosen_element in range(len(element_string_array)):
for chosen_element in [1]:
    
    delta = fitted_params[chosen_element][0]
    tss = fitted_params[chosen_element][1]
    tpp = fitted_params[chosen_element][2]
    tsp = fitted_params[chosen_element][3]
    fermi_energy = 0
    
    #Filling the Hamiltonian and diagonalizing for all k-points
    for i in range(len(k_array)):
        k = k_array[i]
        
        #ss part
        H[0,0] = -2*tss*n.cos(k) - delta/2 
        H_no_sp[0,0] = -2*tss*n.cos(k) - delta/2
        
        #p-p part
        H[1,1] = 2*tpp*n.cos(k) + delta/2
        H_no_sp[1,1] = 2*tpp*n.cos(k) + delta/2
        
        
        #s-px part
        H[0,1] = -2*tsp*sym.I*n.sin(k)
        H[1,0] = 2*tsp*sym.I*n.sin(k)
        
        
        #diagonalizing the Hamiltonian and populating the evecs and evals arrays
        evals_temp,evecs_temp = n.linalg.eigh(H)
        evals_temp=evals_temp.real
        evals[i] = evals_temp
        evecs[i] = evecs_temp
        
        evals_temp_no_sp,evecs_temp_no_sp = n.linalg.eigh(H_no_sp)
        evals_temp_no_sp=evals_temp_no_sp.real
        evals_no_sp[i] = evals_temp_no_sp
        evecs_no_sp[i] = evecs_temp_no_sp
        
    
    
    #Fermi energy determination by counting filled states
    chosen_filling = 0.5
    fermi_energy = n.sort(evals.T[0])[int(len(evals.T[0])//(1/chosen_filling))]
    fermi_energy_no_sp = n.sort(evals_no_sp.T[0])[int(len(evals_no_sp.T[0])//(1/chosen_filling))]
    if chosen_element%2 != 0:
        chosen_filling = 1
        fermi_energy = n.sort(evals.T[0])[int(len(evals.T[0])//(1/chosen_filling))-1]
        fermi_energy_no_sp = n.sort(evals_no_sp.T[0])[int(len(evals_no_sp.T[0])//(1/chosen_filling))-1]

    
    #Plotting the energy eigenvalues
    plt.plot(k_array, evals.T[0] - fermi_energy, linewidth = 5, color = "red", label = r"tsp = %.2f eV" % tsp)
    plt.plot(k_array, evals.T[1] - fermi_energy, linewidth = 5, color="red")
    
    plt.plot(k_array, evals_no_sp.T[0] - fermi_energy_no_sp, linewidth = 4, linestyle="dotted", color="green", label = "tsp=0")
    plt.plot(k_array, evals_no_sp.T[1] - fermi_energy_no_sp, linewidth = 4, linestyle="dotted", color="green")
    
    plt.xlabel(r"$k \; (\pi/a)$", fontsize=15)
    plt.ylabel(r"E - Ef (eV)", fontsize=15)
    plt.tick_params(axis = 'y', labelsize = 15)
    plt.tick_params(axis = 'x', labelsize = 15)
    plt.legend(loc="upper right", fontsize=15)
    plt.grid()
    plt.show()
    plt.close()





#%% 1D, Density of States


element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]
y_lim_array = [7,5,15,15,28,40,25,30,55,35,55,35]
N = 400

#Creating the 4x3 Figures
dpi = 300
multiplier_res = 6
fig_real, ax_real = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=dpi)
fig_k, ax_k = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=dpi)
fig_dft, ax_dft = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=dpi)
ax_real = ax_real.flatten(order="F")
ax_k = ax_k.flatten(order="F")
ax_dft = ax_dft.flatten(order="F")

fig_real.supylabel("DOS $[eV \\cdot Å]^{-1}$", fontsize = 15)
fig_k.supylabel("DOS $[eV \\cdot Å]^{-1}", fontsize = 15)
fig_dft.supylabel("DOS $[eV \\cdot Å]^{-1}$", fontsize = 15)
fig_real.supxlabel("$E-E_F$ (eV)", fontsize = 15)
fig_k.supxlabel("$E-E_F$ (eV)", fontsize = 15)
fig_dft.supxlabel("$E-E_F$ (eV)", fontsize = 15)


for element_idx, element in enumerate(element_string_array):
    
    

    #### DFT DOS Plotting
    dft_dos = n.loadtxt("other_elements//%s//+dos.total" % element, unpack = True)
    ax_dft[element_idx].plot(dft_dos[0], dft_dos[1])
    ax_dft[element_idx].set_xlim(-2,5)
    ax_dft[element_idx].set_ylim(0,y_lim_array[element_idx])
    ax_dft[element_idx].text(0.1,0.7, "%s" % element,transform = ax_dft[element_idx].transAxes, fontsize = 15)
    ax_dft[element_idx].tick_params(axis = 'both', labelsize = 10)





#Stopping a figure from being shown
fig_real.tight_layout()
fig_k.tight_layout()
fig_dft.tight_layout()
plt.close(fig_real)
plt.close(fig_k)
#plt.close(fig_dft)








#%% 2D, Real Space, Finite Hamiltonian

element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]

#Creating the 4x3 Figures (Energy states and edge state localization Eigenvectors)
multiplier_res = 5
fig_gap_slab, axes_gap_slab = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_edge_slab, axes_edge_slab = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
fig_dos_slab, axes_dos_slab = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
axes_gap_slab = axes_gap_slab.flatten(order="F")
axes_edge_slab = axes_edge_slab.flatten(order="F")
axes_dos_slab = axes_dos_slab.flatten(order="F")



#Size of finite sample l-direction (x-direction is finite):
Nl = 100
n_orbs = 6
k_dens_slab = 201
dos_energy_width = 0.05
kdens_strip = k_dens_slab
kpoints_slab = n.linspace(-2*n.pi/(3*n.sqrt(3))-0.5, 2*n.pi/(3*n.sqrt(3))+0.5,k_dens_slab)


for element_idx, element in enumerate(element_string_array):
    
    
    #Parameters
    delta = fitted_params[element_idx][0]
    tss = fitted_params[element_idx][1]
    tpp_sigma = fitted_params[element_idx][2]
    tpp_pi = 0
    tsp = fitted_params[element_idx][3]
    

    #eigenvalues and eigenvectors array initialization (eval indexing: [kx][band] evec indexing: [kx][orbital][band])
    evals = n.zeros((k_dens_slab,n_orbs*Nl), dtype=n.cdouble)
    evecs = n.zeros((k_dens_slab,n_orbs*Nl,n_orbs*Nl), dtype=n.cdouble)
    
    
    #Filling the Hamiltonian and diagonalizing for all k-points
    #Eval Indexing is l=0  [k][band]
    for k_index, kx in enumerate(kpoints_slab):
        H = n.zeros((Nl*n_orbs,Nl*n_orbs),dtype=n.cdouble)
        
        for l in range(Nl):
            
            #position indices (lm = l-1)
            pos_l = n_orbs*l
            pos_lm = n_orbs*(l-1)
            
            #complex exponents (only first (x) lattice direction this time)
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
        evals_temp,evecs_temp = n.linalg.eigh(H)
        evals_temp=evals_temp.real
        evals[k_index] = evals_temp
        evecs[k_index] = evecs_temp
        

 
    
    #Fermi energy determination
    fermi_energy = ( evals[0,0] + evals[0,len(evals[0])//6+2]  )/2 #Alakali center of s bands Fermi Energy
    if element_idx%2==1:
        if element_idx <= 5:
            fermi_energy = evals[0, len(evals[0])//3].real #Alakaline Earth top of s bands Fermi energy
        else:
            fermi_energy = (max(evals[:, len(evals[0])//3-1].real) + min(evals[:, len(evals[0])//3].real))/2
            

    evals = n.array(evals) - fermi_energy
    
    

    
    #Checking if Element is in Topological Phase based on Parameters
    is_topo = True
    if delta > 3*abs(tss + tpp_sigma/2):
        is_topo = False
    
    
    #Determining edge states while excluding Flat Bands { Evec Indexing: [k][site 0 6 orbitals, site 1 6 orbitals, ...][band] }
    edge_tolerance = 0.5
    edge_states = [ band_index for band_index in [len(evals[0])//2-1,len(evals[0])//2] if any ([edge_tolerance < abs(evecs[0][orb][band_index])   for orb in [0,1,2,3,4,5,-1,-2,-3,-4,-5,-6]])  ]
 

    #### Eigenvalues Plotting
    #Plotting energy eigenvalues (red color if hey are edge states)
    for band_index in range(len((evals[0]))):
        if band_index in edge_states and is_topo == True:
            axes_gap_slab[element_idx].plot(kpoints_slab, evals[:,band_index], color = "red", linestyle="dashed", zorder = 2)
        else:
            axes_gap_slab[element_idx].plot(kpoints_slab, evals[:,band_index], color = "green", zorder = 1)
        
    axes_gap_slab[element_idx].text(0.5,0.2, "%s" % (element), fontsize = 15, transform = axes_gap_slab[element_idx].transAxes, horizontalalignment='center')
    axes_gap_slab[element_idx].tick_params(axis = 'y', labelsize = 8)
    axes_gap_slab[element_idx].set_xticklabels([])
    axes_gap_slab[element_idx].set_xticks([])
    


    #### Eigenstate Plotting
    # Edge Eigenstates (adding diferent text depending on if there are edge states or not)
    eigenstate_index_array = edge_states
    label_array= ("Left Edge State", "Right Edge State")
    if len(edge_states) == 2 and is_topo == True:
        gap = abs( min(evals[:,len(evals[0])//2+1]) - max(evals[:,len(evals[0])//2-2]))
        edge_state_center_energy = (max(evals[:,edge_states[0]]) + min(evals[:,edge_states[0]]))/2
        for i,color,style in list(zip(edge_states, ["Blue", "Orange"], ["solid", "dashed"])):
            axes_edge_slab[element_idx].plot(n.linspace(1,Nl,Nl),n.sum([ evecs[0,orb::n_orbs,i]**2  for orb in range(n_orbs)], axis = 0 ), label=label_array[(i+1)%2], color = color, linestyle = style)
        axes_edge_slab[element_idx].text(0.5, 0.5, "%s\nE = %.2f eV\nGap: %.2f eV" % (element, edge_state_center_energy, gap), fontsize = 12, transform = axes_edge_slab[element_idx].transAxes, horizontalalignment='center', verticalalignment='center')
        
    else:
        gap = abs( min(evals[:,len(evals[0])//3]) - max(evals[:,len(evals[0])//3-1]))
        axes_edge_slab[element_idx].text(0.5, 0.5, "%s\nNo Gap State\nGap: %.2f" % (element, gap), fontsize = 12, transform = axes_edge_slab[element_idx].transAxes, horizontalalignment='center', verticalalignment='center')
 
   
    
    #Arranging tick Labels for subplots
    axes_gap_slab[element_idx].set_xlim([min(kpoints_slab),max(kpoints_slab)])
    axes_gap_slab[element_idx].grid()
    fig_gap_slab.supylabel('E (eV)', fontsize = 13)
    fig_edge_slab.supxlabel(r'Unit Cell in $\vec{e}_2$ Direction', fontsize = 13)
    fig_edge_slab.supylabel(r'$|\Psi|^2$', fontsize = 13)
    axes_edge_slab[element_idx].set_xlim([0,Nl+1])
    if (element_idx+1)%4 == 0:
        axes_gap_slab[element_idx].set_xticks([min(kpoints_slab), kpoints_slab[len(kpoints_slab)//2],max(kpoints_slab)])
        axes_gap_slab[element_idx].set_xticklabels(["K",r"$\Gamma$", "K"])
    else:
        axes_gap_slab[element_idx].set_xticks([min(kpoints_slab), kpoints_slab[len(kpoints_slab)//2],max(kpoints_slab)])
        axes_gap_slab[element_idx].set_xticklabels([])
        axes_edge_slab[element_idx].set_xticklabels([])
        
           
    #### DOS Plotting
    evals_dos = n.copy(evals)
    evals_dos = evals_dos.real.reshape(-1)
    evals_dos.sort()
    dos_array = []
    dos_energy_array = []
    dos_edge_array = []
    dos_edge_energy_array = []
    dos_energy_window_min = min(evals_dos)
    while dos_energy_window_min <= max(evals_dos):
        dos_array += [ len([i for idx, i in enumerate(evals_dos) if dos_energy_window_min <= i < dos_energy_window_min + dos_energy_width])/(Nl*kdens_strip*dos_energy_width * (3*n.sqrt(3))/2) ]
        dos_energy_array += [dos_energy_window_min + dos_energy_width/2]
        dos_energy_window_min += dos_energy_width
    gap_dos_idx = [idx for (idx,dos) in enumerate(dos_array) if dos == 0]
    
    axes_dos_slab[element_idx].tick_params(axis = 'both', labelsize = 12)
    axes_dos_slab[element_idx].set_xlim(0,max(dos_array)/10)

    if len(gap_dos_idx) > 0:
        axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[0:gap_dos_idx[0]-1], dos_array[0:gap_dos_idx[0]-1], color='C0')
        axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[gap_dos_idx[-1]-1:-1], dos_array[gap_dos_idx[-1]-1:-1], color='C1')
        
        if element_idx <=5:
            axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[gap_dos_idx[0]-2:gap_dos_idx[0]+1], dos_array[gap_dos_idx[0]-2:gap_dos_idx[0]+1], color='C1')
 
    else:
        if element_idx <= 5:
            transition_point = [idx for idx,i in enumerate(dos_energy_array) if -0.02 < i < 0.02][0]
            axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[0:transition_point], dos_array[0:transition_point], color='C0')
            axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[transition_point:-1], dos_array[transition_point:-1], color='C1')
      
        else:
            transition_point = [idx for idx,i in enumerate(dos_energy_array) if -0.02 < i + fermi_energy < 0.02][0]
            axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[0:transition_point], dos_array[0:transition_point], color='C0')
            axes_dos_slab[element_idx].fill_betweenx(dos_energy_array[transition_point:-1], dos_array[transition_point:-1], color='C1')
      
    #Plotting the topological edge state DOS
    if element_idx <= 6:
        edge_state_dos_idx = [idx_forward for (idx,idx_forward) in zip(gap_dos_idx, gap_dos_idx[1:]) if idx_forward != idx + 1]
        idx_edge_state_start = n.argwhere(dos_energy_array == dos_energy_array[n.abs(dos_energy_array - min(evals[:,edge_states[0]].real)).argmin()])[0][0]
        idx_edge_state_end = n.argwhere(dos_energy_array == dos_energy_array[n.abs(dos_energy_array - max(evals[:,edge_states[1]].real)).argmin()])[0][0]
        
        if element_idx <= 1:
            axes_dos_slab[element_idx].fill_betweenx( dos_energy_array[ idx_edge_state_start : idx_edge_state_end ], n.array(dos_array[idx_edge_state_start : idx_edge_state_end])   , color='red', zorder=10)
        else:
            axes_dos_slab[element_idx].fill_betweenx( dos_energy_array[ idx_edge_state_start : idx_edge_state_end ], n.array(dos_array[idx_edge_state_start : idx_edge_state_end])/2.2   , color='red', zorder=10)
 
    axes_dos_slab[element_idx].grid()
    
    #Adding the element name to the plot
    axes_dos_slab[element_idx].text(0.7,0.15, "%s" % (element), fontsize = 15, transform = axes_dos_slab[element_idx].transAxes)
    
fig_dos_slab.supxlabel("DOS $[eV \\cdot a^2]^{-1}$", fontsize = 17)
fig_dos_slab.supylabel('E (eV)', fontsize = 17)
    

#Stopping a figure from being shown
fig_gap_slab.tight_layout()
fig_edge_slab.tight_layout()
fig_dos_slab.tight_layout()
plt.close(fig_gap_slab)
plt.close(fig_edge_slab)
#plt.close(fig_dos_slab)


    




#%% 2D K-Space Hamiltonian

element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]
kpath = (r"$\Gamma$", "K", "M", r"$\Gamma$")

#Creating the 4x3 Figures (Energy states and edge state localization Eigenvectors)
multiplier_res = 5
fig_hex_k, axes_hex_k = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
axes_hex_k = axes_hex_k.flatten(order="F")


#Initializing the hamiltonian and filling it for all k-points
n_orbs = 6
k_array = n.linspace(-(4*n.pi/(3*n.sqrt(3))),4*n.pi/(3*n.sqrt(3)),201)
H = n.zeros((n_orbs,n_orbs),dtype=n.cdouble)
hopping_scaling = (2.15/1.3)**2



#for element_idx, element in enumerate(element_string_array):
for element_idx, element in enumerate(element_string_array):
    
    #eigenvalues and eigenvectors array initialization (eval indexing: [kx][ky][band] evec indexing: [kx][ky][orbital][band])
    evals = n.zeros((len(k_array),len(k_array),n_orbs), dtype=n.cdouble)
    evecs = n.zeros((len(k_array),len(k_array),n_orbs,n_orbs), dtype=n.cdouble)
    
    #Parameters
    delta = fitted_params[element_idx][0]
    tss = fitted_params[element_idx][1] * hopping_scaling
    tpp_sigma = fitted_params[element_idx][2] * hopping_scaling
    tpp_pi = 0
    tsp = fitted_params[element_idx][3] * hopping_scaling
    fermi_energy = 0
    
    
    #Filling the Hamiltonian and diagonalizing for all k-points
    #on-site energy part
    H[0,0] = -2*delta/3
    H[1,1] = delta/3
    H[2,2] = delta/3
    H[3,3] = -2*delta/3
    H[4,4] = delta/3
    H[5,5] = delta/3
    
    for i in range(len(k_array)):
        kx = k_array[i]
        for j in range(len(k_array)):
            ky = k_array[j]
            
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
            
            #diagonalizing the Hamiltonian and populating the evecs and evals arrays
            evals_temp,evecs_temp = n.linalg.eigh(H)
            evals_temp=evals_temp.real
            evals[i][j] = evals_temp
            evecs[i][j] = evecs_temp

    fermi_energy = [energy for (kx, ky), energy in n.ndenumerate(evals[:,:,0].real) if n.abs(energy - evals[kx,ky,1].real) < 0.01][0]
    if element_idx%2==1:
        fermi_energy = ( n.amax(evals[:,:,1].real) + n.amin(evals[:,:,2].real) )/2#Alakaline Earth between top of s bands and first flat p band
    evals = n.array(evals) - fermi_energy
    
    
    #creating an array for the points of symmetry in the kpath
    i = 0
    sympoints = []
    while i  <= len(k_array):
        #plt.axvline(weights[0][0][i], c = 'black')
        sympoints += [k_array[i]]
        i += 100
    
    
    k_array_path = n.linspace(0,100,234)
    sympoints = [k_array_path[0], k_array_path[100],k_array_path[133], k_array_path[233]]
    
    #Plotting the energy bands
    multiplier_res = 3
    #plt.figure(figsize=(1.87931317779*multiplier_res, 1*multiplier_res), dpi=200)
    for i in range(len(evals[0][0])):
        band_along_path = n.concatenate( (evals.T[i,len(k_array)//2,len(k_array)//2:-1].real, [evals.T[i,len(k_array)//2 + bruh,-1-bruh].real for bruh in range(33) ], [evals.T[i,len(k_array)//2 + 32 - int(bruh*32/100),-1-32 - int(bruh*(len(k_array)//2-32)/100)].real for bruh in range(101)]))
        #band_along_path = n.concatenate( (evals.T[i,len(k_array)//2,len(k_array)//2:-1].real, [evals.T[i,len(k_array)//2 + bruh,-1-bruh].real for bruh in range(33) ], [evals.T[i,len(k_array)//2 + 32 - int(bruh*32/100),-1-32 - int(bruh*(len(k_array)//2-32)/100)].real for bruh in range(101)])) - evals.T[2,0,0].real

        axes_hex_k[element_idx].plot(k_array_path, band_along_path, linewidth=2.5, color= "C0" if i <=1 else "C1")
    
    #Adding Element Name Text
    axes_hex_k[element_idx].text(0.5,0.5, "%s" % (element_string_array[element_idx]), fontsize = 15, transform = axes_hex_k[element_idx].transAxes, horizontalalignment='center')

    #Plot Axis Parameters
    axes_hex_k[element_idx].tick_params(axis = 'y', labelsize = 12)
    axes_hex_k[element_idx].set_xlim(min(k_array_path),max(k_array_path))
    axes_hex_k[element_idx].grid()
    axes_hex_k[element_idx].set_ylim(-10,10)
    if element_idx not in [3,7,11]:
        axes_hex_k[element_idx].set_xticks(n.sort(sympoints),kpath)
        axes_hex_k[element_idx].set_xticklabels([])
    else:
        axes_hex_k[element_idx].set_xticks(n.sort(sympoints),kpath)
        axes_hex_k[element_idx].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelsize = 12)




fig_hex_k.supylabel('E (eV)', fontsize = 17)
fig_hex_k.tight_layout()












#%% Plotting all bands of a chosen element


element_string_array = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"]
s_p_band_indices = [[1,4,2], [1,4,2], [4, 8,7], [4, 8,7], [4,15,13],[4,10,13], [4,15,5],[4,15,5], [4,15,5], [4,13,5], [4,15,5],[4,15,5]]

num_band_array = num_band_array = [14,14,17, 17, 19,19, 19,19, 19,14,19,21]
k_dens_array = [3200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]

for i in [-2]:
    
    kpoints = n.linspace(0,1,k_dens_array[i])
    
    band_data = n.loadtxt("other_elements//%s//+bweights" % element_string_array[i])
    band_data_SO = n.loadtxt("other_elements//%s//spin_orbit//+bweights" % element_string_array[i])
    
    for band_index in range(num_band_array[i]):
    #for band_index in s_p_band_indices[i]:
    #for band_index in [4,5,13, 6]:
        band_energy = [ band_data[band_index + j * num_band_array[i] ][1] for j in range(k_dens_array[i])]
        plt.plot(kpoints, band_energy)
 
    plt.grid()
    plt.ylim(-2,2)
    plt.show()
    plt.close()
    


#%% Hexagonal Real and Reciprocal Lattices, and making a chosen k-path


#### Plotting the Real and Reciprocal Spaces ~~~
multiplier_res = 5
dpi = 300
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(1.9*multiplier_res, 1*multiplier_res), dpi=dpi,  subplot_kw=dict(box_aspect=1))


#Basis Vectors
e1 = n.array([n.sqrt(3),0])
e2 = n.array([n.sqrt(3)/2,3/2])
g1 = n.array([1/n.sqrt(3),-1/3])
g2 = n.array([0,2/3])


#Calculating the point coordinates
lattice_size = 2
real_points = n.array([i*e1 + j*e2 for i in range(-lattice_size, lattice_size+1) for j in range(-lattice_size, lattice_size+1)])
reciprocal_points = n.array([i*g1 + j*g2 for i in range(-lattice_size, lattice_size+1) for j in range(-lattice_size, lattice_size+1)])

#Plotting the lattices
ax[0].scatter(real_points[:,0], real_points[:,1], s = 300)
ax[1].scatter(reciprocal_points[:,0], reciprocal_points[:,1], color = "red", s = 300)

display_size = [n.sqrt(3) * 1.2, 2/(3) *1.2]
#display_size = [n.sqrt(3) * 1.2, 2/(6) *1.2]
titles = ["Real Space", "Reciprocal Space"]
labels = ["a", r"$\frac{2\pi}{a}$"]
for i in [0,1]:
    ax[i].grid()
    ax[i].set_xlim(-display_size[i],display_size[i])
    ax[i].set_ylim(-display_size[i],display_size[i])  
    ax[i].set_title(titles[i], fontsize = 25)
    ax[i].set_xlabel(labels[i], fontsize = 30)
    ax[i].set_ylabel(labels[i], fontsize = 35,  rotation=0)
    ax[i].tick_params(axis = 'both', labelsize = 25)
    
#Plotting the First Brillouin Zone
ax[1].add_patch(mpl.patches.RegularPolygon((0,0), numVertices = 6, radius = n.linalg.norm(g2)/(2*n.cos(n.pi/6)), orientation = n.pi/6, alpha = 0.5, facecolor = "purple"))

ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
fig.tight_layout(pad = 4)

# ~~~


#### K-path Calculation

#Initial Parameters and Chosen Path
kdens = 51
sym_points = dict(G = dict(pos = [0,0], label = r"$\Gamma$"), K = dict(pos = [2/(3*n.sqrt(3)), 0], label = "K"), M = dict(pos = [1/(2*n.sqrt(3)), 1/6], label = "M"))
chosen_path = ("G", "K", "M", "G")
chosen_path_labels = [sym_points[point]["label"] for point in chosen_path]
chosen_path_pos = [sym_points[point]["pos"] for point in chosen_path]
kpath = [chosen_path_pos[0]]
sympoint_idx = [0]

#Calculating the k-points along the path
for segment_idx, (start_point, end_point) in enumerate(zip(chosen_path_pos, chosen_path_pos[1:])):
    segment_vec = n.subtract(end_point, start_point)
    segment_kpoints = [ n.add(i * segment_vec , start_point) for i in n.linspace(0, 1, int(n.linalg.norm(segment_vec)*kdens), endpoint = True )[1:] ]
    kpath += segment_kpoints
    sympoint_idx += [sympoint_idx[-1] + len(segment_kpoints) ]
kpath = n.array(kpath) 
num_kpoints = len(kpath)
plotting_kpath = n.linspace(0, 1, num_kpoints)
plotting_sympoints = [plotting_kpath[idx] for idx in sympoint_idx]
        
#Plotting the k_path
ax[1].plot(kpath[:,0], kpath[:,1], color = "orange", linewidth = 2)



#%% 2D Hex DFT Band Structures

multiplier_res = 5
fig_dft, axes_dft = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
#fig_dft_SO, axes_dft_SO = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=300)
axes_dft = axes_dft.flatten(order="F")
#axes_dft_SO = axes_dft_SO.flatten(order="F")

num_band_hex_array = 2*n.array([14, 14,17, 17,19,19, 19,19, 19,14,19,21])
#num_band_hex_SO_array = 2*n.array([28, 28, 34, 34, 38, 38, 38, 38, 38, 28, 38, 42])


kpoints = n.linspace(0,1, 474)
k_dens = 474
#for i, element in enumerate(element_string_array):
for i, element in enumerate(["Be"]):
    i = i+1
    #Importing Band Data
    band_data = n.loadtxt("other_elements//%s//+bweights_hex_all_points_short_dist" % element_string_array[i])
    #band_data_SO = n.loadtxt("other_elements//%s//+bweights_hex_so" % element_string_array[i])
    dft_bands = [ [ band_data[band_index + j * num_band_hex_array[i] ][1] for j in range(k_dens)] for band_index in range(num_band_hex_array[i])]
    #dft_bands_SO = [ [ band_data_SO[band_index + j * num_band_hex_SO_array[i] ][1] for j in range(k_dens)] for band_index in range(num_band_hex_SO_array[i])]
    
    #Adding the DFT bands to subplots:
    for band in dft_bands:
        axes_dft[i].plot(kpoints, band, color="black", linewidth = 1, zorder=2)
    #for band_SO in dft_bands_SO:
    #    axes_dft_SO[i].plot(kpoints, band_SO, color="black", linewidth = 1, zorder=2)
        
    #Adding element names
    axes_dft[i].text(0.5,0.1, "%s" % (element_string_array[i]), fontsize = 15, transform = axes_dft[i].transAxes, horizontalalignment='center', color="red")
    #axes_dft_SO[i].text(0.5,0.1, "%s" % (element_string_array[i]), fontsize = 15, transform = axes_dft_SO[i].transAxes, horizontalalignment='center', color="red")
 
    #Arranging tick Labels for subplots
    fig_dft.supylabel('E (eV)', fontsize = 13)
    #fig_dft_SO.supylabel('E (eV)', fontsize = 13)
    axes_dft[i].set_xlim([0,1])
    axes_dft[i].set_ylim(-5,8)
    axes_dft[i].set_ylim(-10,10)
    #axes_dft_SO[i].set_xlim([0,1])
    #axes_dft_SO[i].set_ylim(-3,8)
    axes_dft[i].grid()
    #axes_dft_SO[i].grid()

    if (i+1)%4 == 0:
        
        '''
        axes_dft[i].set_xticks([0, 0.5,1])
        axes_dft[i].set_xticklabels([r"$\Gamma$", "K", r"$\Gamma$"])
        '''
        axes_dft[i].set_xticks([0, 0.25,0.75, 1])
        axes_dft[i].set_xticklabels([r"$\Gamma$", "K", "M", r"$\Gamma$"])
        
        
        #axes_dft_SO[i].set_xticks([0, 0.5,1])
        #axes_dft_SO[i].set_xticklabels([r"$\Gamma$", "K", r"$\Gamma$"])


    else:
        
        '''
        axes_dft[i].set_xticks([0, 0.5,1])
        '''
        axes_dft[i].set_xticks([0, 0.42,0.785, 1])
        axes_dft[i].set_xticklabels([])
        #axes_dft_SO[i].set_xticks([0, 0.5,1])
        #axes_dft_SO[i].set_xticklabels([])



#Stopping a figure from being shown
#plt.close(fig_dft)
#plt.close(fig_dft_SO)





#%% 1D Phonon Dispersions (Phonopy)

#Creating the Figure
multiplier_res = 8
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=800)
ax = ax.flatten(order="F")

#Importing Parameters
num_band_points = 51*2
k_array = n.linspace(0, num_band_points, num_band_points)
unaligned_band_indices = [0,1]

#Looping over all elements
for element_idx, element in enumerate(element_string_array):
    
    #Importing the Phonon Quantum Espresso Data
    imported_data = n.loadtxt(f"other_elements//{element}//2q10k.txt", unpack=True)
    phonon_data = [ imported_data[1][ band_idx*num_band_points :  (band_idx+1)*num_band_points] for band_idx in range(len(imported_data[1])//num_band_points) ]
    sympoints = [ k_array[0], k_array[len(k_array)//2], k_array[-1] ]

    
    
    #Plotting the Phonon Bands
    for band_idx, band in enumerate(phonon_data):
        if band_idx not in unaligned_band_indices:
            ax[element_idx].plot(k_array, band*THz_to_meV, color="black", linewidth = 2)
    
    



    
    
    #Adding Element Name Text
    ax[element_idx].text(0.5,0.7, "%s" % (element_string_array[element_idx]), fontsize = 22, transform = ax[element_idx].transAxes, horizontalalignment='center')



    
    #Plot Axis Parameters
    ax[element_idx].tick_params(axis = 'y', labelsize = 20)
    ax[element_idx].set_xlim(k_array[0],k_array[-1])
    ax[element_idx].grid()
    #ax[element_idx].set_ylim(-10,10)
    if element_idx not in [3,7,11]:
        ax[element_idx].set_xticks(n.sort(sympoints),kpath)
        ax[element_idx].set_xticklabels([])
    else:
        ax[element_idx].set_xticks(n.sort(sympoints),kpath)
        ax[element_idx].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelsize = 20)
        


#Overall figure Settings
fig.supylabel('E (meV)', fontsize = 25)
fig.tight_layout()







#%% 2D Phonon Dispersions (Phonopy)

#Creating the Figure
multiplier_res = 8
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(1.4*multiplier_res, 1*multiplier_res), dpi=800)
ax = ax.flatten(order="F")

#Importing Parameters
num_band_points = 51*3
k_array = n.linspace(0, num_band_points, num_band_points)
unaligned_band_indices = []

#Looping over all elements
for element_idx, element in enumerate(element_string_array):
    
    #Importing the Phonon Quantum Espresso Data
    imported_data = n.loadtxt(f"other_elements//{element}//3q12k_hex.txt", unpack=True)
    phonon_data = [ imported_data[1][ band_idx*num_band_points :  (band_idx+1)*num_band_points] for band_idx in range(len(imported_data[1])//num_band_points) ]
    sympoints = [ k_array[0], k_array[len(k_array)//3], k_array[2*len(k_array)//3], k_array[-1] ]

    
    
    #Plotting the Phonon Bands
    for band_idx, band in enumerate(phonon_data):
        if band_idx not in unaligned_band_indices:
            ax[element_idx].plot(k_array, band*THz_to_meV, color="black", linewidth = 2)
    
    
    #Adding Element Name Text
    ax[element_idx].text(0.5,0.7, "%s" % (element_string_array[element_idx]), fontsize = 22, transform = ax[element_idx].transAxes, horizontalalignment='center')



    
    #Plot Axis Parameters
    ax[element_idx].tick_params(axis = 'y', labelsize = 20)
    ax[element_idx].set_xlim(k_array[0],k_array[-1])
    ax[element_idx].grid()
    #ax[element_idx].set_ylim(-10,10)
    if element_idx not in [3,7,11]:
        ax[element_idx].set_xticks(n.sort(sympoints),kpath_2D)
        ax[element_idx].set_xticklabels([])
    else:
        ax[element_idx].set_xticks(n.sort(sympoints),kpath_2D)
        ax[element_idx].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelsize = 20)
        


#Overall figure Settings
fig.supylabel('E (meV)', fontsize = 25)
fig.tight_layout()

