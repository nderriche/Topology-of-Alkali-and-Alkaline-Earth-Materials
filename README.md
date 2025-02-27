# Topology-of-Alkali-and-Alkaline-Earth-Materials

Contains the Python scripts used in the analysis and visualization of the charge-based topological properties of materials. The values associated with parameters and functions in these scripts are tailored for calculations associated with all alkali and alkaline earth 1D and 2D elemental systems specifically, which led to the publication of the following paper showing the emergence of interesting topological edge states in the lighter elements.   
**[Link to Paper](https://iopscience.iop.org/article/10.1088/1361-648X/ad6f64)**

Here is an overall breakdown of the Python scripts included in this repository. They all are structured using code cells containing clear comments explaining the purpose of each section and function, so it is useful to have a look at the contents of the files directly for more detail.

External Density Functional Theory (DFT) orbital-weighted band structure calculations for all of the alkali and alkaline earth elemental 1D chain and 2D hexagonal structures with varying lattice constants first need to be performed, as they are imported and analyzed in the following code. In my case, I have used the DFT software FPLO, and the Python-based framework Phonopy to do so.


## 1. [topological_analysis.py](topological_analysis.py)

The flow of the code cells in this hefty file goes as follows. First, the DFT band structures for all alkali and alkaline earth 1D chains are imported, and effective *s*-*p* sigma bonding and anti-bonding bands are isolated. This is a non-trivial process especially for the heavier elements, since mixing with states other than the *s* and *p* wavefunctions of the appropriate principal quantum number becomes significant. To tackle this problem, an algorithm was written to specifically extract the segments of bands whose majority of their orbital weight corresponds to these orbitals; these are also plotted by the script. 

Then, the Hamiltonian tight binding parameters (the hopping integrals and the *s*-*p* charge transfer energy) are determined by fitting bonding and antibonding band functions to these extracted band structures for all of the elements. These values allow one to determine whether the systems host topological edge states or not in their band gap (see the paper). The code prints out that information for each material and for the 1D chain and 2D hexagonal configurations individually, as they have distinct topological transition conditions.

Following that, a script determining the relaxed lattice constants of the 1D chain systems is presented. It imports DFT calculations performed with different input lattice constants and fits a quadratic function to the energy landscape minimum in order to determine equilibrium distances. These values were then used to perform relaxed DFT calculations which were employed in the steps above, both for the 1D and 2D configurations.

The script then contains code that defines and calculates the real space finite Hamiltonians (along with their eigenvalues and eigenstates) for 1D finite chains and finite strips of the 2D hexagonal structure for all alkali and alkaline earth elements. These algorithms use the fitted Hamiltonian parameters previously determined, and include user-defined parameters such as the size of those finite systems. The same is done for the k-space (momentum space) Hamiltonians in different labelled cells. Furthermore, separate code cells calculate and plot the density of states (DOS) for these two configurations. These, along with the cells that plot the 1D and 2D phonon dispersions calculated using Phonopy, were used for the figures presented in the supplementary material of the paper linked above.

Finally, there are also helper cells that are useful when dealing with certain lattice configurations like the hexagonal structure. An algorithm for plotting the real and reciprocal space lattices (unit cell and first Brillouin zone) along with their defining lattice and reciprocal lattices is provided, along with a way to visualize the path in momentum space chosen for a 2D system based on chosen high-symmetry points; this was useful when plotting the band structure of hexagonal systems. Furthermore, a cell to plot all of the DFT bands of a selected element for visualization purposes is also part of this file.





## 2. [paper_figures.py](paper_figures.py)

This is the Python script used to produce the figures in the published paper linked above, except for Figure 3 which is an explicatory conceptual diagram. It consists of importing the proper data, and using the right matplotlib methods and parameters to neatly group and present the relevant data; they are all plots with many subplots.



## 3. [timn_topology.py](timn_topology.py)

This Python file deals with the study of the topological properties of Ti<sub>4</sub>Mn 1D chains in the context of the interesting material Ti<sub>4</sub>MnBi<sub>2</sub>. This has not yet been published, but it represents a chapter in my Ph.D. thesis (**[link to published thesis](https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0447447)**). It is also heavily sectioned with clearly labelled code cells. It makes heavy use of SymPy to symbolically analyze model Hamiltonians related to this system, and allows one to investigate analytical expressions such as k-dependent eigenvalues and eigenvectors. There is also an analysis of finite chains in order to study the emergence of topological edge states. Moreover, there is an algorithm to study the bands and eigenstates of this system from first principles that involves the non-trivial detection of band crossings and their appropriate unscrambling to allow for k-dependent, band-specific orbital character analysis. This script plots all of those quantities in order to enable efficient data analysis and physical thinking.








