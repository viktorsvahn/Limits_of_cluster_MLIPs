Data and analysis associated with DOI: 10.1021/acs.jctc.5c02043

# Limitations of cluster-trained MLIPs for liquid density and diffusivity
Viktor Svahn<sup>1</sup>, Ioan-Bogdan Magdău<sup>2</sup>, Samuel P. Niblett<sup>3,4</sup>, Gábor Csányi<sup>5</sup>, Kersti Hermansson<sup>1</sup>, Jolla Kullgren<sup>1</sup>

<sup>1</sup>Department of Chemistry-˚Angstr¨om, Uppsala University, Box 538, S-75231 Uppsala, Sweden.
<sup>2</sup>School of Natural and Environmental Sciences, Newcastle University, Newcastle Upon Tyne, NE1 7RU, UK.
<sup>3</sup>Yusuf Hamied Department of Chemistry, University of Cambridge, Lensfield Road, Cambridge, CB2 1EW, UK.
<sup>4</sup>Dassault Syst´emes BIOVIA, 334 Cambridge Science Park, Cambridge CB4 0WN, UK.
<sup>5</sup>Engineering Laboratory, University of Cambridge, Cambridge, CB2 1PZ UK.

**Abstract:**
Machine-learned interatomic potentials (MLIPs) based on quantum-mechanical data are often used as a means to combine the performance of classical force-fields with the accuracy of electronic structure methods. In this work, MLIPs based on the MACE architecture were trained, starting from two publicly available data sets: one based on periodic structures, and the other based on molecular cluster data. Two rather challenging liquid properties are in focus, density and diffusivity, here for the battery- relevant ethylene carbonate and ethyl methyl carbonate solvents and mixtures thereof. The focus of our study is the uncertainties in the generated MLIP models themselves (calculated for committees of models with different regression seeds and different training set sizes) and how these uncertainties reflect on the MD-simulated target properties. The second focus point is whether these uncertainties are small enough to allow the comparison and assessment of different density functional theory (DFT) functionals; here, only a small number of them are compared, but the workflow opens up for a more comprehensive assessment of many DFT functionals. We find that all our MACE-MLIPs, both cluster-trained ones and the periodic-structure-trained ones, produce stable 1 ns NPT trajectories, regardless of training set size and cluster composition, but the MACE-MLIPs trained on cluster data (labeled with the hybrid ωB97X-D3 functional) are found to be sensitive to both the random training seed and data selection, resulting in large uncertainties on the simulated diffusivity and density values.

This repository contains jupyter notebooks corresponding to the following sections in the article:
- *Periodic Training Data: Impact of the DFT Functional on Densities and Diffusivities* (`part1-dft_impact-periodic_data.ipynb`)
- Part 2:  *Property Variability from Models Fitted on Cluster Data* (`part2-property_variability-cluster_data.ipynb`)
- SD statistics: *Data Sets for MLIP Generation* (`SD_statistics.ipynb`)
