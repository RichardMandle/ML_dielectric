# dielectric
Repo for work on dielectric anisotropy calculation with electronic structure methods, and inferrence via machine learning models.

### Notebooks
* small_working_network - this will take you from importing the excel file all the way to comparing the performance of multiple neural networks trained on different fingerprint models
* construct_manifest - this reads in calcualted dE data and compares with experiment and produces Figure 3 in the paper
* dataset_analysis - deep dive into the dataset; it generates Figure 2
* dataset_to_reference - this will take the data and generate .ris file(s) for citation. It'll also deail the very painful process of citing the ~ 400 patents.
* plot_network_performance - plots stuff to do with trained networks; generates Figure 5 in the paper

### Python Modules
* Chemnet - chemical / fingerprinting stuff, basically just pulls things from skfp.fingerprints
* Neurnet - neural network training helper functions
* fionet - file I/O opperations, I'm not sure we use any of it here
* plotnet - plotting functions
* prednet - predictor functions; I am fairly sure this is just old code and won't work with this version.
