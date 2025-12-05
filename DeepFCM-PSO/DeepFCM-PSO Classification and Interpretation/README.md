This folder contains the core modules used to train and evaluate the DeepFCM-PSO framework.

• main.py
Central execution script. Handles data loading, image preprocessing, CNN training and prediction, integration of CNN outputs with clinical features, and k-fold training of the DeepFCM-PSO model. It also saves performance metrics, optimized weight matrices, and generates Grad-CAM and FCM visualizations.


• particle_functions.py
Implements the LévyFCM-PSO optimization algorithm, including particle initialization, weight-matrix evaluation, PSO updates, Lévy flight exploration, and output of the final optimized FCM weights and concept-evolution trajectories.


• plot_fcm.py
Creates the visualization of the learned FCM. It constructs a directed graph from the optimized weights, applies thresholding, and exports a high-resolution FCM diagram.


• compute_mean_values.py
Aggregates the weight matrices across all cross-validation folds, computing the element-wise mean and standard deviation and saving them into Excel files.


• deviations.py
Utility module for computing standard deviations of performance metrics across folds.


• gradcam_file.py
Provides the Grad-CAM implementation for generating heatmaps that highlight the CNN’s most influential image regions and producing overlay visualizations.
