This folder contains the core modules used to train and evaluate the DeepFCM-PSO framework.

• main.py
Central execution script. Handles data loading, image preprocessing, CNN training and prediction, integration of CNN outputs with clinical features, and k-fold training of the DeepFCM-PSO model. It also saves performance metrics, optimized weight matrices, and generates Grad-CAM and FCM visualizations.

#particle_functions.py
Implements the LévyFCM-PSO optimizer. It defines the particle structure, initializes weight matrices, runs PSO with Lévy flights, evaluates fitness, and returns the best FCM weights and concept evolution for each fold.


#plot_fcm.py
Creates a visual graph of the learned FCM. It loads the optimized weights, builds a NetworkX diagram, and saves an interpretable FCM graph image.


#compute_mean_values.py
Reads the weight matrices produced per fold, computes the cross-validated mean and standard deviation, and saves the aggregated results into Excel files.


#deviations.py
A small helper for calculating standard deviations of performance metrics across folds.


#gradcam_file.py
Provides a Grad-CAM implementation for producing heatmaps that highlight the CNN’s decision-relevant regions and overlaying them on the original images.
