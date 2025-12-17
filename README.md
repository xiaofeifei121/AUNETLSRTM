Self-Supervised Hessian Preconditioning for Least-Squares Reverse Time Migration Using an Attention-Based U-Net

This repository contains the official code for implementing Self-supervised Hessian Preconditioning for Least-Squares Reverse Time Migration (LSRTM) using an Attention-Based U-Net. The focus of this project is to develop and evaluate deep learning models for improving the accuracy and efficiency of seismic imaging and inversion, particularly in complex geological models such as the Marmousi model.

Directory Structure

./data
This folder contains all the datasets used for the project, including training, validation, and other relevant datasets necessary for model training and performance evaluation.

./deepinvhessian
This directory contains the core implementation of the deep learning model, which focuses on Hessian-based optimization techniques. It includes code for training, testing, and evaluating the model.

./notebooks
This folder holds Jupyter notebooks used for experimenting with various approaches, visualizing results, and analyzing model performance. It includes code for data exploration and initial model experimentation.

./unet
This directory contains the implementation of the U-Net model, a key component of the architecture used in this project. It includes model definitions, loss functions, and other utilities needed for training and fine-tuning the U-Net network.

Notebooks

marmousi_lstrm.ipynb
This notebook contains the main implementation of the Least Squares Reverse Time Migration (LSRTM) algorithm. It demonstrates how LSRTM is applied to the Marmousi model for seismic imaging and inversion tasks.

marmousi_lstrm_attenunet_end.ipynb
This notebook integrates the AttenuNet architecture with LSRTM for modeling attenuation effects in seismic data. It is used to finalize the training process, with a particular focus on attenuation compensation within the Marmousi model.

marmousi_lstrm_lbfgs.ipynb
This notebook presents the LSRTM implementation using L-BFGS optimization. It compares the standard LSRTM results with those obtained using the L-BFGS method, evaluating the impact on convergence speed and inversion accuracy.

marmousi_lstrm_unet.ipynb
This notebook integrates U-Net with LSRTM, demonstrating how the U-Net model enhances seismic image reconstruction during the inversion process. It uses the Marmousi model as a test case.

marmousi_model.ipynb
This notebook provides the implementation for constructing the Marmousi model, a widely-used synthetic benchmark for seismic inversion research. It includes the setup for the model, data generation, and data visualization techniques.

marmousi_model_born.ipynb
This notebook focuses on the Born approximation for seismic modeling, simulating seismic wave propagation within the Marmousi model. It is particularly useful for studying linearized inversion problems.

Acknowledgements

This work references the paper "Robust Full Waveform Inversion with Deep Hessian Deblurring" by Mustafa Alfarhan, Matteo Ravasi, Fuqiang Chen, and Tariq Alkhalifah.

