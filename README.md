# Batch-Correction-MALDI
Batch correction for Matrix-Assisted Laser Desorption/Ionization Mass Spectrometry Imaging

## Running the Autoencoder–Classifier Holdout Training

To train the model using the **autoencoder (AE) and classifier in an alternating update scheme**, you can use the script `train_ae_classifier_holdout.py`. In this setup, the autoencoder and the classifier are trained interchangeably, meaning that during training the model alternates between updating the autoencoder and updating the classifier. 


To simplify running this training pipeline, a helper shell script named `run_script_train_ae_classifier_holdout.sh` is provided. This script contains the necessary commands and parameters required to launch the training procedure. After adjusting the parameters, you can start the training process by executing the following command from the terminal:

```bash
./run_script_train_ae_classifier_holdout.sh