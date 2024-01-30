# Document
This documentation is about using the framework, PAW, in the two public datasets BCI Competition IV 2a (BCIIV2a) and BCI Competition IV 2b (BCIIV2b). The framework is primarily for the inter-subject and intra-subject problem for EEG under a multi-source-free domain adaptation (MSFDA) scenario. 

## File Description
### Get Data
Use the downloaded raw data to get the experimental data. There are two files for both datasets: 
* raw_to_saved_data.py: Take the EEG data and labels from the raw data (i.e., .gdf or .mat).
* saved_data_to_sample.py: save the data apart from the subject and session information.

### Common Files in Training and Adaptation Phase
* load_data.py: The code for data loader.
* model_EEGNet.py, model_eegtcnet.py: The architecture of base models.
* utils.py: Some code likes to fix random seeds, get the data loader, .etc.
* config.py: All the adjustable argparses。

### Training Phase
* domain discriminator.py: The architecture of domain discriminator.
* training_phase_dd.py: The main file for the training phase.

### Adaptation Phase
* augmentation.py: The data augmentation methods.
* loss.py: The code of entropy loss.
* network.py: The network architectures besides the base models.
* adaptation.py: The main file for the adaptation phase.

## Execute the Program
### Steps to Run Code
1. Build the environment: `conda env create –n new_env_name -f proposed_environment.yml`
2. Download the datasets [BCIIV2a](https://bnci-horizon-2020.eu/database/data-sets) and [BCIIV2b](https://www.bbci.de/competition/iv/index.html) from their websites.
3. Adjust the data path in raw_to_saved_data.py and saved_data_to_sample.py to get the experimental data.
4. Run the training phase: Change the data_path in the utils.py file and run `training_phase_dd.py`.
5. Run the Adaptation phase: Change the data_path in the utils.py file and run `adaptation.py`.

### Argparse
All the default hyper-parameters are the same as the experiments, the common parameters to adjust are as below:
* base_model: eegnet/eegtcnet.
* dataset: The dataset to run (i.e., 2a/2b).
* gpu_id
* name: The name recorded in the wandb.
* not_use_wandb: Setting the flag would not record this run in the wandb.

# Acknowledgement
This research was supported in part by Ministry of Science and Technology Taiwan under grant no. 112-2634-F-A49 -005 and 110-2221-E-A49-078-MY3.