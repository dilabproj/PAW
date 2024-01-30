# Document
This documentation is about the use of framework, PAW, in the two public dataset BCI Competition IV 2a (BCIIV2a) and BCI Competition IV 2b (BCIIV2b). The framework is primarily for the inter-subject and intra-subject problem for EEG under multi-source-free domain adaptation (MSFDA) scenario. 

## File Discription
### Get Data
Use the downloaded raw data to get the experimental data, there are two files for both the datasets: 
* raw_to_saved_data.py: take the EEG data and labels from the raw data (i.e., .gdf or .mat).
* saved_data_to_sample.py: save the data apart from the subject and session information.

### Common Files in Training and Adaptation Phase
* load_data.py: data loader.
* model_EEGNet.py, model_eegtcnet.py: archetecture of base models.
* utils.py: some code like fixed random seed, get the dataloader .etc.
* config.py: all the adjustable argparses。

### Training Phase
* domain discriminator.py: the architecture of domain discriminator.
* training_phase_dd.py: the main file for training phase.


### Adaptation Phase
* augmentation.py：the data augmentation methods.
* loss.py：entropy loss.
* network.py：the network architectures beside the base models.
* adaptation.py: the main file for adaptaiton phase.

## Execute the Program
### Steps to Run Code
1. Build the enviroment: `conda env create –n new_env_name -f proposed_environment.yml`
2. Download the datasets [BCIIV2a](https://bnci-horizon-2020.eu/database/data-sets) and [BCIIV2b](https://www.bbci.de/competition/iv/index.html) from their websites.
3. Adjust the data path in raw_to_saved_data.py and saved_data_to_sample.py to get the experimental data.
4. Run training phase: change the data_path in the utils.py file and run `training_phase_dd.py`.
5. Run Adaptation phase: change the data_path in the utils.py file and run `adaptation.py`.

### Argparse
All the default hyper-parameters are as the same as the experiments, the common parameters to adjust are as belows:
* base_model: eegnet/eegtcnet.
* dataset: the dataset to run (i.e., 2a/2b).
* gpu_id.
* name: name recorded in wandb.
* not_use_wandb: won't record this run in wandb if set the flag.

# Acknowledgement
This research was supported in part by Ministry of Science and Technology Taiwan under grant no. 112-2634-F-A49 -005 and 110-2221-E-A49-078-MY3.