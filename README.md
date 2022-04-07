# images_as_molecular_representations
This work will hopefully accompany the following publication (insert doi here).

# Env Set Up
Project was built using python 3.6 and a virtual environment. Previously RDKit needed conda but given thi is no longer 
the case venv was used. 

```bash
# Install venv and dependencies
sudo apt-get install python3-venv
# Create working directory
mkdir {chosen working directory}
cd chosen_dir
# Make virtual environment
python3 -m venv env_name
# Activate virtual environment
source env_name/bin/activate
# install packages
pip install -r requirements.txt
# RDKit with pip for reference
pip install rdkit-pypi
```

# Running Instructions
Code can be run with the following command and flags. 

```bash
python3 main.py [options]
# OR
python3 sota.py [options]
```
```bash
python3 main.py 

optional arguments:
  -h, --help            show this help message and exit
  -i  # Specify the input type i.e the molecular representation you want
  -d , --dataset  # Choose between solubility and cocrystal dataset
  --no_augs # Dont do image augmentation
  --gpu_idx # Use gpu with given index (only works 0-5)
  --cpu # train on cpu bu pretty sure this does nothing at present
  --no_kfold  # use test/train/val splits. Not all implemented as kfold was main method used. 
  

```

