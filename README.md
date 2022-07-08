# images_as_molecular_representations
This repo accompanies the following publication "Images of chemical structures as molecular representations for deep learning ([https://doi.org/10.1557/s43578-022-00628-9]). All work was carried out on a Linux (Ubuntu) machine, as such I cannot be sure that the instructions will translate directly to other operating systems. Please note this code is not actively maintained however if I am made aware of any issues I will endevour to fix them. If this work has proved useful, then I ask that you consider citing the orgininal paper. Truely I hope this project aids in the adoption of AI for chemical applications. 

# Env Set Up
Project was built using python 3.6 and a virtual environment. Previously RDKit needed conda but given this is no longer 
the case, venv was used. The requirements install was correct at the time of making, but please check your pytorch install versions especially if you plan to use a gpu! The install for the pip version of RDKit is shown on the final line of these instructions, this should install automatically from `requirements.txt` but is included for reference. 

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
```
```bash
optional arguments:
  -i  # Specify the input type i.e the molecular representation you want
  -d , --dataset  # Choose between solubility and cocrystal dataset
  --no_augs # Dont do image augmentation
  --gpu_idx # Use gpu with given index (only works 0-5)
  --cpu # train on cpu bu pretty sure this does nothing at present
```

# How to generate images for custom modelling
If you are reading this so that you can use images of molecules for your own work, then hopefully the following pseudo code will be of use to you. These are the functions and imports needed to generate your own images of chemical structures from smiles codes. I believe RDKit has functionality to generate mols from a variety of unique identifiers so smiles could be swapped out for any one of these e.g InChI - see RDKit docs for their "MolFrom" functions. 
```
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
from joblib import Parallel, delayed

# Compound ID is intended for use in f-strings to name the images something meaningful.
def smile_to_image(smile, compound_id):
    save_path = f'./your_save_path_here/{compound_id}'
    mol = Chem.MolFromSmiles(smile)
    draw = Draw.MolToFile(mol, save_path, size=[250, 250])

# To use multiple CPU cores (which is advised for speeding it up)
def gen_all_solubility_images(smiles_list, compound_id_list):
    Parallel(n_jobs=os.cpu_count())(delayed(smile_to_image)(s, i) for s, i in tqdm.tqdm(zip(smiles_list, compound_id_list), ncols=80))
```

