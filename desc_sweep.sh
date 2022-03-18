#!/bin/bash


#python3 main.py -d cocrystal -i mordred_descriptor
#python3 main.py -d cocrystal -i rdkit_descriptor
#python3 main.py -d cocrystal -i mol2vec
#python3 main.py -d cocrystal -i ecfp
#python3 main.py -d cocrystal -i pubchem_fp
#python3 main.py -d cocrystal -i maccs
#python3 main.py -d cocrystal -i spectrophore

python3 main.py -d solubility -i mordred_descriptor
#python3 main.py -d solubility -i rdkit_descriptor
python3 main.py -d solubility -i mol2vec
python3 main.py -d solubility -i ecfp
python3 main.py -d solubility -i pubchem_fp
python3 main.py -d solubility -i maccs
python3 main.py -d solubility -i spectrophore