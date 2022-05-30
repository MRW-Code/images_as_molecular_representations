import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import tqdm
from joblib import Parallel, delayed
import os

class ImageGenerator():

    def __init__(self, dataset):
        print('GENERATING IMAGES')
        self.dataset = dataset
        if self.dataset == 'solubility':
            self.smiles = pd.read_csv('./data/solubility/raw_water_sol_set.csv').SMILES
            self.id = pd.read_csv('./data/solubility/raw_water_sol_set.csv').CompoundID
            self.gen_all_solubility_images()

        elif self.dataset == 'cocrystal':
            self.smilesA = pd.read_csv('./data/cocrystal/component1_smiles.csv').smiles
            self.smilesB = pd.read_csv('./data/cocrystal/component2_smiles.csv').smiles
            self.idA = pd.read_csv('./data/cocrystal/component1_smiles.csv').api
            self.idB = pd.read_csv('./data/cocrystal/component2_smiles.csv').api
            self.gen_all_cc_images()

    def smile_to_image(self, smile, compound_id):
        mol = Chem.MolFromSmiles(smile)
        draw = Draw.MolToFile(mol, (f'./data/{self.dataset}/images/{compound_id}.png'), size=[250, 250])

    def gen_all_solubility_images(self):
        os.makedirs(f'./data/{self.dataset}/images', exist_ok=True)
        smiles = self.smiles
        id = self.id
        # [self.smile_to_image(s, i) for s, i in tqdm.tqdm(zip(smiles, id))]
        Parallel(n_jobs=os.cpu_count())(delayed(self.smile_to_image)(s, i) for s, i in tqdm.tqdm(zip(smiles, id), ncols=80))

    def gen_all_cc_images(self):
        os.makedirs(f'./data/{self.dataset}/images', exist_ok=True)
        smiles = self.smilesA
        id = self.idA
        ## For one core
        # [self.smile_to_image(s, i) for s,i in tqdm.tqdm(zip(smiles, id))]

        # For multicore
        Parallel(n_jobs=os.cpu_count())(delayed(self.smile_to_image)(s, i) for s, i in tqdm.tqdm(zip(smiles, id), ncols=80))

        smiles = self.smilesB
        id = self.idB
        ## For one core
        # [self.smile_to_image(s, i) for s, i in tqdm.tqdm(zip(smiles, id))]

        # For multicore
        Parallel(n_jobs=os.cpu_count())(delayed(self.smile_to_image)(s, i) for s, i in tqdm.tqdm(zip(smiles, id), ncols=80))
