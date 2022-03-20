import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from deepchem.feat import RDKitDescriptors, Mol2VecFingerprint ,\
    CircularFingerprint, PubChemFingerprint, MACCSKeysFingerprint
from spectrophore import spectrophore
from joblib import Parallel, delayed
from multiprocessing import Manager


from src.utils import args

class RepresentationGenerator:

    def __init__(self, dataset):
        print(f'GENERATING DESCRIPTORS FROM {args.input}')
        self.dataset = dataset
        if self.dataset == 'solubility':
            # self.raw_df = pd.read_csv('./data/solubility/raw_water_sol_set.csv').iloc[0:100, :]
            # self.smiles = self.raw_df.SMILES[0:100]
            # self.id = self.raw_df.CompoundID[0:100]
            self.raw_df = pd.read_csv('./data/solubility/raw_water_sol_set.csv')
            self.smiles = self.raw_df.SMILES
            self.id = self.raw_df.CompoundID
            self.ml_set = self.gen_ml_set_solubility()

        elif self.dataset == 'cocrystal':
            self.raw_df = pd.read_csv('./data/cocrystal/jan_raw_data.csv')
            self.smilesA = pd.read_csv('./data/cocrystal/component1_smiles.csv').smiles
            self.smilesB = pd.read_csv('./data/cocrystal/component2_smiles.csv').smiles
            self.namesA = pd.read_csv('./data/cocrystal/component1_smiles.csv').api
            self.namesB = pd.read_csv('./data/cocrystal/component2_smiles.csv').api
            self.ml_set = self.gen_ml_set_cocrystal()

    def mordred_descriptors_from_smiles(self, smile_list):
        calc = Calculator(descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(p) for p in smile_list]
        mols_updated = [mol for mol in mols if isinstance(mol, Chem.Mol)]
        return pd.DataFrame(calc.pandas(mols_updated, nproc=os.cpu_count()))

    def rdkit_descriptors_from_smiles(self, smile_list):
        get_desc = RDKitDescriptors()
        desc = get_desc.featurize(smile_list)
        return pd.DataFrame(desc)

    def mol2vec_from_smiles(self, smile_list):
        get_desc = Mol2VecFingerprint()
        desc = get_desc.featurize(smile_list)
        return pd.DataFrame(desc)

    def ecfp_from_smiles(self, smile_list):
        get_desc = CircularFingerprint()
        desc = get_desc.featurize(smile_list)
        return pd.DataFrame(desc)

    def pubchem_fp_from_smiles(self, smile_list):
        get_desc = PubChemFingerprint()
        desc = get_desc.featurize(smile_list)
        return pd.DataFrame([desc[x] for x in range(len(desc))])

    def maccs_from_smiles(self, smile_list):
        get_desc = MACCSKeysFingerprint()
        desc = get_desc.featurize(smile_list)
        return pd.DataFrame(desc)

    def spectrophore_from_smiles(self, smile_list, names):
        mols = [Chem.MolFromSmiles(p) for p in smile_list]
        mols = [mol for mol in mols if isinstance(mol, Chem.Mol)]
        mols = [Chem.AddHs(mol) for mol in tqdm(mols)]
        [AllChem.EmbedMolecule(mol, randomSeed=0) for mol in mols]
        calculator = spectrophore.SpectrophoreCalculator(normalization='none')
        names_new = names
        try:
            print('Running Normal Spectrophore Calc')
            desc = [calculator.calculate(mol) for mol in tqdm(mols)]
            return pd.DataFrame(desc), names_new
        except:
            print('Normal Spectrophore Broke - Running with Compromised Data Set!!!')
            desc = []
            names_new = []
            for idx, mol in tqdm(enumerate(mols)):
                try:
                    desc.append(calculator.calculate(mol))
                    names_new.append(names[idx])
                except:
                    print('nope')
            return pd.DataFrame(desc), names_new

    def get_raw_descriptors(self, smiles, names):
        if args.input == 'mordred_descriptor':
            desc_df = self.mordred_descriptors_from_smiles(smiles)
        elif args.input == 'rdkit_descriptor':
            desc_df = self.rdkit_descriptors_from_smiles(smiles)
        elif args.input == 'mol2vec':
            desc_df = self.mol2vec_from_smiles(smiles)
        elif args.input == 'ecfp':
            desc_df = self.ecfp_from_smiles(smiles)
        elif args.input == 'pubchem_fp':
            desc_df = self.pubchem_fp_from_smiles(smiles)
            desc_df.index = names
            desc_df = desc_df.dropna(axis=0)
            return desc_df
        elif args.input == 'maccs':
            desc_df = self.maccs_from_smiles(smiles)
        elif args.input == 'spectrophore':
            desc_df, names = self.spectrophore_from_smiles(smiles, names)
        else:
            AttributeError('Input type argument not implemented ')
        desc_df.index = names
        return desc_df

    def clean_descriptors(self, desc_df):
        df = desc_df.dropna(axis=1).select_dtypes(exclude=['object'])
        return df

    def get_clean_descriptors(self, smiles, names):
        raw_desc = self.get_raw_descriptors(smiles, names)
        clean_desc = self.clean_descriptors(raw_desc)
        return clean_desc

    def gen_ml_set_solubility(self):
        clean_desc = self.get_clean_descriptors(self.smiles, self.id)
        labels_df = self.raw_df.loc[:, ['CompoundID', 'logS']]
        df = pd.merge(labels_df, clean_desc, left_on='CompoundID', right_index=True)
        df = df.drop('CompoundID', axis=1)
        df = df.rename(columns={'logS' : 'label'})
        return df

    def gen_ml_set_cocrystal(self):
        cleanA = self.get_clean_descriptors(self.smilesA, self.namesA)
        cleanB = self.get_clean_descriptors(self.smilesB, self.namesB)
        df1 = pd.merge(self.raw_df, cleanA, left_on='Component1', right_index=True)
        df2 = pd.merge(df1, cleanB, left_on='Component2', right_index=True)
        df2 = df2.drop(['Component1', 'Component2'], axis=1)
        df2 = df2.rename(columns={'Outcome' : 'label'})
        return df2
