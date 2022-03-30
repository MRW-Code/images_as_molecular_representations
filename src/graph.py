from deepchem.feat import WeaveFeaturizer
from src.utils import args
import pandas as pd

class GraphGenerator():

    def __init__(self, dataset):
        print(f'GENERATING GRAPHS FROM {args.input}')
        self.dataset = dataset
        if self.dataset == 'solubility':
            # self.raw_df = pd.read_csv('./data/solubility/raw_water_sol_set.csv').iloc[0:20, :]
            # self.smiles = self.raw_df.SMILES[0:20]
            # self.id = self.raw_df.CompoundID[0:20]
            self.raw_df = pd.read_csv('./data/solubility/raw_water_sol_set.csv')
            self.smiles = self.raw_df.SMILES
            self.id = self.raw_df.CompoundID
            self.ml_set = self.get_sol_ml_set()

        elif self.dataset == 'cocrystal':
            self.raw_df = pd.read_csv('./data/cocrystal/jan_raw_data.csv')
            self.smilesA = pd.read_csv('./data/cocrystal/component1_smiles.csv').smiles
            self.smilesB = pd.read_csv('./data/cocrystal/component2_smiles.csv').smiles
            self.namesA = pd.read_csv('./data/cocrystal/component1_smiles.csv').api
            self.namesB = pd.read_csv('./data/cocrystal/component2_smiles.csv').api
            self.ml_set = self.get_cc_ml_set()

    def weave_graph_from_smiles(self, smile_list):
        maker = WeaveFeaturizer()
        graphs = [maker.featurize(smile)[0] for smile in smile_list]
        return graphs

    def get_graph(self, smiles, names):
        if args.input == 'weave_graph':
            graphs = self.weave_graph_from_smiles(smiles)
        return graphs

    def get_sol_ml_set(self):
        return self.get_graph(self.smiles, self.id)

    def get_cc_ml_set(self):
        graphsA = self.get_graph(self.smilesA, self.namesA)
        graphsB = self.get_graph(self.smilesB, self.namesB)
        return zip(graphsA, graphsB)