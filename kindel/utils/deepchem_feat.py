import numpy as np
from deepchem.feat import (
    MACCSKeysFingerprint,
    RDKitDescriptors,
    MordredDescriptors,
)
from rdkit import Chem
from tqdm import tqdm

from kindel.utils.fingerprint_feat import Featurizer

class DeepChemMACCS(Featurizer):
    """MACCS keys featurizer using DeepChem"""
    def __init__(self, **kwargs):
        self.featurizer = MACCSKeysFingerprint()
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        return self.featurizer.featurize([mol])[0]

class DeepChemRDKit(Featurizer):
    """RDKit descriptors featurizer using DeepChem"""
    def __init__(self, **kwargs):
        self.featurizer = RDKitDescriptors()
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        return self.featurizer.featurize([mol])[0]

class DeepChemMordred(Featurizer):
    """Mordred descriptors featurizer using DeepChem"""
    def __init__(self, **kwargs):
        self.featurizer = MordredDescriptors()
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        features = self.featurizer.featurize([mol])[0]
        # Replace any NaN values with 0
        return np.nan_to_num(features, 0)

def featurize_deepchem(df, smiles_col, featurizer_type="maccs", label_col=None):
    """Wrapper function for DeepChem featurizers"""
    featurizer_map = {
        "maccs": DeepChemMACCS(),
        "rdkit": DeepChemRDKit(),
        "mordred": DeepChemMordred(),
    }
    
    if featurizer_type not in featurizer_map:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}")
        
    featurizer = featurizer_map[featurizer_type]
    fps = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        fp = featurizer._featurize(mol)
        fps.append(fp)
        
    if label_col is not None:
        return np.array(fps), np.array(df[label_col])
    else:
        return np.array(fps)
