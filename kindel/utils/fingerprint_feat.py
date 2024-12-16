import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator, Descriptors
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
#from deepchem.feat import ConvMolFeaturizer
from transformers import AutoTokenizer, AutoModel
import torch

"""Module copied over from DeepChem which is under a MIT license
https://github.com/deepchem/deepchem/blob/master/deepchem/feat/base_classes.py"""


class Featurizer(object):
    """
    Abstract class for calculating a set of features for a molecule.
    Child classes implement the _featurize method for calculating features
    for a single molecule.
    """

    def featurize(
        self, mols: list[Chem.Mol], use_tqdm: bool = True, asarray: bool = True
    ):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        use_tqdm: bool
            Whether a progress bar will be printed in the featurization.
        asarray: bool
            return featurized data as a numpy array (if False, return a list).
        """
        mols = [mols] if isinstance(mols, Chem.Mol) else mols
        features = []
        if use_tqdm:
            mols_iterable = tqdm(mols)
        else:
            mols_iterable = mols
        for mol in mols_iterable:
            if mol is not None:
                features.append(self._featurize(mol))
            else:
                features.append(np.array([]))

        if asarray:
            return np.asarray(features)
        return features

    def _featurize(self, mol: Chem.Mol):
        """
        Calculate features for a single molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        raise NotImplementedError("Featurizer is not defined.")

    def __call__(
        self, mols: list[Chem.Mol], use_tqdm: bool = True, asarray: bool = True
    ):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        use_tqdm: bool
            Whether a progress bar will be printed in the featurization.
        asarray: bool
            return featurized data as a numpy array (if False, return a list).
        """
        return self.featurize(mols, use_tqdm=use_tqdm, asarray=asarray)


class CircularFingerprint(Featurizer):
    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_bond_type: bool = True,
        as_numpy_array: bool = True,
        **kwargs,
    ):
        """
        Interface for MorganFingerprints

        Parameters
        ----------
        radius: int, (default 2)
            Radius of graph to consider
        n_bits: int, (default 2048)
            Number of bits in the fingerprint
        use_chirality: bool, (default False)
            Whether to consider chirality in fingerprint generation
        use_bond_type: bool, (default True)
            Whether to consider bond ordering in the fingerprint generation
        as_numpy_array: bool, (default True)
            Whether or not to return as numpy array

        """
        self.as_numpy_array = as_numpy_array
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=use_chirality,
            useBondTypes=use_bond_type,
        )
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        fingerprint = self.mfpgen.GetFingerprint(mol)
        if self.as_numpy_array:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, arr)
        else:
            arr = fingerprint
        return arr
    
'''class ConvMolFeaturizer(Featurizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.featurizer = deepchem.feat.ConvMolFeaturizer()

    def _featurize(self, mol: Chem.Mol):
        return self.featurizer.featurize(mol)'''

class CombinedFingerprint(Featurizer):
    """Combines Morgan fingerprints with key normalized RDKit descriptors"""
    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_bond_type: bool = True,
        as_numpy_array: bool = True,
        **kwargs,
    ):
        self.as_numpy_array = as_numpy_array
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=use_chirality,
            useBondTypes=use_bond_type,
        )
        self.scaler = StandardScaler()
        self.descriptors = [
            "MolWt",  # Molecular weight
            "MolLogP",  # ALogP
            "TPSA",  # Polar surface area
            "NumRotatableBonds",  # Rotatable bonds
            "NumHAcceptors",  # H-bond acceptors
            "NumHDonors",  # H-bond donors
            "FractionCSP3",  # Fraction SP3 carbons
            "NumAromaticRings",  # Aromatic rings
            "qed"  # Drug-likeness
        ]
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        # Get Morgan fingerprint
        fingerprint = self.mfpgen.GetFingerprint(mol)
        morgan_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint, morgan_arr)
        
        # Calculate descriptors
        desc_values = []
        for desc in self.descriptors:
            if desc == "qed":
                value = Chem.QED.default(mol)
            else:
                value = getattr(Descriptors, desc)(mol)
            desc_values.append(value)
        
        # Normalize descriptors
        desc_normalized = self.scaler.fit_transform(np.array(desc_values).reshape(1, -1))
        
        # Combine features
        return np.concatenate([morgan_arr, desc_normalized.flatten()])

class ChemBERTaFeaturizer(Featurizer):
    """ChemBERTa featurizer using the pretrained model"""
    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        max_length: int = 512,
        as_numpy_array: bool = True,
        **kwargs
    ):
        """
        Parameters
        ----------
        model_name: str
            Name of the pretrained model to use
        max_length: int
            Maximum sequence length for tokenization
        as_numpy_array: bool
            Whether to return numpy array or torch tensor
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.as_numpy_array = as_numpy_array
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        super().__init__(**kwargs)

    def _featurize(self, mol: Chem.Mol):
        """Get ChemBERTa embeddings for a molecule"""
        smiles = Chem.MolToSmiles(mol)
        inputs = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding as molecular representation
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        if self.as_numpy_array:
            return embeddings.cpu().numpy().flatten()
        return embeddings.cpu()
