import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


class FeatureEngineer:

    def __init__(self):
        pass

    def generate_ecfp(
        self, smiles, radius=2, nBits=2048, use_features=False, use_chirality=False
    ):

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros(nBits, dtype=np.uint8)

        feature_list = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=radius,
            nBits=nBits,
            useFeatures=use_features,
            useChirality=use_chirality,
        )

        return np.array(feature_list).astype(np.uint8)

    def generate_maccs(self, smiles):

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print("\n Error: Unable to find molecule from the SMILES string.")
            return None
        else:
            maccs_key = MACCSkeys.GenMACCSKeys(molecule)
            return [int(bit) for bit in maccs_key.ToBitString()]
