import numpy as np
import polars as pl
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

    def generate_ecfp_arr(self, df: pl.DataFrame, smiles_col):

        smiles_arr = df[smiles_col].to_numpy()

        # Vectorize the generate_ecfp function
        ecfp_arr = np.vectorize(self.generate_ecfp, signature="()->(n)")

        # Apply the vectorized function to the SMILES arr
        ecfp_arr = ecfp_arr(smiles_arr)

        return ecfp_arr
