import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from logger import Logger


class FeatureEngineer:

    def __init__(self):
        self.logger = Logger()
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )

    def generate_ecfp(self, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros(self.fp_gen.GetLength(), dtype=np.uint8)
        return self.fp_gen.GetFingerprintAsNumPy(molecule)

    def generate_ecfp_arr(self, df: pl.DataFrame, smiles_col: str):
        self.logger.timed_print(f"Generating ECFP features...")
        smiles_arr = df[smiles_col].to_numpy()
        ecfp_arr = np.vectorize(self.generate_ecfp, signature="()->(n)")
        ecfp_arr = ecfp_arr(smiles_arr)
        self.logger.timed_print(
            f"ECFP fingerprints: {ecfp_arr.shape[1]} features for {ecfp_arr.shape[0]} molecules."
        )
        return ecfp_arr

    def generate_maccs(self, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros(167, dtype=np.uint8)
        return np.array(MACCSkeys.GenMACCSKeys(molecule))

    def generate_maccs_arr(self, df: pl.DataFrame, smiles_col: str):
        self.logger.timed_print(f"Generating MACCS features...")
        smiles_arr = df[smiles_col].to_numpy()
        maccs_arr = np.vectorize(self.generate_maccs, signature="()->(n)")
        maccs_arr = maccs_arr(smiles_arr)
        self.logger.timed_print(
            f"MACCS keys: {maccs_arr.shape[1]} features for {smiles_arr.shape[0]} molecules."
        )
        return maccs_arr
