import numpy as np
import polars as pl

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from logger import Logger


class FeatureEngineer:

    def __init__(self):
        self.logger = Logger()
        self.ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def smiles_to_fp(self, smiles):

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros(self.ecfp_gen.GetLength() + 167 + 512, dtype=np.float32)

        ecfp = self.ecfp_gen.GetFingerprintAsNumPy(molecule)
        maccs = np.array(MACCSkeys.GenMACCSKeys(molecule), dtype=np.float32)
        avalon = np.array(GetAvalonFP(molecule, nBits=512), dtype=np.float32)

        return np.concatenate([ecfp, maccs, avalon])

    def generate_fp_batch(
        self,
        df: pl.DataFrame,
        smiles_col: str,
        batch_size: int,
        verbose: bool,
    ):

        self.logger.timed_print(f"Generating fingerprint features...")
        total, processed = len(df), 0

        for batch in df.iter_slices(batch_size):
            smiles_arr = batch[smiles_col].to_numpy()
            features = np.array([self.smiles_to_fp(smiles) for smiles in smiles_arr])

            processed += len(smiles_arr)
            if verbose:
                self.logger.timed_print(f"Processed {processed}/{total} molecules")

            yield features

    def generate_fp_arr(
        self,
        df: pl.DataFrame,
        smiles_col: str = "molecule_smiles",
        batch_size: int = 10000,
        verbose: bool = True,
    ):
        all_features = []
        for batch_features in self.generate_fp_batch(
            df=df, smiles_col=smiles_col, batch_size=batch_size, verbose=verbose
        ):
            all_features.append(batch_features)

        fp_arr = np.concatenate(all_features, axis=0)
        self.logger.timed_print(
            f"Generated fingerprint array of length {fp_arr.shape[1]} for {fp_arr.shape[0]} molecules."
        )
        return fp_arr
