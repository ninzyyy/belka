import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


class FeatureEngineer:

    def __init__(self):
        pass

    def generate_ecfp(
        self, smiles, radius=2, nBits=1024, use_features=False, use_chirality=False
    ):

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.zeros((nBits,), dtype=np.uint8)

        feature_list = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=radius,
            nBits=nBits,
            useFeatures=use_features,
            useChirality=use_chirality,
        )

        return np.array(feature_list).astype(np.uint8)

    def smiles_to_maccs(smiles):

        molecule = Chem.MolFromSmiles(self, smiles)
        if molecule is None:
            print("\n Error: Unable to find molecule from the SMILES string.")
            return None
        else:
            maccs_key = MACCSkeys.GenMACCSKeys(molecule)
            return [int(bit) for bit in maccs_key.ToBitString()]

    def batch_ecfp(self, df, batch_size=500):

        num_batches = (len(df) + batch_size - 1) // batch_size
        all_fingerprints = []

        for i in range(num_batches):
            batch = df.iloc[i * batch_size : (i + 1) * batch_size]
            fingerprints = batch["molecule_smiles"].apply(self.generate_ecfp)
            all_fingerprints.append(np.vstack(fingerprints.values))
            print(f"Processed batch {i+1}/{num_batches}")

        ecfp_matrix = np.vstack(all_fingerprints)
        ecfp_matrix = ecfp_matrix.astype(np.uint8)
