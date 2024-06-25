import pandas as pd


class DatasetBalancer:

    def __init__(self):
        self.data = None

    def load_data(self, path, file_format):

        try:
            if file_format.lower() == "csv":
                print("\nLoading dataset from csv...")
                self.data = pd.read_csv(path)
                print(f"\nLoaded dataset of shape {self.data.shape}.")
                return self.data
            elif file_format.lower() == "parquet":
                print("\nLoading dataset from parquet...")
                self.data = pd.read_parquet(path)
                print(f"\nLoaded dataset of shape {self.data.shape}.")
                return self.data
            else:
                print(
                    "Error: Unsupported file format specified. Use 'csv' or 'parquet'."
                )
        except FileNotFoundError:
            print(f"Error: The file {path} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def balance_dataset(self, save_format=None, filename="balanced_dataset"):

        print("\nBalancing dataset...")

        if "binds" not in self.data.columns:
            print("Error: The dataset does not contain a 'binds' column.")
            return

        print(f"\nOriginal dataset shape: {self.data.shape}")
        print("Original distribution of target classes:")
        print(self.data["binds"].value_counts())

        zero_df = self.data[self.data["binds"] == 0]
        one_df = self.data[self.data["binds"] == 1]

        sampled_zero = zero_df.sample(n=len(one_df))
        balanced_df = pd.concat([sampled_zero, one_df])
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

        print(f"\nBalanced dataset shape: {balanced_df.shape}")
        print("Balanced distribution of target classes:")
        print(balanced_df["binds"].value_counts())

        if save_format and filename:

            if save_format.lower() == "csv":
                balanced_df.to_csv(f"data/processed_data/{filename}.csv", index=False)
                print(f"\nDataset saved as data/processed_data/{filename}.csv")

            elif save_format.lower() == "parquet":
                balanced_df.to_parquet(f"data/processed_data/{filename}.parquet")
                print(f"\nDataset saved as data/processed_data/{filename}.parquet")

            else:
                print(
                    "Error: Unsupported file format specified. Use 'csv' or 'parquet'."
                )

        return balanced_df


if __name__ == "__main__":

    balancer = DatasetBalancer()
    balancer.load_csv("data/original_data/train.csv")
    balancer.balance_dataset(save_format="csv", filename="TEST")
