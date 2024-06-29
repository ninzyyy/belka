import pandas as pd
import polars as pl


class DatasetBalancer:

    def __init__(self):
        self.data = None

    def load_data(self, path):
        file_format = path[-3:].lower()
        print(f"\nLoading dataset from {file_format}...")

        try:
            if file_format == "csv":
                self.data = pl.read_csv(path)
                print(f"\nLoaded dataset of shape {self.data.shape}.")
                return self.data

            elif file_format == "parquet":
                self.data = pl.read_parquet(path)
                print(f"\nLoaded dataset of shape {self.data.shape}.")
                return self.data

            else:
                print(
                    "Error: Unsupported file format specified. Use 'csv' or 'parquet'."
                )
                return

        except FileNotFoundError:
            print(f"Error: The file {path} does not exist.")

        except Exception as e:
            print(f"An error occurred: {e}")

    def balance_dataset(
        self, save_format=None, filename="balanced_dataset", num_samples=None
    ):

        print("\nBalancing dataset...")
        if "binds" not in self.data.columns:
            print("Error: The dataset does not contain a 'binds' column.")
            return

        print(f"\nOriginal dataset shape: {self.data.shape}")
        print("Original distribution of target classes:")
        print(self.data["binds"].value_counts())

        # Split data into positive and negative samples
        one_df = self.data.filter(pl.col("binds") == 1)
        zero_df = self.data.filter(pl.col("binds") == 0)

        # Set the number of samples to be drawn from the negative class
        if num_samples and num_samples < zero_df.shape[0]:
            n_positive = num_samples
        else:
            n_positive = zero_df.shape[0]

        print(f"\nRandomly sampling {n_positive} samples from the negative class...")

        # Randomly sample from negative class
        sampled_zero = zero_df.sample(n=n_positive, with_replacement=False)

        # Combine the sampled negative class with the positive class
        if n_positive < zero_df.shape[0]:
            sampled_one = one_df.sample(n=n_positive, with_replacement=False)
            balanced_df = pl.concat([sampled_one, sampled_zero])
        else:
            balanced_df = pl.concat([one_df, sampled_zero])

        # Shuffle the final dataset
        balanced_df = balanced_df.sample(fraction=1.0, seed=42)

        print(f"\nBalanced dataset shape: {balanced_df.shape}")
        print("Balanced distribution of target classes:")
        print(balanced_df["binds"].value_counts())

        if save_format and filename:

            if save_format.lower() == "csv":
                balanced_df.write_csv(
                    f"data/processed_data/{filename}.csv", index=False
                )
                print(f"\nDataset saved as data/processed_data/{filename}.csv")

            elif save_format.lower() == "parquet":
                balanced_df.write_parquet(f"data/processed_data/{filename}.parquet")
                print(f"\nDataset saved as data/processed_data/{filename}.parquet")

            else:
                print(
                    "Error: Unsupported file format specified. Use 'csv' or 'parquet'."
                )

        return balanced_df
