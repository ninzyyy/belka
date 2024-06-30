import os
import polars as pl
from logger import Logger


class DatasetBalancer:

    def __init__(self):
        self.logger = Logger()
        self.data = None

    def load_data(self, path):

        # Extract file extension and convert to lowercase
        _, file_extension = os.path.splitext(path)
        file_format = file_extension.lower()[1:]  # Remove the period

        # Mapping of file formats to their respective loading functions
        loaders = {
            "csv": pl.read_csv,
            "parquet": pl.read_parquet,
        }

        try:
            # Select the file loader based on the file format
            if file_format in loaders:
                self.data = loaders[file_format](path)
                self.logger.timed_print(f"Loaded dataset of shape {self.data.shape}.")
                return self.data
            else:
                self.logger.timed_print(
                    f"Error: Unsupported file format '{file_format}'."
                )
        except Exception as e:
            self.logger.timed_print(f"Error loading dataset: {e}")

    def balance_dataset(
        self, save_format=None, filename="balanced_dataset", num_samples=None
    ):

        self.logger.timed_print("Balancing dataset...")

        if "binds" not in self.data.columns:
            self.logger.timed_print(
                "Error: The dataset does not contain a 'binds' column."
            )
            return

        self.logger.timed_print(f"Original dataset shape: {self.data.shape}")
        self.logger.timed_print("Original distribution of target classes:")
        self.logger.timed_print(self.data["binds"].value_counts())

        # Split data into positive and negative samples
        one_df = self.data.filter(pl.col("binds") == 1)
        zero_df = self.data.filter(pl.col("binds") == 0)

        # Set the number of samples to be drawn from each class
        n_positive = (
            min(num_samples, zero_df.shape[0]) if num_samples else zero_df.shape[0]
        )

        self.logger.timed_print(
            f"Randomly sampling {n_positive} samples from each class..."
        )

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

        self.logger.timed_print(f"Balanced dataset shape: {balanced_df.shape}")
        self.logger.timed_print("Balanced distribution of target classes:")
        self.logger.timed_print(balanced_df["binds"].value_counts())

        if save_format and filename:

            if save_format.lower() == "csv":
                balanced_df.write_csv(f"data/processed_data/{filename}.csv")
                self.logger.timed_print(
                    f"Dataset saved as data/processed_data/{filename}.csv"
                )

            elif save_format.lower() == "parquet":
                balanced_df.write_parquet(f"data/processed_data/{filename}.parquet")
                self.logger.timed_print(
                    f"Dataset saved as data/processed_data/{filename}.parquet"
                )

            else:
                self.logger.timed_print(
                    "Error: Unsupported file format specified. Use 'csv' or 'parquet'."
                )

        return balanced_df
