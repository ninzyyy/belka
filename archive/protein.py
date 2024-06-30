class ProteinManager:

    def __init__(self, fasta_path=None):
        self.fasta_content = ""
        if fasta_path:
            self.load_fasta(fasta_path)

    def load_fasta(self, fasta_path):
        try:
            with open(fasta_path, "r") as fasta_file:
                self.fasta_content = fasta_file.read()
        except FileNotFoundError:
            print(f"Error: The file {fasta_path} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_fasta(self):
        return self.fasta_content

    def get_sequence(self):
        return self.fasta_content.split("\n")[1]


if __name__ == "__main__":

    protein_1 = ProteinManager("data/protein_data/1ao6.fasta")
    print(f"\n{protein_1.get_sequence()}")

    protein_2 = ProteinManager("data/protein_data/3I28.fasta")
    print(f"\n{protein_2.get_sequence()}")

    protein_3 = ProteinManager("data/protein_data/7USK.fasta")
    print(f"\n{protein_3.get_sequence()}")
