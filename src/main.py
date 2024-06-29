from data_balancer import DatasetBalancer


def main():
    balancer = DatasetBalancer()
    balancer.load_csv("data/original_data/train.csv")
    balancer.balance_dataset(save_format="csv", filename="TEST")


if __name__ == "__main__":
    main()
