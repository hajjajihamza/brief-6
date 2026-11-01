from sklearn.model_selection import train_test_split
from collections import Counter
from const import RANDOM_KEY

# Afficher la distribution des classes pour chaque split.
def analyze_split_distribution(train, val, test):
    splits = {
        "Train": train,
        "Validation": val,
        "Test": test,
    }

    for split_name, samples in splits.items():
        counts = Counter(sample["class"] for sample in samples)
        total = len(samples)

        print(f"\n--- {split_name} ({total} samples) ---")

        for classe, count in sorted(counts.items()):
            percentage = count / total * 100
            print(f"{classe:<15} : {count:>4} ({percentage:.2f}%)")


# Diviser le dataset en Train (70%), Validation (15%) et Test (15%).
def split_dataset(dataset):
    # Fusionner les samples de tous les groupes
    dataset = [
        sample
        for group in dataset
        for sample in group["samples"]
    ]

    print(f"Total samples : {len(dataset)}")

    # Split : 70% Train et 30% temporaire
    train, temp = train_test_split(
        dataset,
        test_size=0.30,
        random_state=RANDOM_KEY,
        stratify=[sample["class"] for sample in dataset]
    )

    # Split : 15% Validation et 15% Test
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_KEY,
        stratify=[sample["class"] for sample in temp]
    )

    # Afficher les résultats
    total = len(dataset)

    print(f"\nTrain : {len(train)} samples ({len(train) / total * 100:.2f}%)")
    print(f"Val   : {len(val)} samples ({len(val) / total * 100:.2f}%)")
    print(f"Test  : {len(test)} samples ({len(test) / total * 100:.2f}%)")
    print(f"Total : {len(train) + len(val) + len(test)} samples")

    analyze_split_distribution(train, val, test)

    return train, val, test