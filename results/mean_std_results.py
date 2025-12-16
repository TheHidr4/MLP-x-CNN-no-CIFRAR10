import pandas as pd

df = pd.read_csv("results/results.csv")

# CNN com kernel 3
subset_1 = df[(df["model"] == "cnn") & (df["kernel"] == 3)]
mean_acc_1 = subset_1["test_accuracy"].mean()
std_acc_1 = subset_1["test_accuracy"].std()
print(f"CNN (kernel=3): {mean_acc_1:.4f} ± {std_acc_1:.4f}")

# CNN com kernel 5
subset_2 = df[(df["model"] == "cnn") & (df["kernel"] == 5)]
mean_acc_2 = subset_2["test_accuracy"].mean()
std_acc_2 = subset_2["test_accuracy"].std()
print(f"CNN (kernel=5): {mean_acc_2:.4f} ± {std_acc_2:.4f}")

# MLP
subset_3 = df[(df["model"] == "mlp")]
mean_acc_3 = subset_3["test_accuracy"].mean()
std_acc_3 = subset_3["test_accuracy"].std()
print(f"MLP': {mean_acc_3:.4f} ± {std_acc_3:.4f}")


#CNN (kernel=3): 0.7799 ± 0.0151
#CNN (kernel=5): 0.7876 ± 0.0079
#MLP: 0.5080 ± 0.0209