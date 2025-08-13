# import csv
# import matplotlib.pyplot as plt

# def load_single_run(path):
#     days = []
#     blue_pop = []
#     red_pop = []
#     within_trust = []
#     between_trust = []

#     with open(path, newline='') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             days.append(int(row["day"]))
#             blue_pop.append(float(row["blue_pop"]))
#             red_pop.append(float(row["red_pop"]))
#             within_trust.append(float(row["within_trust"]))
#             between_trust.append(float(row["between_trust"]))

#     return days, blue_pop, red_pop, within_trust, between_trust


# def plot_single_run(path):
#     days, blue, red, within, between = load_single_run(path)
#     fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Population subplot
#     axs[0].plot(days, blue, label="Blue House", color="blue")
#     axs[0].plot(days, red, label="Red House", color="red")
#     axs[0].set_title("Population Over Time (Single Run)")
#     axs[0].set_ylabel("Population")
#     axs[0].legend()

#     # Trust subplot
#     axs[1].plot(days, within, label="Within-house Trust", color="green")
#     axs[1].plot(days, between, label="Between-house Trust", color="orange")
#     axs[1].set_title("Trust Evolution Over Time (Single Run)")
#     axs[1].set_xlabel("Day")
#     axs[1].set_ylabel("Trust Level")
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()


# def main():
#     path = "batch_results/run_003.csv"  # Change this path if needed
#     plot_single_run(path)

# if __name__ == "__main__":
#     main()


import csv
import matplotlib.pyplot as plt

def load_single_run(path):
    days = []
    blue_pop = []
    red_pop = []
    within_trust = []
    between_trust = []
    food_spawned = []
    food_picked = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row["day"]))
            blue_pop.append(float(row["blue_pop"]))
            red_pop.append(float(row["red_pop"]))
            within_trust.append(float(row["within_trust"]))
            between_trust.append(float(row["between_trust"]))
            food_spawned.append(float(row.get("food_spawned", 0)))  # fallback to 0 if missing
            food_picked.append(float(row.get("food_picked", 0)))

    return days, blue_pop, red_pop, within_trust, between_trust, food_spawned, food_picked


def plot_single_run(path):
    days, blue, red, within, between, spawned, picked = load_single_run(path)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Population subplot
    axs[0].plot(days, blue, label="Blue House", color="blue")
    axs[0].plot(days, red, label="Red House", color="red")
    axs[0].set_title("Population Over Time")
    axs[0].set_ylabel("Population")
    axs[0].legend()

    # Trust subplot
    axs[1].plot(days, within, label="Within-house Trust", color="green")
    axs[1].plot(days, between, label="Between-house Trust", color="orange")
    axs[1].set_title("Trust Evolution Over Time")
    axs[1].set_ylabel("Trust Level")
    axs[1].legend()

    # Food subplot
    axs[2].plot(days, spawned, label="Food Spawned", color="purple")
    axs[2].plot(days, picked, label="Food Picked", color="brown")
    axs[2].set_title("Food Spawn vs Consumption")
    axs[2].set_xlabel("Day")
    axs[2].set_ylabel("Count")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    path = "batch_results/run_009.csv"
    plot_single_run(path)

if __name__ == "__main__":
    main()
