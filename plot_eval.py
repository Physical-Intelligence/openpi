import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import pickle 

from tqdm import trange, tqdm

def graph_and_value(finetuned_csv,
                    base_csv):
    finetuned_df = pd.read_csv(finetuned_csv)
    base_df = pd.read_csv(base_csv)

    finetuned_sr = finetuned_df["success_rate"]
    base_sr = base_df["success_rate"]

    # Use the smaller dataset length for comparison
    less_n = min(len(base_sr), len(finetuned_sr))

    finetuned_avg = np.mean(finetuned_sr[:less_n])
    base_avg = np.mean(base_sr[:less_n])

    print(f"finetuned avg: {finetuned_avg}")
    print(f"base avg: {base_avg}")

    # Create a wider figure for a stretched x-axis
    plt.figure(figsize=(14, 5))

    x = np.arange(less_n)  # environment numbers 0..299
    width = 1.5            # make bars thicker (0.8 of the bin width)

    # Plot base first, then finetuned on top
    plt.bar(
        x,
        base_sr[:less_n],
        width,
        color="steelblue",
        alpha=0.6,
        label="Base",
        edgecolor=None
    )
    plt.bar(
        x,
        finetuned_sr[:less_n],
        width,
        color="darkorange",
        alpha=0.6,
        label="Finetuned",
        edgecolor=None
    )

    plt.xlabel("Environment Number")
    plt.ylabel("Success Rate")
    plt.title("Finetuned vs Base Success Rates per Environment (Overlaid Bars)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")

    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig("comparison_all.png", dpi=300)
    plt.close()

def train_stats(sched_pkl="./test_logs_legacy/scheduler_00000010.pkl"):
    with open(sched_pkl, "rb") as f:
        scheduler = pickle.load(f)
    
    archive = scheduler.archive

    all_traj = archive.data("solution")
    print(len(all_traj))

def extract_unseen(finetune_sched_pkl="./test_logs_legacy/scheduler_00000010.pkl",
                   sched_pkl="./test_logs/scheduler_00000105.pkl"):
    with open(finetune_sched_pkl, "rb") as f:
        training_scheduler = pickle.load(f)

    with open(sched_pkl, "rb") as f:
        eval_scheduler = pickle.load(f)

    training_params = training_scheduler.archive.data("solution")
    eval_params = eval_scheduler.archive.data("solution")

    sol_id_to_keep = []
    for sol_id, sol in tqdm(enumerate(training_params)):
        # print(sol)
        # print(eval_params[sol_id])
        if not np.array_equal(sol, eval_params[sol_id]):
            sol_id_to_keep.append(sol_id)
    
    return sol_id_to_keep

def graph_and_value_unseen(finetuned_csv,
                           base_csv,
                           to_keep_sol_ids):
    finetuned_df = pd.read_csv(finetuned_csv)
    base_df = pd.read_csv(base_csv)

    old_finetuned_sr = np.array(finetuned_df["success_rate"])
    old_base_sr = np.array(base_df["success_rate"])

    finetuned_sr = old_finetuned_sr[to_keep_sol_ids]
    base_sr = old_base_sr[to_keep_sol_ids]

    # Use the smaller dataset length for comparison
    less_n = min(len(base_sr), len(finetuned_sr))

    finetuned_avg = np.mean(finetuned_sr[:less_n])
    base_avg = np.mean(base_sr[:less_n])

    print(f"finetuned avg: {finetuned_avg}")
    print(f"base avg: {base_avg}")

    # Create a wider figure for a stretched x-axis
    plt.figure(figsize=(14, 5))

    x = np.arange(less_n)  # environment numbers 0..299
    width = 1.5            # make bars thicker (0.8 of the bin width)

    # Plot base first, then finetuned on top
    plt.bar(
        x,
        base_sr[:less_n],
        width,
        color="steelblue",
        alpha=0.6,
        label="Base",
        edgecolor=None
    )
    plt.bar(
        x,
        finetuned_sr[:less_n],
        width,
        color="darkorange",
        alpha=0.6,
        label="Finetuned",
        edgecolor=None
    )

    plt.xlabel("Environment Number")
    plt.ylabel("Success Rate")
    plt.title("Finetuned vs Base Success Rates per Environment (Overlaid Bars)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")

    # Adjust spacing and save
    plt.tight_layout()
    plt.savefig("comparison_unseen.png", dpi=300)
    plt.close()

# graph_and_value_unseen("./qd/eval_logs_finetuned/summary.csv",
#                         "./qd/eval_logs_base/summary.csv",
#                         sol_id_to_keep)

graph_and_value("./eval_base_libero_logs/summary.csv",
                "./eval_finetuned_libero_logs/summary.csv")