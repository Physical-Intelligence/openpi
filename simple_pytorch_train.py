import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _demo_sort_key(name: str) -> int:
    # demo_0, demo_1, ...
    return int(name.split("_")[1])


def load_hdf5_state_action(
    hdf5_path: Path,
    *,
    state_source: str = "ee_gripper",
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demos = sorted([k for k in data.keys() if k.startswith("demo_")], key=_demo_sort_key)

        for demo_name in demos:
            g = data[demo_name]
            if state_source == "ee_gripper":
                # 6 + 2 = 8 dims
                s = np.concatenate([g["obs"]["ee_states"][:], g["obs"]["gripper_states"][:]], axis=-1)
            elif state_source == "joint_gripper":
                # 7 + 2 = 9 dims
                s = np.concatenate([g["obs"]["joint_states"][:], g["obs"]["gripper_states"][:]], axis=-1)
            elif state_source == "robot_states":
                # 9 dims
                s = g["robot_states"][:]
            else:
                raise ValueError(f"Unsupported state_source: {state_source}")

            a = g["actions"][:]  # 7 dims
            if s.shape[0] != a.shape[0]:
                raise ValueError(f"{demo_name}: state length {s.shape[0]} != action length {a.shape[0]}")

            states.append(s.astype(np.float32))
            actions.append(a.astype(np.float32))

    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_count += bs
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser("Simple PyTorch BC training on LIBERO HDF5")
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path(
            "/home/ziyang10/openpi/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5"
        ),
    )
    parser.add_argument("--state-source", choices=["ee_gripper", "joint_gripper", "robot_states"], default="ee_gripper")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-path", type=Path, default=Path("simple_bc_model.pt"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.hdf5.exists():
        raise FileNotFoundError(f"HDF5 not found: {args.hdf5}")

    states_np, actions_np = load_hdf5_state_action(args.hdf5, state_source=args.state_source)
    n, in_dim = states_np.shape
    out_dim = actions_np.shape[1]

    print(f"Loaded dataset from {args.hdf5}")
    print(f"Total frames: {n}, state_dim: {in_dim}, action_dim: {out_dim}")

    X = torch.from_numpy(states_np)
    Y = torch.from_numpy(actions_np)

    perm = torch.randperm(n)
    train_size = int(n * args.train_ratio)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    x_mean = X[train_idx].mean(0, keepdim=True)
    x_std = X[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    X = (X - x_mean) / x_std

    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MLPPolicy(in_dim=in_dim, out_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

        train_loss = total_loss / max(total_count, 1)
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "state_source": args.state_source,
                "state_dim": in_dim,
                "action_dim": out_dim,
            }
            torch.save(ckpt, args.save_path)

    print(f"Done. Best val_loss={best_val:.6f}, checkpoint saved to {args.save_path.resolve()}")


if __name__ == "__main__":
    main()
