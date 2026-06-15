import os
import csv
import argparse

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedLoss import top_attention_loss
from GUNTAM.IO.PrepareTensor import sample_positive_pairs_from_particle_ids


def get_hard_negative_fraction(epoch):
    
    # if epoch < 3:
    #     return 0.0
    # if epoch < 10:
    #     return 0.1
    # if epoch < 20:
    #     return 0.2
    # if epoch < 30:
    #     return 0.3
    # if epoch < 40:
    #     return 0.4
    # return 0.5
    return 0.0


def get_phases():
    return [
        (1, 3, 0.0, "#e0e0e0"),
        (4, 10, 0.1, "#1f78b4"),
        (11, 20, 0.2, "#33a02c"),
        (21, 30, 0.3, "#ff7f00"),
        (31, 40, 0.4, "#e31a1c"),
        (41, 50, 0.5, "#6a3d9a"),
    ]


def add_phase_background(max_epoch):
    for start_epoch, end_epoch, frac, color in get_phases():
        if start_epoch <= max_epoch:
            plt.axvspan(
                start_epoch - 0.5,
                min(end_epoch, max_epoch) + 0.5,
                color=color,
                alpha=0.22,
                label=f"hard neg frac = {frac}",
            )


def split_file_indices(num_files, validation_fraction):
    if num_files < 2:
        raise RuntimeError("Need at least 2 tensor files to create a train/validation split.")

    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    split_index = int(num_files * (1.0 - validation_fraction))

    if split_index <= 0:
        split_index = 1

    if split_index >= num_files:
        split_index = num_files - 1

    train_file_indices = list(range(0, split_index))
    validation_file_indices = list(range(split_index, num_files))

    return train_file_indices, validation_file_indices


def flatten_particle_ids(particle_ids):
    if particle_ids.dim() == 3:
        particle_ids = particle_ids[0, :, 0]
    elif particle_ids.dim() == 2:
        if particle_ids.shape[-1] == 1:
            particle_ids = particle_ids[:, 0]
        else:
            particle_ids = particle_ids[0]
    else:
        particle_ids = particle_ids.view(-1)

    return particle_ids.long()


def flatten_mask(padding_mask):
    if padding_mask.dim() == 2:
        padding_mask = padding_mask[0]
    else:
        padding_mask = padding_mask.view(-1)

    return padding_mask.bool()


def tensor_stats(name, tensor):
    if tensor is None or tensor.numel() == 0:
        print(name, "empty")
        return

    tensor = tensor.detach().float()
    print(
        name,
        "numel=",
        tensor.numel(),
        "min=",
        tensor.min().item(),
        "max=",
        tensor.max().item(),
        "mean=",
        tensor.mean().item(),
    )


def get_debug_pair_tensors(loss_debug, possible_left_names, possible_right_names):
    left = None
    right = None

    for name in possible_left_names:
        if name in loss_debug:
            left = loss_debug[name]
            break

    for name in possible_right_names:
        if name in loss_debug:
            right = loss_debug[name]
            break

    return left, right


def sample_debug_random_negatives(particle_ids, valid_mask, num_pairs, device):
    valid_indices = torch.where(valid_mask & (particle_ids >= 0))[0]

    if valid_indices.numel() < 2:
        valid_indices = torch.where(particle_ids >= 0)[0]

    if valid_indices.numel() < 2:
        return None, None

    sampled_left = []
    sampled_right = []
    max_attempts = max(1000, num_pairs * 200)

    for _ in range(max_attempts):
        if len(sampled_left) >= num_pairs:
            break

        random_positions = torch.randint(
            low=0,
            high=valid_indices.numel(),
            size=(2,),
            device=device,
        )
        left = valid_indices[random_positions[0]]
        right = valid_indices[random_positions[1]]

        if left.item() == right.item():
            continue

        if particle_ids[left].item() == particle_ids[right].item():
            continue

        sampled_left.append(left)
        sampled_right.append(right)

    if len(sampled_left) == 0:
        return None, None

    return torch.stack(sampled_left), torch.stack(sampled_right)


def print_pair_truth_debug(
    stage,
    epoch,
    file_idx,
    event_idx,
    pairs1,
    pairs2,
    target,
    particle_ids,
    padding_mask,
    attention_map,
    loss_debug,
):
    particle_ids = flatten_particle_ids(particle_ids).to(attention_map.device)
    valid_mask = flatten_mask(padding_mask).to(attention_map.device)
    pairs1 = pairs1.long()
    pairs2 = pairs2.long()
    target = target.detach()

    positive_left_ids = particle_ids[pairs1]
    positive_right_ids = particle_ids[pairs2]
    positive_same = positive_left_ids == positive_right_ids

    print()
    print("=" * 80)
    print("PAIR TRUTH DEBUG")
    print("stage:", stage)
    print("epoch:", epoch + 1)
    print("file_idx:", file_idx)
    print("event_idx:", event_idx)
    print("attention_map shape:", tuple(attention_map.shape))
    print("particle_ids shape:", tuple(particle_ids.shape))
    print("padding_mask shape:", tuple(valid_mask.shape))
    print("padding_mask true count:", valid_mask.sum().item())
    print("padding_mask false count:", (~valid_mask).sum().item())
    print("particle_id min:", particle_ids.min().item())
    print("particle_id max:", particle_ids.max().item())
    print("unique particle ids:", torch.unique(particle_ids).numel())
    print("target unique values:", torch.unique(target.detach().cpu()))
    print("positive pair count:", pairs1.numel())
    print("positive same-particle count:", positive_same.sum().item())
    print("positive different-particle count:", (~positive_same).sum().item())

    preview_count = min(10, pairs1.numel())
    print("first positive pairs:")
    for k in range(preview_count):
        print(
            int(pairs1[k].item()),
            int(pairs2[k].item()),
            int(positive_left_ids[k].item()),
            int(positive_right_ids[k].item()),
            bool(positive_same[k].item()),
        )

    if "positive_scores" in loss_debug:
        direct_positive_scores = attention_map[pairs1, pairs2]
        score_difference = (
            direct_positive_scores.detach() - loss_debug["positive_scores"].detach()
        ).abs()
        tensor_stats("positive_scores", loss_debug["positive_scores"])
        tensor_stats("direct_positive_scores", direct_positive_scores)
        tensor_stats("abs difference positive_scores vs direct", score_difference)

    if "random_negative_scores" in loss_debug:
        tensor_stats("random_negative_scores", loss_debug["random_negative_scores"])

    print("loss_debug keys:", sorted(list(loss_debug.keys())))

    loss_neg_left, loss_neg_right = get_debug_pair_tensors(
        loss_debug,
        [
            "random_negative_pairs1",
            "random_neg_pairs1",
            "random_negative_i",
            "random_neg_i",
            "negative_pairs1",
            "negative_i",
            "neg_i",
        ],
        [
            "random_negative_pairs2",
            "random_neg_pairs2",
            "random_negative_j",
            "random_neg_j",
            "negative_pairs2",
            "negative_j",
            "neg_j",
        ],
    )

    if loss_neg_left is not None and loss_neg_right is not None:
        loss_neg_left = loss_neg_left.long().to(attention_map.device)
        loss_neg_right = loss_neg_right.long().to(attention_map.device)
        loss_neg_left_ids = particle_ids[loss_neg_left]
        loss_neg_right_ids = particle_ids[loss_neg_right]
        loss_neg_different = loss_neg_left_ids != loss_neg_right_ids

        print("loss random negative pair count:", loss_neg_left.numel())
        print("loss random negative different-particle count:", loss_neg_different.sum().item())
        print("loss random negative same-particle count:", (~loss_neg_different).sum().item())
        print("first loss random negative pairs:")

        preview_count = min(10, loss_neg_left.numel())
        for k in range(preview_count):
            print(
                int(loss_neg_left[k].item()),
                int(loss_neg_right[k].item()),
                int(loss_neg_left_ids[k].item()),
                int(loss_neg_right_ids[k].item()),
                bool(loss_neg_different[k].item()),
            )
    else:
        print("loss random negative pair indices: not present in loss_debug")

    debug_neg_left, debug_neg_right = sample_debug_random_negatives(
        particle_ids=particle_ids,
        valid_mask=valid_mask,
        num_pairs=10,
        device=attention_map.device,
    )

    if debug_neg_left is not None and debug_neg_right is not None:
        debug_neg_left_ids = particle_ids[debug_neg_left]
        debug_neg_right_ids = particle_ids[debug_neg_right]
        debug_neg_scores = attention_map[debug_neg_left, debug_neg_right]
        debug_neg_sigmoid = torch.sigmoid(debug_neg_scores)

        print("independent debug negative pairs:")
        for k in range(debug_neg_left.numel()):
            print(
                int(debug_neg_left[k].item()),
                int(debug_neg_right[k].item()),
                int(debug_neg_left_ids[k].item()),
                int(debug_neg_right_ids[k].item()),
                float(debug_neg_sigmoid[k].item()),
            )

        tensor_stats("independent_debug_negative_scores", debug_neg_scores)
        tensor_stats("independent_debug_negative_sigmoid", debug_neg_sigmoid)
    else:
        print("independent debug negative pairs: none sampled")

    print("=" * 80)
    print()


def save_metrics_csv(epoch_history, checkpoint_dir):
    metrics_csv_path = os.path.join(checkpoint_dir, "training_gap_metrics.csv")

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "hard_negative_fraction",
                "average_loss",
                "mean_pos_sigmoid",
                "mean_random_neg_sigmoid",
                "gap_pos_random_neg",
                "successful_events",
                "skipped_events",
            ],
        )
        writer.writeheader()
        writer.writerows(epoch_history)

    print("Saved metrics CSV:", metrics_csv_path)


def save_validation_metrics_csv(validation_history, checkpoint_dir):
    metrics_csv_path = os.path.join(checkpoint_dir, "validation_gap_metrics.csv")

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "hard_negative_fraction",
                "validation_average_loss",
                "validation_mean_pos_sigmoid",
                "validation_mean_random_neg_sigmoid",
                "validation_gap_pos_random_neg",
                "evaluated_events",
                "skipped_events",
            ],
        )
        writer.writeheader()
        writer.writerows(validation_history)

    print("Saved validation metrics CSV:", metrics_csv_path)


def save_train_eval_metrics_csv(train_eval_history, checkpoint_dir):
    metrics_csv_path = os.path.join(checkpoint_dir, "train_eval_gap_metrics.csv")

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "hard_negative_fraction",
                "train_eval_average_loss",
                "train_eval_mean_pos_sigmoid",
                "train_eval_mean_random_neg_sigmoid",
                "train_eval_gap_pos_random_neg",
                "evaluated_events",
                "skipped_events",
            ],
        )
        writer.writeheader()
        writer.writerows(train_eval_history)

    print("Saved train-set evaluation metrics CSV:", metrics_csv_path)


def save_gap_plot(epoch_history, attention_plot_dir):
    if len(epoch_history) == 0:
        return

    epochs_plot = [row["epoch"] for row in epoch_history]
    gaps_plot = [row["gap_pos_random_neg"] for row in epoch_history]
    pos_plot = [row["mean_pos_sigmoid"] for row in epoch_history]
    rand_neg_plot = [row["mean_random_neg_sigmoid"] for row in epoch_history]

    max_epoch = max(epochs_plot)

    plt.figure(figsize=(12, 6))
    add_phase_background(max_epoch)
    plt.plot(
        epochs_plot,
        gaps_plot,
        marker="o",
        linewidth=2,
        color="black",
        label="gap: pos - random neg",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean sigmoid positive - mean sigmoid random negative")
    plt.title("Training gap between positive pairs and random negative pairs")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    gap_plot_path = os.path.join(attention_plot_dir, "gap_pos_vs_random_neg.png")
    plt.savefig(gap_plot_path, dpi=200)
    plt.close()
    print("Saved gap plot:", gap_plot_path)

    plt.figure(figsize=(12, 6))
    add_phase_background(max_epoch)
    plt.plot(
        epochs_plot,
        pos_plot,
        marker="o",
        linewidth=2,
        color="black",
        label="mean sigmoid positives",
    )
    plt.plot(
        epochs_plot,
        rand_neg_plot,
        marker="s",
        linewidth=2,
        color="red",
        label="mean sigmoid random negatives",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean sigmoid score")
    plt.title("Training positive vs random negative sigmoid scores")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    means_plot_path = os.path.join(attention_plot_dir, "mean_sigmoid_pos_vs_random_neg.png")
    plt.savefig(means_plot_path, dpi=200)
    plt.close()
    print("Saved sigmoid means plot:", means_plot_path)


def save_loss_plot(epoch_history, attention_plot_dir):
    if len(epoch_history) == 0:
        return

    epochs_plot = [row["epoch"] for row in epoch_history]
    losses_plot = [row["average_loss"] for row in epoch_history]

    max_epoch = max(epochs_plot)

    plt.figure(figsize=(12, 6))
    add_phase_background(max_epoch)
    plt.plot(
        epochs_plot,
        losses_plot,
        marker="o",
        linewidth=2,
        color="black",
        label="average loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training average loss with progressive introduction of hard negatives")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    loss_plot_path = os.path.join(
        attention_plot_dir,
        "average_loss_progressive_introduction_of_hard_negatives.png",
    )
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print("Saved loss plot:", loss_plot_path)


def save_validation_gap_plot(validation_history, attention_plot_dir):
    if len(validation_history) == 0:
        return

    epochs_plot = [row["epoch"] for row in validation_history]
    gaps_plot = [row["validation_gap_pos_random_neg"] for row in validation_history]
    pos_plot = [row["validation_mean_pos_sigmoid"] for row in validation_history]
    rand_neg_plot = [row["validation_mean_random_neg_sigmoid"] for row in validation_history]

    max_epoch = max(epochs_plot)

    plt.figure(figsize=(12, 6))
    add_phase_background(max_epoch)
    plt.plot(
        epochs_plot,
        gaps_plot,
        marker="o",
        linewidth=2,
        color="black",
        label="validation gap: pos - random neg",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean sigmoid positive - mean sigmoid random negative")
    plt.title("Validation gap on unseen events")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    gap_plot_path = os.path.join(attention_plot_dir, "validation_gap_pos_vs_random_neg.png")
    plt.savefig(gap_plot_path, dpi=200)
    plt.close()
    print("Saved validation gap plot:", gap_plot_path)

    plt.figure(figsize=(12, 6))
    add_phase_background(max_epoch)
    plt.plot(
        epochs_plot,
        pos_plot,
        marker="o",
        linewidth=2,
        color="black",
        label="validation mean sigmoid positives",
    )
    plt.plot(
        epochs_plot,
        rand_neg_plot,
        marker="s",
        linewidth=2,
        color="red",
        label="validation mean sigmoid random negatives",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean sigmoid score")
    plt.title("Validation positive vs random negative sigmoid scores on unseen events")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    means_plot_path = os.path.join(
        attention_plot_dir,
        "validation_mean_sigmoid_pos_vs_random_neg.png",
    )
    plt.savefig(means_plot_path, dpi=200)
    plt.close()
    print("Saved validation sigmoid means plot:", means_plot_path)


def save_validation_loss_plot(validation_history, attention_plot_dir):
    if len(validation_history) == 0:
        return

    epochs_plot = [row["epoch"] for row in validation_history]
    losses_plot = [row["validation_average_loss"] for row in validation_history]

    max_epoch = max(epochs_plot)

    plt.figure(figsize=(12, 6))
    add_phase_background(max_epoch)
    plt.plot(
        epochs_plot,
        losses_plot,
        marker="o",
        linewidth=2,
        color="black",
        label="validation average loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation average loss")
    plt.title("Validation average loss on unseen events")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")
    plt.tight_layout()
    loss_plot_path = os.path.join(
        attention_plot_dir,
        "validation_average_loss_progressive_introduction_of_hard_negatives.png",
    )
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print("Saved validation loss plot:", loss_plot_path)


def evaluate_model_after_epoch(
    model,
    dataset,
    cfg,
    validation_file_indices,
    epoch,
    hard_negative_fraction,
    max_positive_pairs,
    max_eval_events,
    random_seed,
    debug_pair_checks,
    debug_pair_checks_events,
):
    model.eval()

    eval_loss_sum = 0.0
    evaluated_events = 0
    skipped_events = 0

    eval_pos_sigmoid_sum = 0.0
    eval_pos_count = 0

    eval_random_neg_sigmoid_sum = 0.0
    eval_random_neg_count = 0
    debug_checks_done = 0

    torch.manual_seed(random_seed)

    if cfg.device_acc.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    with torch.no_grad():
        for file_idx in validation_file_indices:
            if evaluated_events >= max_eval_events:
                break

            print()
            print("=" * 80)
            print(f"Validation: loading held-out file_idx={file_idx}")
            print("=" * 80)

            file_data = dataset.get_file(file_idx)

            hits_tensor = file_data["hits_tensor"]
            padding_mask = file_data["padding_mask"]
            hit_to_particle_tensor = file_data["hit_to_particle_tensor"]

            n_events = hits_tensor.shape[0]

            for event_idx in range(n_events):
                if evaluated_events >= max_eval_events:
                    break

                batched_hits_cpu = hits_tensor[event_idx]
                batched_mask_cpu = padding_mask[event_idx]
                particle_ids_cpu = hit_to_particle_tensor[event_idx, 0]

                pairs1, pairs2, target = sample_positive_pairs_from_particle_ids(
                    particle_ids_cpu,
                    max_positive_pairs=max_positive_pairs,
                )

                if pairs1.numel() == 0:
                    skipped_events += 1
                    continue

                batched_hits = batched_hits_cpu.to(cfg.device_acc, dtype=model.dtype)
                batched_mask = batched_mask_cpu.to(cfg.device_acc)

                pairs1 = pairs1.to(cfg.device_acc).long()
                pairs2 = pairs2.to(cfg.device_acc).long()
                target = target.to(cfg.device_acc).float()
                particle_ids = particle_ids_cpu.to(cfg.device_acc).long()

                _, attention_weights = model(batched_hits, batched_mask)

                attention_map = attention_weights

                if attention_map.dim() == 3:
                    attention_map = attention_map[0]

                loss, loss_debug = top_attention_loss(
                    attention_map,
                    pairs1,
                    pairs2,
                    target,
                    particle_ids,
                    batched_mask,
                    return_debug=True,
                    hard_negative_fraction=hard_negative_fraction,
                    debug_print=(global_step < 5),
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite validation loss at file_idx={file_idx}, event_idx={event_idx}: {loss.item()}"
                    )

                if debug_pair_checks and debug_checks_done < debug_pair_checks_events:
                    print_pair_truth_debug(
                        stage="validation",
                        epoch=epoch,
                        file_idx=file_idx,
                        event_idx=event_idx,
                        pairs1=pairs1,
                        pairs2=pairs2,
                        target=target,
                        particle_ids=particle_ids,
                        padding_mask=batched_mask,
                        attention_map=attention_map,
                        loss_debug=loss_debug,
                    )
                    debug_checks_done += 1

                eval_loss_sum += loss.item()
                evaluated_events += 1

                positive_scores = loss_debug["positive_scores"]
                random_negative_scores = loss_debug["random_negative_scores"]

                if positive_scores.numel() > 0:
                    positive_sigmoid = torch.sigmoid(positive_scores)
                    eval_pos_sigmoid_sum += positive_sigmoid.sum().item()
                    eval_pos_count += positive_sigmoid.numel()

                if random_negative_scores.numel() > 0:
                    random_negative_sigmoid = torch.sigmoid(random_negative_scores)
                    eval_random_neg_sigmoid_sum += random_negative_sigmoid.sum().item()
                    eval_random_neg_count += random_negative_sigmoid.numel()

                if evaluated_events % 20 == 0:
                    print(f"Validation evaluated events: {evaluated_events}/{max_eval_events}")

            del file_data
            del hits_tensor
            del padding_mask
            del hit_to_particle_tensor

            if cfg.device_acc.type == "cuda":
                torch.cuda.empty_cache()

    validation_average_loss = eval_loss_sum / evaluated_events if evaluated_events > 0 else float("nan")
    validation_mean_pos_sigmoid = eval_pos_sigmoid_sum / eval_pos_count if eval_pos_count > 0 else float("nan")
    validation_mean_random_neg_sigmoid = (
        eval_random_neg_sigmoid_sum / eval_random_neg_count
        if eval_random_neg_count > 0
        else float("nan")
    )
    validation_gap_pos_random_neg = validation_mean_pos_sigmoid - validation_mean_random_neg_sigmoid

    print()
    print("=" * 80)
    print("Validation summary on held-out events")
    print("hard_negative_fraction:", hard_negative_fraction)
    print("evaluated_events:", evaluated_events)
    print("skipped_events:", skipped_events)
    print("validation_average_loss:", validation_average_loss)
    print("validation_mean_pos_sigmoid:", validation_mean_pos_sigmoid)
    print("validation_mean_random_neg_sigmoid:", validation_mean_random_neg_sigmoid)
    print("validation_gap_pos_random_neg:", validation_gap_pos_random_neg)
    print("=" * 80)

    return {
        "validation_average_loss": validation_average_loss,
        "validation_mean_pos_sigmoid": validation_mean_pos_sigmoid,
        "validation_mean_random_neg_sigmoid": validation_mean_random_neg_sigmoid,
        "validation_gap_pos_random_neg": validation_gap_pos_random_neg,
        "evaluated_events": evaluated_events,
        "skipped_events": skipped_events,
    }


def build_config():
    cfg = SeedConfig()

    cfg.device_acc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.input_tensor_path = "/gpfs/workdir/thibauts/tensor_output_charge_MH11000"
    cfg.dataset_name = "seeding_data_charge_MH11000"

    cfg.embedding_feature = [0, 1, 2, 3]
    cfg.high_level_features = []
    cfg.cosine_processing = []

    cfg.fourier_num_frequencies = [10, 10, 10, 10]
    cfg.dim_max = [500.0, 500.0, 500.0, 1.0]
    cfg.shift = [0.0, 0.0, 0.0, 0.0]

    cfg.dim_embedding = 128
    cfg.nb_layers_t = 2
    cfg.feed_forward_ratio = 4
    cfg.nb_heads = 4
    cfg.dropout = 0.1
    cfg.regression = False

    return cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--max_train_events", type=int, default=None)
    parser.add_argument("--max_eval_events", type=int, default=20)
    parser.add_argument("--max_positive_pairs", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--debug_pair_checks", type=int, default=1)
    parser.add_argument("--debug_pair_checks_events", type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(1993)

    cfg = build_config()

    num_epochs = args.num_epochs
    max_positive_pairs = args.max_positive_pairs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    print_every = args.print_every
    debug_pair_checks = bool(args.debug_pair_checks)
    debug_pair_checks_events = args.debug_pair_checks_events

    if args.run_name is None:
        run_name = f"MH11000_charge_E{num_epochs}_P{max_positive_pairs}_lr1e-3_true_validation"
    else:
        run_name = args.run_name

    checkpoint_dir = f"/gpfs/workdir/thibauts/dune_training_checkpoints_{run_name}"
    attention_plot_dir = f"/gpfs/workdir/thibauts/attention_plots_{run_name}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(attention_plot_dir, exist_ok=True)

    print("Device:", cfg.device_acc)

    if cfg.device_acc.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    metadata_path = os.path.join(
        cfg.input_tensor_path,
        f"metadata_{cfg.dataset_name}.pt",
    )

    print("Expected metadata path:", metadata_path)
    print("Metadata exists:", os.path.exists(metadata_path))

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=cfg.dataset_name,
        tensor_names=[
            "hits_tensor",
            "hit_to_particle_tensor",
            "padding_mask",
        ],
        device=torch.device("cpu"),
    )

    num_files = len(dataset.file_paths)

    train_file_indices, validation_file_indices = split_file_indices(
        num_files=num_files,
        validation_fraction=args.validation_fraction,
    )

    print()
    print("Dataset:")
    print("num_files:", num_files)
    print("file_paths:", dataset.file_paths)
    print("validation_fraction:", args.validation_fraction)
    print("num_train_files:", len(train_file_indices))
    print("num_validation_files:", len(validation_file_indices))
    print("train_file_indices:", train_file_indices)
    print("validation_file_indices:", validation_file_indices)

    model = SeedTransformer(cfg).to(cfg.device_acc)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print()
    print("Full training settings:")
    print("num_epochs:", num_epochs)
    print("max_train_events:", args.max_train_events)
    print("max_eval_events:", args.max_eval_events)
    print("max_positive_pairs:", max_positive_pairs)
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("checkpoint_dir:", checkpoint_dir)
    print("attention_plot_dir:", attention_plot_dir)
    print("input_tensor_path:", cfg.input_tensor_path)
    print("dataset_name:", cfg.dataset_name)
    print("embedding_feature:", cfg.embedding_feature)
    print("fourier_num_frequencies:", cfg.fourier_num_frequencies)
    print("dim_max:", cfg.dim_max)
    print("shift:", cfg.shift)
    print("debug_pair_checks:", debug_pair_checks)
    print("debug_pair_checks_events:", debug_pair_checks_events)

    global_step = 0
    epoch_history = []
    train_eval_history = []
    validation_history = []
    training_debug_checks_done = 0

    if cfg.device_acc.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        hard_negative_fraction = get_hard_negative_fraction(epoch)

        print()
        print("=" * 80)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("hard_negative_fraction:", hard_negative_fraction)
        print("=" * 80)

        model.train()

        epoch_loss_sum = 0.0
        successful_events = 0
        skipped_events = 0

        epoch_pos_sigmoid_sum = 0.0
        epoch_pos_count = 0

        epoch_random_neg_sigmoid_sum = 0.0
        epoch_random_neg_count = 0

        stop_epoch_early = False

        for file_idx in train_file_indices:
            if stop_epoch_early:
                break

            if args.max_train_events is not None and successful_events >= args.max_train_events:
                break

            print()
            print("#" * 80)
            print(f"Training: loading train file_idx={file_idx}")
            print("#" * 80)

            file_data = dataset.get_file(file_idx)

            hits_tensor = file_data["hits_tensor"]
            padding_mask = file_data["padding_mask"]
            hit_to_particle_tensor = file_data["hit_to_particle_tensor"]

            print("Loaded file on CPU:")
            print("hits_tensor:", hits_tensor.shape, hits_tensor.device)
            print("padding_mask:", padding_mask.shape, padding_mask.device)
            print("hit_to_particle_tensor:", hit_to_particle_tensor.shape, hit_to_particle_tensor.device)

            n_events = hits_tensor.shape[0]

            for event_idx in range(n_events):
                if args.max_train_events is not None and successful_events >= args.max_train_events:
                    stop_epoch_early = True
                    break

                batched_hits_cpu = hits_tensor[event_idx]
                batched_mask_cpu = padding_mask[event_idx]
                particle_ids_cpu = hit_to_particle_tensor[event_idx, 0]

                pairs1, pairs2, target = sample_positive_pairs_from_particle_ids(
                    particle_ids_cpu,
                    max_positive_pairs=max_positive_pairs,
                )

                if pairs1.numel() == 0:
                    skipped_events += 1
                    continue

                batched_hits = batched_hits_cpu.to(cfg.device_acc, dtype=model.dtype)
                batched_mask = batched_mask_cpu.to(cfg.device_acc)

                pairs1 = pairs1.to(cfg.device_acc).long()
                pairs2 = pairs2.to(cfg.device_acc).long()
                target = target.to(cfg.device_acc).float()
                particle_ids = particle_ids_cpu.to(cfg.device_acc).long()

                optimizer.zero_grad(set_to_none=True)

                _, attention_weights = model(batched_hits, batched_mask)

                attention_map = attention_weights

                if attention_map.dim() == 3:
                    attention_map = attention_map[0]

                loss, loss_debug = top_attention_loss(
                    attention_map,
                    pairs1,
                    pairs2,
                    target,
                    particle_ids,
                    batched_mask,
                    return_debug=True,
                    hard_negative_fraction=hard_negative_fraction,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch}, file_idx={file_idx}, event_idx={event_idx}: {loss.item()}"
                    )

                positive_scores = loss_debug["positive_scores"]
                random_negative_scores = loss_debug["random_negative_scores"]

                if debug_pair_checks and training_debug_checks_done < debug_pair_checks_events:
                    print_pair_truth_debug(
                        stage="training",
                        epoch=epoch,
                        file_idx=file_idx,
                        event_idx=event_idx,
                        pairs1=pairs1,
                        pairs2=pairs2,
                        target=target,
                        particle_ids=particle_ids,
                        padding_mask=batched_mask,
                        attention_map=attention_map,
                        loss_debug=loss_debug,
                    )
                    training_debug_checks_done += 1

                with torch.no_grad():
                    if positive_scores.numel() > 0:
                        positive_sigmoid = torch.sigmoid(positive_scores)
                        epoch_pos_sigmoid_sum += positive_sigmoid.sum().item()
                        epoch_pos_count += positive_sigmoid.numel()

                    if random_negative_scores.numel() > 0:
                        random_negative_sigmoid = torch.sigmoid(random_negative_scores)
                        epoch_random_neg_sigmoid_sum += random_negative_sigmoid.sum().item()
                        epoch_random_neg_count += random_negative_sigmoid.numel()

                loss.backward()
                optimizer.step()

                loss_value = loss.item()

                epoch_loss_sum += loss_value
                successful_events += 1
                global_step += 1

                if successful_events == 1 or successful_events % print_every == 0:
                    print()
                    print("-" * 80)
                    print(
                        f"epoch={epoch + 1} | "
                        f"global_step={global_step} | "
                        f"file_idx={file_idx} | "
                        f"event_idx={event_idx}"
                    )
                    print("hard_negative_fraction:", hard_negative_fraction)
                    print("sampled positive pairs:", pairs1.numel())
                    print("loss:", loss_value)
                    print("num positive pairs:", loss_debug["num_positive_pairs"].item())
                    print("num negative candidates:", loss_debug["num_negative_candidates"].item())
                    print("num selected negatives:", loss_debug["num_selected_negatives"].item())

                    if positive_scores.numel() > 0:
                        print(
                            "batch mean sigmoid positives:",
                            torch.sigmoid(positive_scores).mean().item(),
                        )

                    if random_negative_scores.numel() > 0:
                        print(
                            "batch mean sigmoid random negatives:",
                            torch.sigmoid(random_negative_scores).mean().item(),
                        )

                    if cfg.device_acc.type == "cuda":
                        print("current memory GB:", torch.cuda.memory_allocated() / 1024**3)
                        print("peak memory GB:", torch.cuda.max_memory_allocated() / 1024**3)

            del file_data
            del hits_tensor
            del padding_mask
            del hit_to_particle_tensor

            if cfg.device_acc.type == "cuda":
                torch.cuda.empty_cache()

        avg_loss = epoch_loss_sum / successful_events if successful_events > 0 else float("nan")
        mean_pos_sigmoid = epoch_pos_sigmoid_sum / epoch_pos_count if epoch_pos_count > 0 else float("nan")
        mean_random_neg_sigmoid = (
            epoch_random_neg_sigmoid_sum / epoch_random_neg_count
            if epoch_random_neg_count > 0
            else float("nan")
        )
        gap_pos_random_neg = mean_pos_sigmoid - mean_random_neg_sigmoid

        print()
        print("=" * 80)
        print(f"Epoch {epoch + 1} training summary")
        print("hard_negative_fraction:", hard_negative_fraction)
        print("successful_events:", successful_events)
        print("skipped_events:", skipped_events)
        print("average_loss:", avg_loss)
        print("mean_pos_sigmoid:", mean_pos_sigmoid)
        print("mean_random_neg_sigmoid:", mean_random_neg_sigmoid)
        print("gap_pos_random_neg:", gap_pos_random_neg)

        if cfg.device_acc.type == "cuda":
            print("peak memory GB:", torch.cuda.max_memory_allocated() / 1024**3)

        print("=" * 80)

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "hard_negative_fraction": hard_negative_fraction,
                "average_loss": avg_loss,
                "mean_pos_sigmoid": mean_pos_sigmoid,
                "mean_random_neg_sigmoid": mean_random_neg_sigmoid,
                "gap_pos_random_neg": gap_pos_random_neg,
                "successful_events": successful_events,
                "skipped_events": skipped_events,
            }
        )

        train_eval_metrics = evaluate_model_after_epoch(
            model=model,
            dataset=dataset,
            cfg=cfg,
            validation_file_indices=train_file_indices,
            epoch=epoch,
            hard_negative_fraction=hard_negative_fraction,
            max_positive_pairs=max_positive_pairs,
            max_eval_events=200,
            random_seed=12345,
            debug_pair_checks=False,
            debug_pair_checks_events=0,
        )

        train_eval_history.append(
            {
                "epoch": epoch + 1,
                "hard_negative_fraction": hard_negative_fraction,
                "train_eval_average_loss": train_eval_metrics["validation_average_loss"],
                "train_eval_mean_pos_sigmoid": train_eval_metrics["validation_mean_pos_sigmoid"],
                "train_eval_mean_random_neg_sigmoid": train_eval_metrics["validation_mean_random_neg_sigmoid"],
                "train_eval_gap_pos_random_neg": train_eval_metrics["validation_gap_pos_random_neg"],
                "evaluated_events": train_eval_metrics["evaluated_events"],
                "skipped_events": train_eval_metrics["skipped_events"],
            }
        )

        print()
        print("=" * 80)
        print("Train-set evaluation summary")
        print("train_eval_average_loss:", train_eval_metrics["validation_average_loss"])
        print("train_eval_mean_pos_sigmoid:", train_eval_metrics["validation_mean_pos_sigmoid"])
        print("train_eval_mean_random_neg_sigmoid:", train_eval_metrics["validation_mean_random_neg_sigmoid"])
        print("train_eval_gap_pos_random_neg:", train_eval_metrics["validation_gap_pos_random_neg"])
        print("=" * 80)

        validation_metrics = evaluate_model_after_epoch(
            model=model,
            dataset=dataset,
            cfg=cfg,
            validation_file_indices=validation_file_indices,
            epoch=epoch,
            hard_negative_fraction=hard_negative_fraction,
            max_positive_pairs=max_positive_pairs,
            max_eval_events=args.max_eval_events,
            random_seed=12345,
            debug_pair_checks=debug_pair_checks,
            debug_pair_checks_events=debug_pair_checks_events,
        )

        validation_history.append(
            {
                "epoch": epoch + 1,
                "hard_negative_fraction": hard_negative_fraction,
                "validation_average_loss": validation_metrics["validation_average_loss"],
                "validation_mean_pos_sigmoid": validation_metrics["validation_mean_pos_sigmoid"],
                "validation_mean_random_neg_sigmoid": validation_metrics["validation_mean_random_neg_sigmoid"],
                "validation_gap_pos_random_neg": validation_metrics["validation_gap_pos_random_neg"],
                "evaluated_events": validation_metrics["evaluated_events"],
                "skipped_events": validation_metrics["skipped_events"],
            }
        )

        save_metrics_csv(epoch_history, checkpoint_dir)
        save_gap_plot(epoch_history, attention_plot_dir)
        save_loss_plot(epoch_history, attention_plot_dir)

        save_train_eval_metrics_csv(train_eval_history, checkpoint_dir)

        save_validation_metrics_csv(validation_history, checkpoint_dir)
        save_validation_gap_plot(validation_history, attention_plot_dir)
        save_validation_loss_plot(validation_history, attention_plot_dir)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"dune_seed_transformer_epoch_{epoch + 1}.pt",
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "hard_negative_fraction": hard_negative_fraction,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "mean_pos_sigmoid": mean_pos_sigmoid,
                "mean_random_neg_sigmoid": mean_random_neg_sigmoid,
                "gap_pos_random_neg": gap_pos_random_neg,
                "train_eval_average_loss": train_eval_metrics["validation_average_loss"],
                "train_eval_mean_pos_sigmoid": train_eval_metrics["validation_mean_pos_sigmoid"],
                "train_eval_mean_random_neg_sigmoid": train_eval_metrics["validation_mean_random_neg_sigmoid"],
                "train_eval_gap_pos_random_neg": train_eval_metrics["validation_gap_pos_random_neg"],
                "validation_average_loss": validation_metrics["validation_average_loss"],
                "validation_mean_pos_sigmoid": validation_metrics["validation_mean_pos_sigmoid"],
                "validation_mean_random_neg_sigmoid": validation_metrics["validation_mean_random_neg_sigmoid"],
                "validation_gap_pos_random_neg": validation_metrics["validation_gap_pos_random_neg"],
                "cfg": cfg,
                "train_file_indices": train_file_indices,
                "validation_file_indices": validation_file_indices,
            },
            checkpoint_path,
        )

        print("Saved checkpoint:", checkpoint_path)

    print()
    print("Full training finished successfully.")
    print("Final training metrics CSV:", os.path.join(checkpoint_dir, "training_gap_metrics.csv"))
    print("Final train-set evaluation metrics CSV:", os.path.join(checkpoint_dir, "train_eval_gap_metrics.csv"))
    print("Final validation metrics CSV:", os.path.join(checkpoint_dir, "validation_gap_metrics.csv"))
    print("Final training gap plot:", os.path.join(attention_plot_dir, "gap_pos_vs_random_neg.png"))
    print("Final validation gap plot:", os.path.join(attention_plot_dir, "validation_gap_pos_vs_random_neg.png"))


if __name__ == "__main__":
    main()
