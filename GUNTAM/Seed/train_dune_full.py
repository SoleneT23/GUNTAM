import os
import csv
import argparse

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedLoss import top_attention_loss
from GUNTAM.IO.PrepareTensor import sample_positive_pairs_from_particle_ids




def save_attention_heatmap(
    model,
    dataset,
    cfg,
    file_idx,
    event_idx,
    output_dir,
    max_plot_hits=500,
):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    file_data = dataset.get_file(file_idx)

    hits_tensor = file_data["hits_tensor"]
    padding_mask = file_data["padding_mask"]

    batched_hits_cpu = hits_tensor[event_idx]       # [1, 5000, 3]
    batched_mask_cpu = padding_mask[event_idx]      # [1, 5000]

    batched_hits = batched_hits_cpu.to(cfg.device_acc, dtype=model.dtype)
    batched_mask = batched_mask_cpu.to(cfg.device_acc)

    with torch.no_grad():
        encoded_space_points, attention_weights = model(
            batched_hits,
            batched_mask,
        )

    attention_map = attention_weights

    if attention_map.dim() == 3:
        attention_map = attention_map[0]

    attention_map = torch.sigmoid(attention_map.detach().cpu())

    padding_mask_cpu = batched_mask_cpu.detach().cpu()
    real_hit_mask = ~padding_mask_cpu[0]

    real_indices = torch.nonzero(
        real_hit_mask,
        as_tuple=False,
    ).squeeze(-1)

    real_indices = real_indices[:max_plot_hits]

    attention_to_plot = attention_map[real_indices][:, real_indices]

    output_path = os.path.join(
        output_dir,
        (
            f"overfit_one_event_attention_heatmap"
            f"_file{file_idx}"
            f"_event{event_idx}"
            f"_real_first{len(real_indices)}.png"
        ),
    )

    plt.figure(figsize=(8, 7))
    im = plt.imshow(
        attention_to_plot.numpy(),
        aspect="auto",
    )

    cbar = plt.colorbar(im, label="sigmoid(attention score)")
    cbar.formatter = ticker.FormatStrFormatter("%.6f")
    cbar.update_ticks()

    plt.title(
        (
            f"Overfit one-event attention heatmap | file {file_idx}, "
            f"event {event_idx} | first {len(real_indices)} real hits"
        )
    )
    plt.xlabel("Hit index")
    plt.ylabel("Hit index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved overfit one-event attention heatmap:", output_path)

    del file_data
    del hits_tensor
    del padding_mask

    if cfg.device_acc.type == "cuda":
        torch.cuda.empty_cache()


def get_hard_negative_fraction(epoch):
    if epoch < 3:
        return 0.0
    if epoch < 10:
        return 0.1
    if epoch < 20:
        return 0.2
    if epoch < 30:
        return 0.3
    if epoch < 40:
        return 0.4
    return 0.5


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


def save_metrics_csv(epoch_history, checkpoint_dir):
    metrics_csv_path = os.path.join(
        checkpoint_dir,
        "training_gap_metrics.csv",
    )

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
    metrics_csv_path = os.path.join(
        checkpoint_dir,
        "validation_gap_metrics.csv",
    )

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
    plt.title("Gap between positive pairs and random negative pairs")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()

    gap_plot_path = os.path.join(
        attention_plot_dir,
        "gap_pos_vs_random_neg.png",
    )

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
    plt.title("Positive vs random negative sigmoid scores")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()

    means_plot_path = os.path.join(
        attention_plot_dir,
        "mean_sigmoid_pos_vs_random_neg.png",
    )

    plt.savefig(means_plot_path, dpi=200)
    plt.close()

    print("Saved sigmoid means plot:", means_plot_path)


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
    plt.title("Validation gap between positive pairs and random negative pairs")
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()

    gap_plot_path = os.path.join(
        attention_plot_dir,
        "validation_gap_pos_vs_random_neg.png",
    )

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
    plt.title("Validation positive vs random negative sigmoid scores")
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
    plt.title("Average loss with progressive introduction of hard negatives")
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
    plt.title("Validation average loss with progressive introduction of hard negatives")
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
    hard_negative_fraction,
    max_positive_pairs=2000,
    max_eval_events=100,
    random_seed=12345,
):
    model.eval()

    eval_loss_sum = 0.0
    evaluated_events = 0
    skipped_events = 0

    eval_pos_sigmoid_sum = 0.0
    eval_pos_count = 0

    eval_random_neg_sigmoid_sum = 0.0
    eval_random_neg_count = 0

    torch.manual_seed(random_seed)

    if cfg.device_acc.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    with torch.no_grad():
        num_files = len(dataset.file_paths)

        for file_idx in range(num_files):
            if evaluated_events >= max_eval_events:
                break

            print()
            print("=" * 80)
            print(f"Validation: loading file {file_idx + 1}/{num_files}")
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

                batched_hits = batched_hits_cpu.to(
                    cfg.device_acc,
                    dtype=model.dtype,
                )

                batched_mask = batched_mask_cpu.to(cfg.device_acc)

                pairs1 = pairs1.to(cfg.device_acc).long()
                pairs2 = pairs2.to(cfg.device_acc).long()
                target = target.to(cfg.device_acc).float()
                particle_ids = particle_ids_cpu.to(cfg.device_acc).long()

                _, attention_weights = model(
                    batched_hits,
                    batched_mask,
                )

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
                        f"Non-finite validation loss at file_idx={file_idx}, event_idx={event_idx}: {loss.item()}"
                    )

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

    if evaluated_events > 0:
        validation_average_loss = eval_loss_sum / evaluated_events
    else:
        validation_average_loss = float("nan")

    if eval_pos_count > 0:
        validation_mean_pos_sigmoid = eval_pos_sigmoid_sum / eval_pos_count
    else:
        validation_mean_pos_sigmoid = float("nan")

    if eval_random_neg_count > 0:
        validation_mean_random_neg_sigmoid = eval_random_neg_sigmoid_sum / eval_random_neg_count
    else:
        validation_mean_random_neg_sigmoid = float("nan")

    validation_gap_pos_random_neg = validation_mean_pos_sigmoid - validation_mean_random_neg_sigmoid

    print()
    print("=" * 80)
    print("Validation summary")
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


def save_pair_confusion_matrix(
    model,
    dataset,
    cfg,
    output_dir,
    max_positive_pairs=2000,
    threshold=0.5,
    max_events=100,
):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    used_events = 0

    with torch.no_grad():
        num_files = len(dataset.file_paths)

        for file_idx in range(num_files):
            if used_events >= max_events:
                break

            print()
            print("=" * 80)
            print(f"Confusion matrix evaluation: loading file {file_idx + 1}/{num_files}")
            print("=" * 80)

            file_data = dataset.get_file(file_idx)

            hits_tensor = file_data["hits_tensor"]
            padding_mask = file_data["padding_mask"]
            hit_to_particle_tensor = file_data["hit_to_particle_tensor"]

            n_events = hits_tensor.shape[0]

            for event_idx in range(n_events):
                if used_events >= max_events:
                    break

                batched_hits_cpu = hits_tensor[event_idx]
                batched_mask_cpu = padding_mask[event_idx]
                particle_ids_cpu = hit_to_particle_tensor[event_idx, 0]

                pairs1, pairs2, target = sample_positive_pairs_from_particle_ids(
                    particle_ids_cpu,
                    max_positive_pairs=max_positive_pairs,
                )

                if pairs1.numel() == 0:
                    continue

                batched_hits = batched_hits_cpu.to(
                    cfg.device_acc,
                    dtype=model.dtype,
                )

                batched_mask = batched_mask_cpu.to(cfg.device_acc)

                pairs1 = pairs1.to(cfg.device_acc).long()
                pairs2 = pairs2.to(cfg.device_acc).long()
                target = target.to(cfg.device_acc).float()
                particle_ids = particle_ids_cpu.to(cfg.device_acc).long()

                _, attention_weights = model(
                    batched_hits,
                    batched_mask,
                )

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
                    hard_negative_fraction=0.0,
                )

                positive_scores = loss_debug["positive_scores"]
                random_negative_scores = loss_debug["random_negative_scores"]

                if positive_scores.numel() == 0:
                    continue

                if random_negative_scores.numel() == 0:
                    continue

                positive_probs = torch.sigmoid(positive_scores)
                negative_probs = torch.sigmoid(random_negative_scores)

                positive_pred = positive_probs >= threshold
                negative_pred = negative_probs >= threshold

                tp += positive_pred.sum().item()
                fn += (~positive_pred).sum().item()
                fp += negative_pred.sum().item()
                tn += (~negative_pred).sum().item()

                used_events += 1

                if used_events % 10 == 0:
                    print(f"Evaluated events: {used_events}/{max_events}")

            del file_data
            del hits_tensor
            del padding_mask
            del hit_to_particle_tensor

            if cfg.device_acc.type == "cuda":
                torch.cuda.empty_cache()

    total = tp + tn + fp + fn

    if total == 0:
        print("No pairs found for confusion matrix.")
        return

    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    print()
    print("=" * 80)
    print("Pair confusion matrix")
    print("threshold:", threshold)
    print("evaluated events:", used_events)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("=" * 80)

    matrix = [
        [tn, fp],
        [fn, tp],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Predicted negative", "Predicted positive"])
    ax.set_yticklabels(["True negative", "True positive"])

    ax.set_title(
        f"Pair confusion matrix\n"
        f"threshold={threshold}, accuracy={accuracy:.3f}, F1={f1:.3f}"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                fontsize=14,
            )

    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "pair_confusion_matrix_random_negatives.png",
    )

    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved confusion matrix:", output_path)

    metrics_path = os.path.join(
        output_dir,
        "pair_confusion_matrix_metrics.txt",
    )

    with open(metrics_path, "w") as f:
        f.write(f"threshold: {threshold}\n")
        f.write(f"evaluated_events: {used_events}\n")
        f.write(f"TN: {tn}\n")
        f.write(f"FP: {fp}\n")
        f.write(f"FN: {fn}\n")
        f.write(f"TP: {tp}\n")
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")

    print("Saved confusion matrix metrics:", metrics_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="Number of training epochs. Use 30 or 40 for your comparison.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional name used in checkpoint and plot directories.",
    )

    parser.add_argument(
        "--max_eval_events",
        type=int,
        default=100,
        help="Number of events used for validation after each epoch.",
    )

    args = parser.parse_args()

    cfg = SeedConfig()

    cfg.device_acc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.input_tensor_path = "/gpfs/workdir/thibauts/tensor_output_shuffled"
    cfg.dataset_name = "seeding_data_MH5000"

    cfg.embedding_feature = [0, 1, 2]
    cfg.high_level_features = []
    cfg.cosine_processing = []

    cfg.fourier_num_frequencies = [10, 10, 10]
    cfg.dim_max = [500.0, 500.0, 500.0]
    cfg.shift = [0.0, 0.0, 0.0]

    cfg.dim_embedding = 128
    cfg.nb_layers_t = 2
    cfg.feed_forward_ratio = 4
    cfg.nb_heads = 4
    cfg.dropout = 0.1
    cfg.regression = False

    num_epochs = args.num_epochs
    max_positive_pairs = 2000
    learning_rate = 1e-3
    weight_decay = 1e-2

    print_every = 50

    if args.run_name is None:
        run_name = f"E{num_epochs}_P{max_positive_pairs}_lr1e-3_progressive_introduction_of_hard_negatives_gap"
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

    print()
    print("Dataset:")
    print("num_files:", num_files)
    print("file_paths:", dataset.file_paths)

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
    print("max_positive_pairs:", max_positive_pairs)
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("checkpoint_dir:", checkpoint_dir)
    print("attention_plot_dir:", attention_plot_dir)
    print("progressive introduction of hard negatives phases:")
    print("  epochs 1-3: hard_negative_fraction = 0.0")
    print("  epochs 4-10: hard_negative_fraction = 0.1")
    print("  epochs 11-20: hard_negative_fraction = 0.2")
    print("  epochs 21-30: hard_negative_fraction = 0.3")
    print("  epochs 31-40: hard_negative_fraction = 0.4")
    print("  epochs 41-50: hard_negative_fraction = 0.5")

    global_step = 0
    epoch_history = []
    validation_history = []

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

        for file_idx in range(num_files):
            print()
            print("#" * 80)
            print(f"Loading file {file_idx + 1}/{num_files}")
            print("#" * 80)

            file_data = dataset.get_file(file_idx)

            hits_tensor = file_data["hits_tensor"]
            padding_mask = file_data["padding_mask"]
            hit_to_particle_tensor = file_data["hit_to_particle_tensor"]

            print("Loaded file on CPU:")
            print("hits_tensor:", hits_tensor.shape, hits_tensor.device)
            print("padding_mask:", padding_mask.shape, padding_mask.device)
            print(
                "hit_to_particle_tensor:",
                hit_to_particle_tensor.shape,
                hit_to_particle_tensor.device,
            )

            n_events = hits_tensor.shape[0]

            for event_idx in range(n_events):
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

                batched_hits = batched_hits_cpu.to(
                    cfg.device_acc,
                    dtype=model.dtype,
                )

                batched_mask = batched_mask_cpu.to(cfg.device_acc)

                pairs1 = pairs1.to(cfg.device_acc).long()
                pairs2 = pairs2.to(cfg.device_acc).long()
                target = target.to(cfg.device_acc).float()
                particle_ids = particle_ids_cpu.to(cfg.device_acc).long()

                optimizer.zero_grad(set_to_none=True)

                _, attention_weights = model(
                    batched_hits,
                    batched_mask,
                )

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

                if global_step % 100 == 0:
                    print("hard_negative_fraction:", hard_negative_fraction)
                    print("num positive pairs:", loss_debug["num_positive_pairs"].item())
                    print("num negative candidates:", loss_debug["num_negative_candidates"].item())
                    print("num selected negatives:", loss_debug["num_selected_negatives"].item())

                with torch.no_grad():
                    positive_scores = loss_debug["positive_scores"]
                    random_negative_scores = loss_debug["random_negative_scores"]

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
                        print(
                            "current memory GB:",
                            torch.cuda.memory_allocated() / 1024**3,
                        )
                        print(
                            "peak memory GB:",
                            torch.cuda.max_memory_allocated() / 1024**3,
                        )

            del file_data
            del hits_tensor
            del padding_mask
            del hit_to_particle_tensor

            if cfg.device_acc.type == "cuda":
                torch.cuda.empty_cache()

        if successful_events > 0:
            avg_loss = epoch_loss_sum / successful_events
        else:
            avg_loss = float("nan")

        if epoch_pos_count > 0:
            mean_pos_sigmoid = epoch_pos_sigmoid_sum / epoch_pos_count
        else:
            mean_pos_sigmoid = float("nan")

        if epoch_random_neg_count > 0:
            mean_random_neg_sigmoid = epoch_random_neg_sigmoid_sum / epoch_random_neg_count
        else:
            mean_random_neg_sigmoid = float("nan")

        gap_pos_random_neg = mean_pos_sigmoid - mean_random_neg_sigmoid

        print()
        print("=" * 80)
        print(f"Epoch {epoch + 1} summary")
        print("hard_negative_fraction:", hard_negative_fraction)
        print("successful_events:", successful_events)
        print("skipped_events:", skipped_events)
        print("average_loss:", avg_loss)
        print("mean_pos_sigmoid:", mean_pos_sigmoid)
        print("mean_random_neg_sigmoid:", mean_random_neg_sigmoid)
        print("gap_pos_random_neg:", gap_pos_random_neg)

        if cfg.device_acc.type == "cuda":
            print(
                "peak memory GB:",
                torch.cuda.max_memory_allocated() / 1024**3,
            )

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

        validation_metrics = evaluate_model_after_epoch(
            model=model,
            dataset=dataset,
            cfg=cfg,
            hard_negative_fraction=hard_negative_fraction,
            max_positive_pairs=max_positive_pairs,
            max_eval_events=args.max_eval_events,
            random_seed=12345,
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

        save_metrics_csv(
            epoch_history=epoch_history,
            checkpoint_dir=checkpoint_dir,
        )

        save_gap_plot(
            epoch_history=epoch_history,
            attention_plot_dir=attention_plot_dir,
        )

        save_loss_plot(
            epoch_history=epoch_history,
            attention_plot_dir=attention_plot_dir,
        )

        save_validation_metrics_csv(
            validation_history=validation_history,
            checkpoint_dir=checkpoint_dir,
        )

        save_validation_gap_plot(
            validation_history=validation_history,
            attention_plot_dir=attention_plot_dir,
        )

        save_validation_loss_plot(
            validation_history=validation_history,
            attention_plot_dir=attention_plot_dir,
        )

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
                "validation_average_loss": validation_metrics["validation_average_loss"],
                "validation_mean_pos_sigmoid": validation_metrics["validation_mean_pos_sigmoid"],
                "validation_mean_random_neg_sigmoid": validation_metrics["validation_mean_random_neg_sigmoid"],
                "validation_gap_pos_random_neg": validation_metrics["validation_gap_pos_random_neg"],
                "cfg": cfg,
            },
            checkpoint_path,
        )

        print("Saved checkpoint:", checkpoint_path)

    save_attention_heatmap(
        model=model,
        dataset=dataset,
        cfg=cfg,
        file_idx=3,
        event_idx=219,
        output_dir=attention_plot_dir,
        max_plot_hits=500,
    )

    save_pair_confusion_matrix(
        model=model,
        dataset=dataset,
        cfg=cfg,
        output_dir=attention_plot_dir,
        max_positive_pairs=max_positive_pairs,
        threshold=0.5,
        max_events=100,
    )

    print()
    print("Full training finished successfully.")
    print("Final training metrics CSV:", os.path.join(checkpoint_dir, "training_gap_metrics.csv"))
    print("Final validation metrics CSV:", os.path.join(checkpoint_dir, "validation_gap_metrics.csv"))
    print("Final training gap plot:", os.path.join(attention_plot_dir, "gap_pos_vs_random_neg.png"))
    print("Final validation gap plot:", os.path.join(attention_plot_dir, "validation_gap_pos_vs_random_neg.png"))
    print(
        "Final training sigmoid means plot:",
        os.path.join(attention_plot_dir, "mean_sigmoid_pos_vs_random_neg.png"),
    )
    print(
        "Final validation sigmoid means plot:",
        os.path.join(attention_plot_dir, "validation_mean_sigmoid_pos_vs_random_neg.png"),
    )


if __name__ == "__main__":
    main()