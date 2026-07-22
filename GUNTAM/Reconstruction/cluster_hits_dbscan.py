import os
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig


DEFAULT_CHECKPOINT_PATH = (
    "/gpfs/workdir/thibauts/"
    "dune_training_checkpoints_MH11000_norm_fastlocal_10000events_val2000_E20_lr1e4_nohard/"
    "dune_seed_transformer_epoch_20.pt"
)

DEFAULT_OUTPUT_DIR = "/gpfs/workdir/thibauts/dbscan_fastlocal_epoch20"


def build_config():
    cfg = SeedConfig()

    cfg.device_acc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.input_tensor_path = "/gpfs/workdir/thibauts/tensor_output_charge_log_norm_MH11000"
    cfg.dataset_name = "seeding_data_charge_log_norm_MH11000"

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


def squeeze_attention(attention_map):
    if attention_map.dim() == 4:
        attention_map = attention_map[0, 0]
    elif attention_map.dim() == 3:
        attention_map = attention_map[0]

    return attention_map


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    return checkpoint


def get_dataset(cfg):
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

    return dataset


def compute_cluster_metrics(labels, true_particle_ids):
    labels = np.asarray(labels)
    true_particle_ids = np.asarray(true_particle_ids)

    if labels.size == 0:
        return {
            "n_pred_clusters": 0,
            "n_true_particles": 0,
            "noise_fraction": float("nan"),
            "cluster_purity": float("nan"),
            "particle_completeness": float("nan"),
            "ari": float("nan"),
            "nmi": float("nan"),
        }

    predicted_clusters = sorted([c for c in np.unique(labels) if c != -1])
    true_particles = sorted(np.unique(true_particle_ids))

    n_pred_clusters = len(predicted_clusters)
    n_true_particles = len(true_particles)

    noise_fraction = float((labels == -1).mean())

    purity_num = 0
    purity_den = 0

    for cluster_id in predicted_clusters:
        cluster_mask = labels == cluster_id
        cluster_true = true_particle_ids[cluster_mask]

        if cluster_true.size == 0:
            continue

        _, counts = np.unique(cluster_true, return_counts=True)
        purity_num += counts.max()
        purity_den += cluster_true.size

    cluster_purity = purity_num / purity_den if purity_den > 0 else float("nan")

    completeness_num = 0
    completeness_den = 0

    for pid in true_particles:
        particle_mask = true_particle_ids == pid
        particle_labels = labels[particle_mask]

        completeness_den += particle_labels.size

        particle_labels_non_noise = particle_labels[particle_labels != -1]

        if particle_labels_non_noise.size == 0:
            continue

        _, counts = np.unique(particle_labels_non_noise, return_counts=True)
        completeness_num += counts.max()

    particle_completeness = (
        completeness_num / completeness_den if completeness_den > 0 else float("nan")
    )

    try:
        ari = float(adjusted_rand_score(true_particle_ids, labels))
    except Exception:
        ari = float("nan")

    try:
        nmi = float(normalized_mutual_info_score(true_particle_ids, labels))
    except Exception:
        nmi = float("nan")

    return {
        "n_pred_clusters": n_pred_clusters,
        "n_true_particles": n_true_particles,
        "noise_fraction": noise_fraction,
        "cluster_purity": cluster_purity,
        "particle_completeness": particle_completeness,
        "ari": ari,
        "nmi": nmi,
    }


def cluster_event_with_dbscan(
    attention_weights,
    padding_mask,
    particle_ids,
    score_threshold,
    min_samples,
    max_real_hits,
):
    attention_map = squeeze_attention(attention_weights)

    padding_mask_flat = flatten_mask(padding_mask)
    particle_ids_flat = flatten_particle_ids(particle_ids)

    real_hit_mask = (~padding_mask_flat) & (particle_ids_flat >= 0)
    real_indices = torch.nonzero(real_hit_mask, as_tuple=False).view(-1)

    n_real_hits = real_indices.numel()

    if n_real_hits == 0:
        raise RuntimeError("No real hits in this event.")

    if n_real_hits > max_real_hits:
        raise RuntimeError(
            f"Too many real hits for dense DBSCAN: {n_real_hits}. "
            f"max_real_hits={max_real_hits}."
        )

    real_indices_device = real_indices.to(attention_map.device)

    logits = attention_map[real_indices_device][:, real_indices_device]

    scores = torch.sigmoid(logits)

    scores = 0.5 * (scores + scores.T)

    distances = 1.0 - scores
    distances.fill_diagonal_(0.0)

    distances_np = distances.detach().cpu().float().numpy()

    eps = 1.0 - score_threshold

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(distances_np)

    true_particle_ids = particle_ids_flat[real_indices].cpu().numpy()
    real_indices_np = real_indices.cpu().numpy()

    return labels, true_particle_ids, real_indices_np


def save_event_labels(
    output_dir,
    file_idx,
    event_idx,
    labels,
    true_particle_ids,
    real_indices,
    score_threshold,
    eps,
    min_samples,
):
    output_path = output_dir / f"dbscan_labels_file{file_idx}_event{event_idx}.npz"

    np.savez_compressed(
        output_path,
        labels=labels,
        true_particle_ids=true_particle_ids,
        real_indices=real_indices,
        score_threshold=score_threshold,
        eps=eps,
        min_samples=min_samples,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--score_threshold", type=float, default=0.90)
    parser.add_argument("--min_samples", type=int, default=3)
    parser.add_argument("--max_events", type=int, default=20)
    parser.add_argument("--max_real_hits", type=int, default=4000)

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    output_dir = Path(args.output_dir)
    score_threshold = args.score_threshold
    min_samples = args.min_samples
    max_events = args.max_events
    max_real_hits = args.max_real_hits

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config()

    print("Device:", cfg.device_acc)

    if cfg.device_acc.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Loading checkpoint:")
    print(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path, cfg.device_acc)

    model = SeedTransformer(cfg).to(cfg.device_acc)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    dataset = get_dataset(cfg)

    if "validation_file_indices" in checkpoint:
        file_indices = [int(x) for x in checkpoint["validation_file_indices"]]
        print("Using validation_file_indices from checkpoint.")
    else:
        num_files = len(dataset.file_paths)
        split_index = int(num_files * 0.8)
        file_indices = list(range(split_index, num_files))
        print("Using default validation split.")

    print("file_indices:", file_indices)
    print("score_threshold:", score_threshold)
    print("eps:", 1.0 - score_threshold)
    print("min_samples:", min_samples)
    print("max_events:", max_events)
    print("max_real_hits:", max_real_hits)
    print("output_dir:", output_dir)

    summary_rows = []

    evaluated_events = 0
    skipped_events = 0

    for file_idx in file_indices:
        if evaluated_events >= max_events:
            break

        print()
        print("=" * 80)
        print("Loading file_idx:", file_idx)
        print("=" * 80)

        file_data = dataset.get_file(file_idx)

        hits_tensor = file_data["hits_tensor"]
        padding_mask_tensor = file_data["padding_mask"]
        hit_to_particle_tensor = file_data["hit_to_particle_tensor"]

        n_events = hits_tensor.shape[0]

        for event_idx in range(n_events):
            if evaluated_events >= max_events:
                break

            hits_cpu = hits_tensor[event_idx]
            mask_cpu = padding_mask_tensor[event_idx]
            particle_ids_cpu = hit_to_particle_tensor[event_idx, 0]

            particle_ids_flat = flatten_particle_ids(particle_ids_cpu)
            padding_mask_flat = flatten_mask(mask_cpu)

            real_hit_mask = (~padding_mask_flat) & (particle_ids_flat >= 0)
            n_real_hits = int(real_hit_mask.sum().item())

            if n_real_hits < min_samples:
                skipped_events += 1
                continue

            if n_real_hits > max_real_hits:
                print(
                    f"Skipping file_idx={file_idx}, event_idx={event_idx}: "
                    f"n_real_hits={n_real_hits} > max_real_hits={max_real_hits}"
                )
                skipped_events += 1
                continue

            true_ids_this_event = particle_ids_flat[real_hit_mask]
            unique_true_ids = torch.unique(true_ids_this_event)

            if unique_true_ids.numel() < 2:
                skipped_events += 1
                continue

            hits_gpu = hits_cpu.to(cfg.device_acc, dtype=model.dtype)
            mask_gpu = mask_cpu.to(cfg.device_acc)

            print(
                f"Forward + DBSCAN file_idx={file_idx}, event_idx={event_idx}, "
                f"n_real_hits={n_real_hits}, n_true_particles={unique_true_ids.numel()}"
            )

            with torch.no_grad():
                _, attention_weights = model(hits_gpu, mask_gpu)

            labels, true_particle_ids, real_indices = cluster_event_with_dbscan(
                attention_weights=attention_weights,
                padding_mask=mask_cpu,
                particle_ids=particle_ids_cpu,
                score_threshold=score_threshold,
                min_samples=min_samples,
                max_real_hits=max_real_hits,
            )

            metrics = compute_cluster_metrics(labels, true_particle_ids)

            eps = 1.0 - score_threshold

            labels_path = save_event_labels(
                output_dir=output_dir,
                file_idx=file_idx,
                event_idx=event_idx,
                labels=labels,
                true_particle_ids=true_particle_ids,
                real_indices=real_indices,
                score_threshold=score_threshold,
                eps=eps,
                min_samples=min_samples,
            )

            row = {
                "file_idx": file_idx,
                "event_idx": event_idx,
                "n_real_hits": n_real_hits,
                "score_threshold": score_threshold,
                "eps": eps,
                "min_samples": min_samples,
                "n_pred_clusters": metrics["n_pred_clusters"],
                "n_true_particles": metrics["n_true_particles"],
                "noise_fraction": metrics["noise_fraction"],
                "cluster_purity": metrics["cluster_purity"],
                "particle_completeness": metrics["particle_completeness"],
                "ari": metrics["ari"],
                "nmi": metrics["nmi"],
                "labels_path": str(labels_path),
            }

            summary_rows.append(row)

            print("n_pred_clusters:", metrics["n_pred_clusters"])
            print("n_true_particles:", metrics["n_true_particles"])
            print("noise_fraction:", metrics["noise_fraction"])
            print("cluster_purity:", metrics["cluster_purity"])
            print("particle_completeness:", metrics["particle_completeness"])
            print("ari:", metrics["ari"])
            print("nmi:", metrics["nmi"])

            evaluated_events += 1

            del hits_gpu
            del mask_gpu
            del attention_weights

            if cfg.device_acc.type == "cuda":
                torch.cuda.empty_cache()

        del file_data
        del hits_tensor
        del padding_mask_tensor
        del hit_to_particle_tensor

        if cfg.device_acc.type == "cuda":
            torch.cuda.empty_cache()

    if len(summary_rows) == 0:
        raise RuntimeError(
            "No event was clustered. "
            "Try increasing max_real_hits or max_events, or lowering min_samples."
        )

    summary_csv = output_dir / "dbscan_summary.csv"

    fieldnames = list(summary_rows[0].keys())

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    mean_cluster_purity = np.nanmean([r["cluster_purity"] for r in summary_rows])
    mean_particle_completeness = np.nanmean(
        [r["particle_completeness"] for r in summary_rows]
    )
    mean_noise_fraction = np.nanmean([r["noise_fraction"] for r in summary_rows])
    mean_ari = np.nanmean([r["ari"] for r in summary_rows])
    mean_nmi = np.nanmean([r["nmi"] for r in summary_rows])

    print()
    print("=" * 80)
    print("DBSCAN reconstruction finished")
    print("evaluated_events:", evaluated_events)
    print("skipped_events:", skipped_events)
    print("summary_csv:", summary_csv)
    print("mean_cluster_purity:", mean_cluster_purity)
    print("mean_particle_completeness:", mean_particle_completeness)
    print("mean_noise_fraction:", mean_noise_fraction)
    print("mean_ari:", mean_ari)
    print("mean_nmi:", mean_nmi)
    print("=" * 80)


if __name__ == "__main__":
    main()