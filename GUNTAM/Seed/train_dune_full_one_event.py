import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedLoss import top_attention_loss
from GUNTAM.IO.PrepareTensor import sample_positive_pairs_from_particle_ids


def print_attention_debug(attention_map, batched_mask, pairs1, pairs2):
    with torch.no_grad():
        real_hit_mask = ~batched_mask[0]
        real_attention = attention_map[real_hit_mask][:, real_hit_mask]
        finite_scores = real_attention[torch.isfinite(real_attention)]

        print("raw attention logits stats on real-hit square:")
        print("  real hits:", real_hit_mask.sum().item())
        print("  finite entries:", finite_scores.numel())

        if finite_scores.numel() > 0:
            print("  min:", finite_scores.min().item())
            print("  max:", finite_scores.max().item())
            print("  mean:", finite_scores.mean().item())
            print("  std:", finite_scores.std().item())

            sig = torch.sigmoid(finite_scores)
            print("sigmoid attention stats on real-hit square:")
            print("  min:", sig.min().item())
            print("  max:", sig.max().item())
            print("  mean:", sig.mean().item())
            print("  std:", sig.std().item())

        pos_scores = attention_map[pairs1, pairs2]
        pos_scores = pos_scores[torch.isfinite(pos_scores)]

        print("positive pair logits stats:")
        print("  positive entries:", pos_scores.numel())

        if pos_scores.numel() > 0:
            print("  min:", pos_scores.min().item())
            print("  max:", pos_scores.max().item())
            print("  mean:", pos_scores.mean().item())
            print("  std:", pos_scores.std().item())

            pos_sig = torch.sigmoid(pos_scores)
            print("positive pair sigmoid stats:")
            print("  min:", pos_sig.min().item())
            print("  max:", pos_sig.max().item())
            print("  mean:", pos_sig.mean().item())
            print("  std:", pos_sig.std().item())


def print_loss_debug(loss_debug):
    if loss_debug is None:
        return

    positive_scores = loss_debug["positive_scores"]
    negative_scores = loss_debug["negative_scores"]

    positive_scores = positive_scores[torch.isfinite(positive_scores)]
    negative_scores = negative_scores[torch.isfinite(negative_scores)]

    print("loss-selected score stats:")
    print("  positive entries:", positive_scores.numel())
    print("  negative entries:", negative_scores.numel())

    if positive_scores.numel() > 0:
        positive_sigmoid = torch.sigmoid(positive_scores)
        print("  positive logits min:", positive_scores.min().item())
        print("  positive logits max:", positive_scores.max().item())
        print("  positive logits mean:", positive_scores.mean().item())
        print("  positive logits std:", positive_scores.std().item())
        print("  positive sigmoid min:", positive_sigmoid.min().item())
        print("  positive sigmoid max:", positive_sigmoid.max().item())
        print("  positive sigmoid mean:", positive_sigmoid.mean().item())
        print("  positive sigmoid std:", positive_sigmoid.std().item())

    if negative_scores.numel() > 0:
        negative_sigmoid = torch.sigmoid(negative_scores)
        print("  negative logits min:", negative_scores.min().item())
        print("  negative logits max:", negative_scores.max().item())
        print("  negative logits mean:", negative_scores.mean().item())
        print("  negative logits std:", negative_scores.std().item())
        print("  negative sigmoid min:", negative_sigmoid.min().item())
        print("  negative sigmoid max:", negative_sigmoid.max().item())
        print("  negative sigmoid mean:", negative_sigmoid.mean().item())
        print("  negative sigmoid std:", negative_sigmoid.std().item())

    if positive_scores.numel() > 0 and negative_scores.numel() > 0:
        gap = (
            torch.sigmoid(positive_scores).mean().item()
            - torch.sigmoid(negative_scores).mean().item()
        )
        print("  sigmoid gap positive-minus-negative:", gap)


def print_gradient_debug(model):
    print("Gradient check:")

    for name, param in model.named_parameters():
        if "attention" in name.lower() or "matching" in name.lower():
            if param.grad is None:
                print(name, "grad: None")
            else:
                print(name, "grad norm:", param.grad.detach().norm().item())


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

def main():
    cfg = SeedConfig()

    cfg.device_acc = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.input_tensor_path = "/gpfs/workdir/thibauts/tensor_output_shuffled"
    cfg.dataset_name = "seeding_data_MH5000"

    cfg.embedding_feature = [0, 1, 2]
    cfg.high_level_features = []
    cfg.cosine_processing = []

    cfg.fourier_num_frequencies = [10, 10, 10]

   
    # raw coordinates are on the scale of hundreds
    # test [500.0, 500.0, 500.0].
    cfg.dim_max = [1.0, 1.0, 1.0]

    cfg.shift = [0.0, 0.0, 0.0]

   
    cfg.dim_embedding = 128
    cfg.nb_layers_t = 2
    cfg.feed_forward_ratio = 4
    cfg.nb_heads = 4
    cfg.dropout = 0.1
    cfg.regression = False
    
    num_steps = 1000
    overfit_file_idx = 0
    overfit_event_idx = 0

    max_positive_pairs = 2000
    learning_rate = 1e-3
    weight_decay = 1e-2

    print_every = 50

    checkpoint_dir = "/gpfs/workdir/thibauts/dune_overfit_one_event_checkpoints"
    attention_plot_dir = "/gpfs/workdir/thibauts/dune_overfit_one_event_attention_plots"

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

    if overfit_file_idx < 0 or overfit_file_idx >= num_files:
        raise ValueError(
            f"overfit_file_idx={overfit_file_idx} is invalid. "
            f"Dataset has num_files={num_files}."
        )

    model = SeedTransformer(cfg).to(cfg.device_acc)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print()
    print("One-event overfit settings:")
    print("num_steps:", num_steps)
    print("overfit_file_idx:", overfit_file_idx)
    print("overfit_event_idx:", overfit_event_idx)
    print("max_positive_pairs:", max_positive_pairs)
    print("learning_rate:", learning_rate)
    print("weight_decay:", weight_decay)
    print("checkpoint_dir:", checkpoint_dir)
    print("attention_plot_dir:", attention_plot_dir)
    print("hard-negative curriculum:")
    print("  steps 0-499:   hard_negative_fraction = 0.0")
    print("  steps 500-999: hard_negative_fraction = 0.20")

    if cfg.device_acc.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print()
    print("=" * 80)
    print("Loading one file for overfit test")
    print("=" * 80)

    file_data = dataset.get_file(overfit_file_idx)

    hits_tensor = file_data["hits_tensor"]
    padding_mask = file_data["padding_mask"]
    hit_to_particle_tensor = file_data["hit_to_particle_tensor"]

    print("Loaded file on CPU:")
    print("hits_tensor:", hits_tensor.shape, hits_tensor.device)
    print("padding_mask:", padding_mask.shape, padding_mask.device)
    print("hit_to_particle_tensor:", hit_to_particle_tensor.shape, hit_to_particle_tensor.device)

    n_events = hits_tensor.shape[0]

    if overfit_event_idx < 0 or overfit_event_idx >= n_events:
        raise ValueError(
            f"overfit_event_idx={overfit_event_idx} is invalid. "
            f"Selected file has n_events={n_events}."
        )

    batched_hits_cpu = hits_tensor[overfit_event_idx]                  # [1, 5000, 3]
    batched_mask_cpu = padding_mask[overfit_event_idx]                 # [1, 5000]
    particle_ids_cpu = hit_to_particle_tensor[overfit_event_idx, 0]    # [5000, 1]

    print()
    print("Selected overfit event:")
    print("batched_hits_cpu:", batched_hits_cpu.shape)
    print("batched_mask_cpu:", batched_mask_cpu.shape)
    print("particle_ids_cpu:", particle_ids_cpu.shape)

    real_hit_mask_cpu = ~batched_mask_cpu[0]
    real_hits_cpu = batched_hits_cpu[0, real_hit_mask_cpu]

    print("real hits:", real_hits_cpu.shape[0])

    if real_hits_cpu.shape[0] > 0:
        print(
            "x min/max/mean/std:",
            real_hits_cpu[:, 0].min().item(),
            real_hits_cpu[:, 0].max().item(),
            real_hits_cpu[:, 0].mean().item(),
            real_hits_cpu[:, 0].std().item(),
        )
        print(
            "y min/max/mean/std:",
            real_hits_cpu[:, 1].min().item(),
            real_hits_cpu[:, 1].max().item(),
            real_hits_cpu[:, 1].mean().item(),
            real_hits_cpu[:, 1].std().item(),
        )
        print(
            "z min/max/mean/std:",
            real_hits_cpu[:, 2].min().item(),
            real_hits_cpu[:, 2].max().item(),
            real_hits_cpu[:, 2].mean().item(),
            real_hits_cpu[:, 2].std().item(),
        )

    batched_hits = batched_hits_cpu.to(
        cfg.device_acc,
        dtype=model.dtype,
    )
    batched_mask = batched_mask_cpu.to(cfg.device_acc)

    loss_history = []
    hard_negative_fraction_history = []

    print()
    print("=" * 80)
    print("Starting one-event overfit training")
    print("=" * 80)

    for step in range(num_steps):
        pairs1_cpu, pairs2_cpu, target_cpu = sample_positive_pairs_from_particle_ids(
            particle_ids_cpu,
            max_positive_pairs=max_positive_pairs,
        )

        if pairs1_cpu.numel() == 0:
            raise RuntimeError("The selected overfit event has no positive pairs.")

        if step == 0:
            pid_flat = particle_ids_cpu.squeeze(-1)

            same_pid_fraction = (
                pid_flat[pairs1_cpu] == pid_flat[pairs2_cpu]
            ).float().mean().item()

            print("Positive pair same-particle fraction:", same_pid_fraction)
            print("pairs1 min/max:", pairs1_cpu.min().item(), pairs1_cpu.max().item())
            print("pairs2 min/max:", pairs2_cpu.min().item(), pairs2_cpu.max().item())

            if same_pid_fraction != 1.0:
                raise RuntimeError(
                    "Positive pair sampler sanity check failed: "
                    "not all sampled positive pairs have the same particle_id."
                )

        pairs1 = pairs1_cpu.to(cfg.device_acc).long()
        pairs2 = pairs2_cpu.to(cfg.device_acc).long()
        target = target_cpu.to(cfg.device_acc).float()

        if step < 500:
            hard_negative_fraction = 0.0
        else:
            hard_negative_fraction = 0.20

        hard_negative_fraction_history.append(hard_negative_fraction)

        optimizer.zero_grad(set_to_none=True)

        encoded_space_points, attention_weights = model(
            batched_hits,
            batched_mask,
        )

        attention_map = attention_weights

        if attention_map.dim() == 3:
            attention_map = attention_map[0]

        if step % print_every == 0:
            loss, loss_debug = top_attention_loss(
                attention_map,
                pairs1,
                pairs2,
                target,
                return_debug=True,
                hard_negative_fraction=hard_negative_fraction,
            )
        else:
            loss = top_attention_loss(
                attention_map,
                pairs1,
                pairs2,
                target,
                hard_negative_fraction=hard_negative_fraction,
            )
            loss_debug = None

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step={step}: {loss.item()}"
            )

        loss.backward()

        if step % print_every == 0:
            print()
            print("-" * 80)
            print(f"overfit step={step} | loss={loss.item()}")
            print("hard_negative_fraction:", hard_negative_fraction)

            print_attention_debug(
                attention_map=attention_map,
                batched_mask=batched_mask,
                pairs1=pairs1,
                pairs2=pairs2,
            )

            print_loss_debug(loss_debug)

            print_gradient_debug(model)

            if cfg.device_acc.type == "cuda":
                print(
                    "current memory GB:",
                    torch.cuda.memory_allocated() / 1024**3,
                )
                print(
                    "peak memory GB:",
                    torch.cuda.max_memory_allocated() / 1024**3,
                )

        optimizer.step()

        loss_history.append(loss.item())

    print()
    print("=" * 80)
    print("One-event overfit summary")
    print("num_steps:", num_steps)
    print("first loss:", loss_history[0])
    print("last loss:", loss_history[-1])
    print("min loss:", min(loss_history))
    print("max loss:", max(loss_history))
    print("first hard_negative_fraction:", hard_negative_fraction_history[0])
    print("last hard_negative_fraction:", hard_negative_fraction_history[-1])

    if cfg.device_acc.type == "cuda":
        print(
            "peak memory GB:",
            torch.cuda.max_memory_allocated() / 1024**3,
        )

    print("=" * 80)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        (
            f"dune_seed_transformer_overfit"
            f"_file{overfit_file_idx}"
            f"_event{overfit_event_idx}"
            f"_steps{num_steps}"
            f"_curriculum_random_then_hard005.pt"
        ),
    )

    torch.save(
        {
            "num_steps": num_steps,
            "overfit_file_idx": overfit_file_idx,
            "overfit_event_idx": overfit_event_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "hard_negative_fraction_history": hard_negative_fraction_history,
            "cfg": cfg,
        },
        checkpoint_path,
    )

    print("Saved overfit checkpoint:", checkpoint_path)

    del file_data
    del hits_tensor
    del padding_mask
    del hit_to_particle_tensor

    if cfg.device_acc.type == "cuda":
        torch.cuda.empty_cache()

    save_attention_heatmap(
        model=model,
        dataset=dataset,
        cfg=cfg,
        file_idx=overfit_file_idx,
        event_idx=overfit_event_idx,
        output_dir=attention_plot_dir,
        max_plot_hits=500,
    )

    print()
    print("One-event overfit training finished successfully.")


if __name__ == "__main__":
    main()
