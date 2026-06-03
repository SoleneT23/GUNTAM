import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Seed.SeedTransformer import SeedTransformer
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.Seed.SeedLoss import top_attention_loss
from GUNTAM.IO.PrepareTensor import sample_positive_pairs_from_particle_ids


def get_hard_negative_fraction(epoch: int) -> float:
    if epoch < 3:
        return 0.0
    if epoch < 20:
        return 0.1
    return 0.2


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

    batched_hits_cpu = hits_tensor[event_idx]
    batched_mask_cpu = padding_mask[event_idx]

    batched_hits = batched_hits_cpu.to(cfg.device_acc, dtype=model.dtype)
    batched_mask = batched_mask_cpu.to(cfg.device_acc)

    with torch.no_grad():
        _, attention_weights = model(
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
            f"final_attention_heatmap"
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
            f"Final attention heatmap | file {file_idx}, "
            f"event {event_idx} | first {len(real_indices)} real hits"
        )
    )
    plt.xlabel("Hit index")
    plt.ylabel("Hit index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved final attention heatmap:", output_path)

    del file_data
    del hits_tensor
    del padding_mask

    if cfg.device_acc.type == "cuda":
        torch.cuda.empty_cache()


def print_attention_debug(attention_map, batched_mask, pairs1, pairs2):
    print("raw attention logits stats on real-hit square:")

    real_hit_mask = ~batched_mask[0]
    real_attention = attention_map[real_hit_mask][:, real_hit_mask]

    finite_mask = torch.isfinite(real_attention)
    finite_scores = real_attention[finite_mask]

    print("  real hits:", real_hit_mask.sum().item())
    print("  finite entries:", finite_scores.numel())

    if finite_scores.numel() > 0:
        print("  min:", finite_scores.detach().min().item())
        print("  max:", finite_scores.detach().max().item())
        print("  mean:", finite_scores.detach().mean().item())
        print("  std:", finite_scores.detach().std().item())

        sig = torch.sigmoid(finite_scores.detach())
        print("sigmoid attention stats on real-hit square:")
        print("  min:", sig.min().item())
        print("  max:", sig.max().item())
        print("  mean:", sig.mean().item())
        print("  std:", sig.std().item())

    pos_debug_scores = attention_map[pairs1, pairs2]
    pos_debug_scores = pos_debug_scores[torch.isfinite(pos_debug_scores)]

    print("positive pair logits stats:")
    print("  positive entries:", pos_debug_scores.numel())

    if pos_debug_scores.numel() > 0:
        print("  min:", pos_debug_scores.detach().min().item())
        print("  max:", pos_debug_scores.detach().max().item())
        print("  mean:", pos_debug_scores.detach().mean().item())
        print("  std:", pos_debug_scores.detach().std().item())

        pos_sig = torch.sigmoid(pos_debug_scores.detach())
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

    if positive_scores.numel() > 0:
        print("  positive logits min:", positive_scores.min().item())
        print("  positive logits max:", positive_scores.max().item())
        print("  positive logits mean:", positive_scores.mean().item())
        print("  positive logits std:", positive_scores.std().item())

        positive_sigmoid = torch.sigmoid(positive_scores)
        print("  positive sigmoid mean:", positive_sigmoid.mean().item())
        print("  positive sigmoid std:", positive_sigmoid.std().item())

    print("  negative entries:", negative_scores.numel())

    if negative_scores.numel() > 0:
        print("  negative logits min:", negative_scores.min().item())
        print("  negative logits max:", negative_scores.max().item())
        print("  negative logits mean:", negative_scores.mean().item())
        print("  negative logits std:", negative_scores.std().item())

        negative_sigmoid = torch.sigmoid(negative_scores)
        print("  negative sigmoid mean:", negative_sigmoid.mean().item())
        print("  negative sigmoid std:", negative_sigmoid.std().item())

    if positive_scores.numel() > 0 and negative_scores.numel() > 0:
        print(
            "  sigmoid gap positive-minus-negative:",
            torch.sigmoid(positive_scores).mean().item()
            - torch.sigmoid(negative_scores).mean().item(),
        )


def print_gradient_debug(model):
    print("Gradient check:")

    for name, param in model.named_parameters():
        if "attention" in name.lower() or "matching" in name.lower():
            if param.grad is None:
                print(name, "grad: None")
            else:
                print(name, "grad norm:", param.grad.detach().norm().item())


def main():
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

    num_epochs = 30
    max_positive_pairs = 2000
    learning_rate = 1e-3
    weight_decay = 1e-2

    print_every = 50

    checkpoint_dir = "/gpfs/workdir/thibauts/dune_training_checkpoints_E10_P2000_lr1e-3_curriculum"
    attention_plot_dir = "/gpfs/workdir/thibauts/attention_plots_E10_P2000_lr1e-3_curriculum"

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
    print("hard negative phases:")
    print("  epochs 1-3: hard_negative_fraction = 0.0")
    print("  epochs 4-20: hard_negative_fraction = 0.1")
    print("  epochs 21-30: hard_negative_fraction = 0.2")

    global_step = 0

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
            print("hit_to_particle_tensor:", hit_to_particle_tensor.shape, hit_to_particle_tensor.device)

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

                if global_step % 100 == 0:
                    print_attention_debug(
                        attention_map=attention_map,
                        batched_mask=batched_mask,
                        pairs1=pairs1,
                        pairs2=pairs2,
                    )

                if global_step % 100 == 0:
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
                else:
                    loss = top_attention_loss(
                        attention_map,
                        pairs1,
                        pairs2,
                        target,
                        particle_ids,
                        batched_mask,
                        hard_negative_fraction=hard_negative_fraction,
                    )
                    loss_debug = None

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch}, "
                        f"file_idx={file_idx}, event_idx={event_idx}: {loss.item()}"
                    )

                if loss_debug is not None:
                    print("hard_negative_fraction:", hard_negative_fraction)
                    print_loss_debug(loss_debug)

                loss.backward()

                if global_step % 100 == 0:
                    print_gradient_debug(model)

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

        print()
        print("=" * 80)
        print(f"Epoch {epoch + 1} summary")
        print("hard_negative_fraction:", hard_negative_fraction)
        print("successful_events:", successful_events)
        print("skipped_events:", skipped_events)
        print("average_loss:", avg_loss)

        if cfg.device_acc.type == "cuda":
            print(
                "peak memory GB:",
                torch.cuda.max_memory_allocated() / 1024**3,
            )

        print("=" * 80)

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

    print()
    print("Full training finished successfully.")


if __name__ == "__main__":
    main()