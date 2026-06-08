import random
import sys
import os
import math
import torch
import time
import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from GUNTAM.Seed.SeedTransformer import SeedTransformer
import GUNTAM.Seed.SeedLoss as Losses
from GUNTAM.Seed.Config import SeedConfig
from GUNTAM.IO.DataLoader import DataLoader
from GUNTAM.Transformer.Utils import ts_print
import GUNTAM.Transformer.Utils as Utils
import GUNTAM.Seed.Reconstruction as Reconstruction
from GUNTAM.Seed.Monitoring import PerformanceMonitor
from GUNTAM.IO.PrepareTensor import compute_barcode, prepare_tensor, sample_positive_pairs_from_particle_ids


def initialize_loss_dictionary(active_components: list, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Initialize a loss dictionary with zero values for active loss components.

    Args:
        active_components: List of active loss component names.
        device: Torch device for tensor initialization.

    Returns:
        Initialized loss dictionary with zero values.
    """

    # Helper to add a key lazily
    def add_loss_key(key: str):
        if key not in event_loss_log:
            event_loss_log[key] = torch.tensor(0.0, device=device)

    # Initialize per-event losses dynamically based on active loss components
    event_loss_log = {"total": torch.tensor(0.0, device=device)}

    # Attention variants
    if "attention" in active_components:
        add_loss_key("attention")
    if "topk_attention" in active_components:
        add_loss_key("topk_attention")
    if "full_attention" in active_components:
        add_loss_key("full_attention")
    if "attention_next" in active_components:
        add_loss_key("attention_next")
    if "attention_back" in active_components:
        add_loss_key("attention_back")

    # Classification losses
    if "hit_BCE" in active_components:
        add_loss_key("hit_BCE")

    return event_loss_log


def train_model(
    model: SeedTransformer,
    train_file_indices: list,
    dataset: DataLoader,
    nb_events: int, 
    cfg: SeedConfig,
    writer: SummaryWriter,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    start_epoch: int = 0,
) -> SeedTransformer:
    """
    Train the transformer model for seed reconstruction.

    Args:
        model: The transformer model to be trained.
        train_file_indices: List of file indices for training data.
        dataset: The dataset object containing training data.
        nb_events: Number of events per file.
        cfg: Configuration object with training parameters.
        writer: TensorBoard writer for logging.
        optimiser: Optimizer for training.
        scheduler: Learning rate scheduler.
        start_epoch: Starting epoch number (default: 0).

    Returns:
        The trained transformer model.
    """
    
    epoch_nb = cfg.epoch_nb

    # Loop over the number of epoch starting from start_epoch
    ts_print("Starting the training of the transformer model for seed reconstruction")
    ts_print("Train from epoch ", start_epoch, " to ", start_epoch + epoch_nb)

    # Print active loss components
    active_losses = []
    for component, weight in cfg.loss_config.items():
        active_losses.append(f"{component} (weight: {weight})")

    ts_print("Active loss components: " + ", ".join(active_losses))

    if optimiser is not None and scheduler is not None: 
        print(f"Initial learning rate: {scheduler.get_last_lr()}")

    for epoch in range(start_epoch, start_epoch + epoch_nb):
        ts_print("Epoch: ", epoch)
        entry = 0 # counter increases for every processed event

        # Track epoch-level losses
        epoch_train_losses = []
        epoch_val_losses = []

        # Deterministic split of files into training/validation sets (no shuffling), validation may be empty
        files = list(train_file_indices)
        n_val_files = int(cfg.val_fraction * len(files)) if hasattr(cfg, "val_fraction") else 0
        val_files_set = set(files[-n_val_files:]) if n_val_files > 0 else set()

        for file_idx in files: # the tensor_data_i_MH5000.pt files
            # Decide status per file to keep train/val files separate
            status = "Validation" if file_idx in val_files_set else "Training"
            if status == "Validation":
                model.eval()
            else:
                model.train()

            model_dtype = model.dtype
            # Load the data
            batch_data = dataset.get_file(file_idx)
            
            # Checking files are on CPU
            # print("hits_tensor device:", batch_data["hits_tensor"].device)
            # print("hit_to_particle_tensor device:", batch_data["hit_to_particle_tensor"].device)
            # print("padding_mask device:", batch_data["padding_mask"].device)

            hits_tensor = batch_data["hits_tensor"] # we don't move the files to GPU, they stay on CPU, only the current event will be moved to GPU
            hit_to_particle_tensor = batch_data["hit_to_particle_tensor"]
            padding_mask = batch_data["padding_mask"]
            
            # Iterate through each event in this batch with a random order
            num_events_in_batch = hits_tensor.shape[0]
            event_indices = list(range(num_events_in_batch))
            random.shuffle(event_indices) # training does not see events in the same order every epoch

            for event_idx in event_indices:
                grad_enabled = status == "Training"
                event_loss_log = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc) # a container for numbers, not part of the computation graph

                with torch.set_grad_enabled(grad_enabled):
                    batched_hits = hits_tensor[event_idx].to(cfg.device_acc) # [num_bin=1, max_hit_input, num_hit_features]
                    event_particle_ids = hit_to_particle_tensor[event_idx].to(cfg.device_acc)  # [num_bin, max_hit_input, 1]
                    batched_mask = padding_mask[event_idx].to(cfg.device_acc) # [num_bin, max_hit_input]
                    particle_ids_bin = event_particle_ids[0, :, 0] # [max_hit_input]
                    
                    pairs1, pairs2, target = sample_positive_pairs_from_particle_ids(particle_ids_bin, max_positive_pairs=20_000) # try 100_000
                    
                    if pairs1.numel() == 0:
                        continue
                    
                    event_loss_terms = initialize_loss_dictionary(list(cfg.loss_config.keys()), cfg.device_acc)
                    
                    if status == "Training":
                        optimiser.zero_grad()
                    
                    encoded_space_points, attention_maps = model(
                        batched_hits,
                        batched_mask,
                    )
                    
                    attention_map_bin = attention_maps[0] # shape [max_hit_input, max_hit_input]
                    
                    if attention_map_bin.dim() == 3:
                        attention_map_bin = attention_map_bin.squeeze(0)
                    
                    if cfg.has_loss_component("attention"):
                        event_loss_terms["attention"] = Losses.attention_loss(
                            attention_map_bin,
                            pairs1,
                            pairs2,
                            target,
                        )
                        
                    if cfg.has_loss_component("full_attention"):
                        event_loss_terms["full_attention"] = Losses.full_attention_loss(
                            attention_map_bin,
                            pairs1,
                            pairs2,
                            target,
                        )
                        
                    if cfg.has_loss_component("topk_attention"):
                        event_loss_terms["topk_attention"] = Losses.top_attention_loss(
                            attention_map_bin,
                            pairs1,
                            pairs2,
                            target,
                        )
                        
                    if cfg.has_loss_component("attention_next"):
                        event_loss_terms["attention_next"] = Losses.attention_next_loss(
                            attention_map_bin,
                            pairs1,
                            pairs2,
                            target,
                        )

                    if cfg.has_loss_component("attention_back"):
                        event_loss_terms["attention_back"] = Losses.attention_backward_loss(
                            attention_map_bin,
                            pairs1,
                            pairs2,
                            target,
                        )

                    total_loss = torch.tensor(
                        0.0, # scalar tensor
                        device=cfg.device_acc,
                        dtype=model_dtype,
                    )
                
                    for key, value in event_loss_terms.items():
                        if key == "total":
                            continue

                        event_loss_log[key] += value.detach()

                        weighted_value = cfg.get_loss_weight(key) * value
                        event_loss_log["total"] += weighted_value.detach()

                    if status == "Training":
                        total_loss = total_loss + weighted_value

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        raise ValueError(
                            f"Loss became NaN/Inf at epoch {epoch}, "
                            f"file_idx {file_idx}, event_idx {event_idx}"
                        )
                    
                    if status == "Training":
                        total_loss.backward()
                        optimiser.step()
                        
                        if scheduler is not None:
                            scheduler.step()
                
                if writer:
                    step = epoch * nb_events + entry

                    for key, value in event_loss_log.items():
                        writer.add_scalar(
                            f"loss_components/{key}/{status}",
                            value.item(),
                            step,
                        )

                    if optimiser is not None:
                        writer.add_scalar(
                            f"learning_rate/{status}",
                            optimiser.param_groups[0]["lr"],
                            step,
                        )
                    
                if status == "Training":
                    epoch_train_losses.append(event_loss_log["total"].item())
                else:
                    epoch_val_losses.append(event_loss_log["total"].item())

                entry += 1

        if epoch_train_losses: # if list is not empty
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            ts_print(
                f"Epoch {epoch} - Average Training Loss: "
                f"{avg_train_loss:.6f} ({len(epoch_train_losses)} events)"
            )

            if writer:
                writer.add_scalar("loss_epoch/Training", avg_train_loss, epoch)
            
        if epoch_val_losses:
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            ts_print(
                f"Epoch {epoch} - Average Validation Loss: "
                f"{avg_val_loss:.6f} ({len(epoch_val_losses)} events)"
            )

            if writer:
                writer.add_scalar("loss_epoch/Validation", avg_val_loss, epoch)           
        
        if (epoch + 1) % 10 == 0:
            backup_path = cfg.model_path.replace(
                ".pt",
                f"_backup_epoch_{epoch + 1}.pt",
            )

            model.save(
                epoch=epoch,
                path=backup_path,
                optimizer=optimiser,
                scheduler=scheduler,
            )

            print(f"Saved backup checkpoint to {backup_path}")           
                    
    return model


def run_model(
    model: SeedTransformer,
    hits_tensor: torch.Tensor,
    padding_mask: torch.Tensor,
    hit_to_particle_tensor: torch.Tensor,
    cfg: SeedConfig,
) -> tuple:
    """Run inference for one event and reconstruct seeds using top-k attention.

    Expected input shapes for one event:
    
        hits_tensor:
            [1, max_hit_input, num_features]
            
        padding_mask:
            [1, max_hit_input]
            True = padding
            False = not padding
        
        hit_to_particle_tensor:
            [1, max_hit_input, 1] or [1, max_hit_input]
            particle_id >= 0 : real particle hit
            particle_id == -1: orphan hit
            particle_id == -2: padding hit
        
    Returns:
        - event_seeds: list indexed by bin index. Since there is only one bin per event, we only get event_seeds[0].
        - event_hit_scores: list indexed by bin index. In Dune case, this is None.
        - event_attention_maps: list indexed by bin index. Stores the sigmoid attention map restricted to real particle hits.
        - timing values: event_duration, transformer_duration, regression_duration, seed_reconstruction_duration.
    """
    
    
    event_duration = 0.0
    transformer_duration = 0.0
    regression_duration = 0.0
    seed_reconstruction_duration = 0.0

    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        event_start_time = time.perf_counter()

    event_seeds: List[Any] = []
    event_hit_scores: List[Optional[np.ndarray]] = []
    event_attention_maps: List[Optional[np.ndarray]] = []

    # Safety checks
    if hits_tensor.dim() != 3:
        raise ValueError(
            f"Expected hits_tensor with shape [1, max_hit_input, num_features], "
            f"but got shape {tuple(hits_tensor.shape)}"
        )
        
    if padding_mask.dim() != 2:
        raise ValueError(
            f"Expected padding_mask with shape [1, max_hit_input],"
            f"but got shape {tuple(padding_mask.shape)}"
        )
        
    if hit_to_particle_tensor.dim() == 3:
        hit_to_particle_flat = hit_to_particle_tensor[..., 0] # converts [1, max_hit_input, 1] to [1, max_hit_input]
    elif hit_to_particle_tensor.dim() == 2:
        hit_to_particle_flat = hit_to_particle_tensor
    else:
        raise ValueError(
            f"Expected hit_to_particle_tensor with shape [1, max_hit_input, 1],"
            f"or [1, max_hit_input], but got shape {tuple(hit_to_particle_tensor.shape)}"
        )
    
    with torch.inference_mode():
        # Timing: Transformer inference (encoding + attention)
        if cfg.timing_enabled:
            Utils.sync_device(cfg.device_acc)
            t0 = time.perf_counter()

        hits_tensor_gpu = hits_tensor.to(cfg.device_acc, dtype=model.dtype)
        padding_mask_gpu = padding_mask.to(cfg.device_acc, dtype=torch.bool)

        # Obtain encoded embeddings and attention weights from the model
        encoded_space_point, attention_weights = model(hits_tensor_gpu, padding_mask_gpu)

        if cfg.timing_enabled:
            Utils.sync_device(cfg.device_acc)
            transformer_duration = time.perf_counter() - t0

        # Timing: Parameter regression (+ optional pairwise scoring)
        if cfg.timing_enabled:
            Utils.sync_device(cfg.device_acc)
            r0 = time.perf_counter()

        hit_score = None

        if cfg.timing_enabled:
            Utils.sync_device(cfg.device_acc)
            regression_duration = time.perf_counter() - r0

    # Timing: Seed reconstruction across all bins
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        seed_reconstruction_start = time.perf_counter()

    padding_mask_cpu = padding_mask.detach().cpu().bool()
    hit_to_particle_cpu = hit_to_particle_flat.detach().cpu()
    attention_weights_cpu = attention_weights.detach().cpu()
    
    num_bins = hits_tensor.shape[0] # 1
    
    for bin_idx in range(num_bins):
        padding_bin = padding_mask_cpu[bin_idx]
        particle_bin = hit_to_particle_cpu[bin_idx]
        
        valid_hit_mask = (~padding_bin) & (particle_bin >= 0)
        
        if not valid_hit_mask.any():
            event_seeds.append([])
            event_hit_scores.append(None)
            event_attention_maps.append(None)
            continue
        
        attention_bin = attention_weights_cpu[bin_idx]
        
        attention_bin = attention_bin.squeeze()
        
        if attention_bin.dim() == 3:
            attention_bin = attention_bin.mean(dim=0)
        
        if attention_bin.dim() != 2:
            raise ValueError(
                f"Expected attention_bin to become 2D, but got shape "
                f"{tuple(attention_bin.shape)}"
            )
            
        neighbor_matrix = torch.sigmoid(attention_bin)
        
        max_selection = getattr(cfg, "max_selection", 5)
        
        bin_seeds = topk_seed_reconstruction(
            attention_map=neighbor_matrix, # the attention matrix after sigmoid
            hit_to_particle=particle_bin, # shape [5000]
            max_selection=max_selection, # how many top neighbors to select for each hit
        )
        
        event_seeds.append(bin_seeds)
        event_hit_scores.append(None)
        attention_valid = neighbor_matrix[valid_hit_mask, :][:, valid_hit_mask]
        
        event_attention_maps.append(
            attention_valid.detach().float().numpy()
        )
    
    if cfg.timing_enabled:
        Utils.sync_device(cfg.device_acc)
        seed_reconstruction_duration = (
            time.perf_counter() - seed_reconstruction_start
        )
        event_duration = time.perf_counter() - event_start_time
        
    return (
        event_seeds,
        event_hit_scores,
        event_attention_maps,
        event_duration,
        transformer_duration,
        regression_duration,
        seed_reconstruction_duration,
    )
    

def main():
    """
    Main function to run the training of the transformer model for seed reconstruction
    """
    # Parse the command line argument
    cfg = SeedConfig()
    cfg.parse_args()

    # Print starting information
    ts_print("Starting the training of the transformer model for seed reconstruction")
    cfg.print_config()
    ts_print(f"Using device: {cfg.device_acc}")

    # If on CUDA, set matmul precision to high for potential speedup (requires PyTorch 2.0+ and compatible hardware)
    if cfg.device_acc.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # TODO: make into an optional argument
    log_dir = "training_seeding"

    # Create the model using configuration parameters
    model = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=cfg.device_acc,
        dtype=cfg.transformer_config.dtype,
    )
    model.to(cfg.device_acc)
    # Create optimizer right now we are using AdamW as it seems to perform well with transformer models
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # Create learning rate scheduler with cosine annealing and minimum learning rate
    scheduler = Utils.create_cosine_schedule_with_min_lr(
        opt,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=cfg.num_training_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )

    start_epoch = 0

    # Check if the metadata file exist
    barcode = compute_barcode(cfg.preprocessing_config)
    metadata_path = f"{cfg.input_tensor_path}/metadata_{cfg.dataset_name}_{barcode}.pt"
    if not os.path.exists(metadata_path) or cfg.recompute_tensor:
        ts_print("Computing new tensor dataset from input files with barcode: ", barcode)
        # Set hit/particle features in preprocessing config
        cfg.preprocessing_config.hit_features = ["x", "y", "z", "r", "phi", "eta"]
        cfg.preprocessing_config.particle_features = ["d0", "z0", "phi", "eta", "pT", "q", "m"]
        # Pass preprocessing_config to prepare_tensor
        prepare_tensor(cfg.preprocessing_config)

    ts_print("Loading existing Seeding Dataset from disk")
    # Load the existing dataset (force_recreate=False will load existing files)
    tensor_list = {
        "hits_tensor",
        "hit_to_particle_tensor",
        "padding_mask",
    }
    dataset = DataLoader(
        dataset_dir=cfg.input_tensor_path,
        dataset_name=f"{cfg.dataset_name}_{barcode}",
        tensor_names=list(tensor_list),
        device=cfg.device_acc,
    )

    # Split the dataset at the file level to ensure proper train/test separation
    num_files = dataset.get_file_number()
    test_fraction = cfg.test_fraction
    train_files = math.ceil((1 - test_fraction) * num_files)

    ts_print(f"Dataset has {num_files} files, using {train_files} for training and {num_files - train_files} for testing")

    # Create file-based train and test indices
    train_file_indices = list(range(train_files))
    test_file_indices = list(range(train_files, num_files))
    
    if train_files == 0:
        raise ValueError("No training files selected. Reduce cfg.test_fraction.")

    
    train_size = dataset.get_batch_size(0, train_files - 1)

    # Keep training file order deterministic (no shuffling)
    if cfg.epoch_nb > 0:

        # Resume training from existing model/checkpoint if specified
        if cfg.resume_training:
            ts_print(f"Resuming training from {cfg.model_path}...")
            # Check if a checkpoint file exists
            start_epoch = model.load(
                path=cfg.model_path,
                device=cfg.device_acc,
                optimizer=opt,
                scheduler=scheduler,
            )
            # check if the model dtype matches the config, print a warning if not
            if model.dtype != cfg.transformer_config.dtype:
                ts_print(f"Warning: Loaded model dtype {model.dtype} does not match config dtype {cfg.dtype}.")

            # Load previous tensorboard logs
            if os.path.exists(log_dir):
                # Append to existing log dir so TensorBoard combines all runs
                writer = SummaryWriter(log_dir=log_dir)
            else:
                ts_print(f"Warning: TensorBoard log directory {log_dir} not found. Creating new logs.")
                writer = SummaryWriter(log_dir)
        else:
            # If not resuming, create new TensorBoard writer
            writer = SummaryWriter(log_dir)

        # Print model summary
        model.print_model_info()

        # Calculate the total number of training events across all training files
        ts_print(f"Training on {train_size} events across {len(train_file_indices)} files")
        # Very important: compile the model
        # if cfg.device_acc.type != "mps":
        #     model = torch.compile(model)
        # Train the model
        ts_print("Starting training of the model")
        model = train_model(
            model,
            train_file_indices,
            dataset,
            train_size,
            cfg,
            writer,
            opt,
            scheduler,
            start_epoch=start_epoch,
        )
        ts_print("Training completed")

        # Save the model with full state including architecture parameters
        model.save(
            epoch=start_epoch + cfg.epoch_nb - 1,
            path=cfg.model_path,
            optimizer=opt,
            scheduler=scheduler,
        )

        # Delete training variables to free some memory
        del model
        del opt
        del scheduler
        del train_file_indices
        torch.cuda.empty_cache()

        writer.close()

    # Load model configuration from checkpoint or use config defaults
    model_val = SeedTransformer(
        transformer_config=cfg.transformer_config,
        device_acc=cfg.device_acc,
        dtype=cfg.transformer_config.dtype,
    )

    if cfg.no_test or len(test_file_indices) == 0:
        ts_print("Skipping evaluation")
        return

    model_val.load(
        path=cfg.model_path,
        device=cfg.device_acc,
    )
    model_val.to(cfg.device_acc)
    model_val = model_val.to(cfg.transformer_config.dtype)
    model_val.eval()
    # if cfg.device_acc.type != "mps":
    #     model_val = torch.compile(model_val)

    # Perform validation using test_file_indices
    print("Starting model evaluation with test dataset...")
    processing_times = []
    transformer_times = []
    regression_times = []
    seed_reconstruction_times = []
    total_time = 0.0
    start_event = 0
    event_counter = 0

    # monitoring = PerformanceMonitor(
    #     full_print=False,
    #     save_plots=True,
    #     min_common_hits=3,
    #     min_truth_hits=3,
    #     truth_r_tol=1e-3,
    # )

    for file_idx in test_file_indices:
        batch_data = dataset.get_file(file_idx)
        end_event = start_event + dataset.get_batch_size(file_idx, file_idx)
        event_counter += end_event - start_event
        for event in range(start_event, end_event):
            event_idx = event - start_event
            start_time = time.perf_counter() if cfg.timing_enabled else None
            (
                batch_seeds,
                batch_hit_scores,
                batch_attention_maps,
                batch_processing_times,
                batch_transformer_times,
                batch_regression_times,
                batch_seed_reconstruction_times,
            ) = run_model(model_val, batch_data["hits_tensor"][event_idx], batch_data["padding_mask"][event_idx], batch_data["hit_to_particle_tensor"][event_idx], cfg)
            total_time += time.perf_counter() - start_time if cfg.timing_enabled else 0.0

            # Accumulate timings for speed summary
            if cfg.timing_enabled:
                processing_times.append(batch_processing_times)
                transformer_times.append(batch_transformer_times)
                regression_times.append(batch_regression_times)
                seed_reconstruction_times.append(batch_seed_reconstruction_times)
                del batch_processing_times
                del batch_transformer_times
                del batch_regression_times
                del batch_seed_reconstruction_times


        start_event = end_event

    print("Model evaluation completed.")
    #monitoring.performance_analysis()

    # Final speed summary with component breakdown
    if cfg.timing_enabled:
        total_time
        Utils.sync_device(cfg.device_acc)
        avg_time_per_event = np.mean(processing_times)
        std_time_per_event = np.std(processing_times)
        min_time_per_event = np.min(processing_times)
        max_time_per_event = np.max(processing_times)

        # Component-wise statistics
        avg_transformer_time = np.mean(transformer_times)
        std_transformer_time = np.std(transformer_times)
        avg_regression_time = np.mean(regression_times)
        std_regression_time = np.std(regression_times)
        avg_seed_reconstruction_time = np.mean(seed_reconstruction_times)
        std_seed_reconstruction_time = np.std(seed_reconstruction_times)

        print("\n" + "=" * 80)
        print("PROCESSING SPEED SUMMARY")
        print("=" * 80)
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Number of events processed: {event_counter}")
        print(f"Average events per second: {event_counter / total_time:.2f}")
        print(f"Average events per minute: {event_counter / total_time * 60:.1f}")
        print()
        print("COMPONENT BREAKDOWN:")
        print("-" * 50)
        print(f"{'Component':<20} {'Avg Time':<12} {'Std':<8} {'% of Total':<12}")
        print("-" * 50)
        print(f"{'Total per event:':<20} {avg_time_per_event:.3f}s{'':<4} {std_time_per_event:.3f}s {'100.0%':<12}")
        print(
            f"{'Transformer:':<20} {avg_transformer_time:.3f}s{'':<4} {std_transformer_time:.3f}s "
            f"{avg_transformer_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print(
            f"{'Regression:':<20} {avg_regression_time:.3f}s{'':<4} {std_regression_time:.3f}s "
            f"{avg_regression_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print(
            f"{'Seed reconstruction:':<20} {avg_seed_reconstruction_time:.3f}s{'':<4}"
            f"{std_seed_reconstruction_time:.3f}s "
            f"{avg_seed_reconstruction_time / avg_time_per_event * 100:.1f}%{'':<4}"
        )
        print("-" * 50)
        print(f"Min/Max time per event: {min_time_per_event:.3f}s / {max_time_per_event:.3f}s")


if __name__ == "__main__":
    sys.exit(main())
