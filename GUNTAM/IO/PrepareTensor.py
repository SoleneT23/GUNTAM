from typing import List, Tuple, Dict
import glob
import math
import multiprocessing
import os

import pandas as pd
import numpy as np
import torch

from GUNTAM.Transformer.BinTensor import no_bin
from GUNTAM.IO.PreprocessingConfig import PreprocessingConfig

import h5py


def _build_good_pairs_tensors(
    data_batch: pd.DataFrame,
    hit_to_particle: pd.Series,
) -> torch.Tensor:
    """
    Build a tensor of all hit pairs and their labels (same particle or not) for a batch of events,
    organized by bins.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events, must have 'event_id' column.
        hit_to_particle: Series containing the mapping from hits to particles, indexed the same as data_batch. It is given by the _process_single_batch() below.

    Returns:
        A PyTorch tensor of shape [num_events, num_bins=1, num_pairs, 3] where each pair is represented as
        (hit_idx1, hit_idx2, label) and label is 1 if the hits belong to the same particle.
    """
    print("    Building good pairs tensor (one bin per event)...")
    
    unique_events = sorted(data_batch["event_id"].unique())
    
    # Collect all pairs organized by event
    all_pairs_by_event: Dict[int, np.ndarray] = {}
    max_pairs = 0

    for event_id in unique_events:
        event_mask = data_batch["event_id"] == event_id
        event_hits = data_batch[event_mask] # subset of full dataframe, only one event
        
        event_particle_ids = hit_to_particle.loc[event_hits.index].to_numpy()
        n_hits = len(event_hits) # counts how many hits are in the event, defines how many possible pairs

        if n_hits > 0:
            i_indices = np.arange(n_hits)[:, None]
            j_indices = np.arange(n_hits)[None, :]

            not_self_mask = i_indices != j_indices
            same_particle_mask = event_particle_ids[i_indices] == event_particle_ids[j_indices] # builds a nhits x nhits boolean matrix

            # Exclude orphans (-1) and padding (-2)
            valid_particle_mask = (
                (event_particle_ids[i_indices] >= 0) &
                (event_particle_ids[j_indices] >= 0)
            )

            valid_mask = not_self_mask & same_particle_mask & valid_particle_mask
            
            i_valid, j_valid = np.where(valid_mask) # indices of valid pairs
            
            if len(i_valid) > 0:
                labels = np.ones(len(i_valid), dtype=np.int64)
                pairs = np.stack([i_valid, j_valid, labels], axis=1)
            else:
                pairs = np.empty((0, 3), dtype=np.int64)
        else:
            pairs = np.empty((0, 3), dtype=np.int64)
            
        all_pairs_by_event[event_id] = pairs
        max_pairs = max(max_pairs, len(pairs))
        
    num_events = len(unique_events)
    pairs_tensor = torch.zeros((num_events, 1, max_pairs, 3), dtype=torch.long) # num_bins = 1

    for event_idx, event_id in enumerate(unique_events):
        pairs = all_pairs_by_event[event_id]
        if len(pairs) > 0:
            pairs_tensor[event_idx, 0, :len(pairs), :] = torch.tensor(pairs, dtype=torch.long)

    return pairs_tensor


def build_positive_pairs_from_particle_ids(
    particle_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build all positive pairs from particle ids.
    
    Args:
        particle_ids: Tensor of shape [max_hit_input] or [max_hit_input, 1]
        
    Returns:
        pairs1: [N_pos]
        pairs2: [N_pos]
        target: [N_pos] all ones
    """
    if particle_ids.dim() == 2:
        particle_ids = particle_ids.squeeze(-1)
    
    device = particle_ids.device
    
    valid_mask = particle_ids >= 0
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1) # indices of valid real hits

    if valid_indices.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_float
    
    valid_particle_ids = particle_ids[valid_indices] 
    
    pairs1_list = []
    pairs2_list = []
    
    unique_particles = torch.unique(valid_particle_ids) # which particles exist in this event
    
    for pid in unique_particles:
        hit_indices = valid_indices[valid_particle_ids == pid]
        
        n = hit_indices.numel()
        
        if n < 2:
            continue
        
        # Generate every possible ordered pairs of hits for one particle
        i = hit_indices.repeat_interleave(n)
        j = hit_indices.repeat(n)
        
        not_self = i!=j
        
        pairs1_list.append(i[not_self])
        pairs2_list.append(j[not_self])
        
    if len(pairs1_list) == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_float
    
    pairs1 = torch.cat(pairs1_list)
    pairs2 = torch.cat(pairs2_list)
    target = torch.ones(pairs1_list.shape[0], dtype = torch.float32, device = device)
    
    return pairs1, pairs2, target
    
    
def sample_positive_pairs_from_particle_ids(
    particle_ids: torch.Tensor,
    max_positive_pairs: int = 100_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample positive pairs from particle ids.
    
    Args:
        particle_ids: Tensor of shape [max_hit_input] or [max_hit_input,1]
        max_positive_pairs: maximum number of positive pairs to return per event
        
    Returns:
        pairs1: [N_pos]
        pairs2: [N_pos]
        target: [N_pos], all ones 
        all tensors have size at most max_positive_pairs
    """
    if particle_ids.dim() == 2:
        particle_ids = particle_ids.squeeze(-1)
    
    device = particle_ids.device
    
    valid_mask = particle_ids >= 0
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    
    if valid_indices.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_float
        
    valid_particle_ids = particle_ids[valid_indices]
    unique_particles = torch.unique(valid_particle_ids)
    
    pairs1_list = []
    pairs2_list = []
    
    counts = []
    
    for pid in unique_particles:
        n = (valid_particle_ids == pid).sum().item()
        if n >= 2: 
            counts.append((pid, n, n*(n-1)))
        
    total_possible_pairs = sum(x[2] for x in counts)
    
    if total_possible_pairs == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_float
    
    for pid, n, num_pairs_pid in counts:
        hit_indices = valid_indices[valid_particle_ids == pid]
        
        k_pid = int(max_positive_pairs*num_pairs_pid / total_possible_pairs)
        k_pid = max(k_pid, 1)
        
        idx1 = torch.randint(0, n, (k_pid,), device=device)
        idx2 = torch.randint(0, n, (k_pid,), device=device)
        
        same = idx1 == idx2
        while same.any():
            idx2[same] = torch.randint(0, n, (same.sum().item(),), device=device)
            same = idx1 == idx2

        pairs1_list.append(hit_indices[idx1])
        pairs2_list.append(hit_indices[idx2])
    
    pairs1 = torch.cat(pairs1_list)
    pairs2 = torch.cat(pairs2_list)

    if pairs1.numel() > max_positive_pairs:
        perm = torch.randperm(pairs1.numel(), device=device)[:max_positive_pairs]
        pairs1 = pairs1[perm]
        pairs2 = pairs2[perm]
        
    target = torch.ones(pairs1.shape[0], dtype=torch.float32, device=device)
    
    return pairs1.long(), pairs2.long(), target

        
def _to_tensor(
    data_batch: pd.DataFrame,
    hit_to_particle: pd.Series,
    hit_features: List[str],
    max_hits_per_event: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert hit data and hit-to-particle mapping into PyTorch tensors. 
    We keep a dummy bin dimension of size 1 for compatibility with Train.py.
    
    Args:
    data_batch: DataFrame containing the hit data for a batch of events, must have 'event_id' column.
    hit_to_particle: Series containing the mapping from hits to particle_id, indexed the same as data_batch. # so it is just a copy of the particle_id column of data_batch right ? hit_to_particle = data_batch["particle_id"].copy()
    hit_features: List of column names in data_batch to use as hit features.
    max_hits_per_event: Maximum number of hits kept per event. Events with fewer hits are padded; events with more hits are truncated.
    Returns:
    Tuple of (hits_tensor, hit_to_particle_tensor) where:
    - hits_tensor: PyTorch tensor of shape [num_events, num_bins=1, max_hits_per_event, num_hit_features]
      containing the hit features.
    - hit_to_particle_tensor: PyTorch tensor of shape [num_events, num_bins=1, max_hits_per_event, 1].
    """

    unique_events = sorted(data_batch["event_id"].unique())
    num_events = len(unique_events)
    num_hit_features = len(hit_features)
    num_bins = 1 
    
    # Initialize tensors with proper shapes [num_events, num_bins=1, max_hits_per_event, features]
    hits_tensor = torch.zeros(
        (num_events, num_bins, max_hits_per_event, num_hit_features),
        dtype=torch.float32,
    )
    
    hit_to_particle_tensor = torch.full( # torch.zeros() is not used because 0 is a valid_particle_id
        (num_events, num_bins, max_hits_per_event, 1),
        fill_value=-2, # padding convention 
    )

    # Fill tensors event by event
    for event_idx, event_id in enumerate(unique_events):
        event_hits = data_batch[data_batch["event_id"] == event_id]
        
        num_event_hits = len(event_hits)
        n_keep = min(num_event_hits, max_hits_per_event)# number of rows to keep
        
        if n_keep > 0:
            # Fill hits_tensor with hit_features
            hits_tensor[event_idx, 0, :n_keep, :] = torch.tensor(
                event_hits[hit_features].iloc[:n_keep].values, # .values convert from dataframe(has index column, column names) to numpy array (numeric array), shape of final product is [n_keep, number_of_features]
                dtype=torch.float32,
            )
            
            event_hit_to_particle = hit_to_particle.loc[event_hits.index].iloc[:n_keep]
            hit_to_particle_tensor[event_idx, 0, :n_keep, 0] = torch.tensor(
                event_hit_to_particle.values,
                dtype=torch.int32,
            )

    return hits_tensor, hit_to_particle_tensor 


def _add_padding(
    data_batch: pd.DataFrame,
    cfg: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Pad or truncate hits event-by-event so that each event has cfg.max_hit_input rows.
    
    Conventions:
    - particle_id >=0 : real hit belonging to a real particle
    - particle_id == -1 : orphan hit
    - particle_id == -2 : padding hit
    
    Args:
        data_batch: DataFrame containing the hit data for a batch of events,
            must have 'event_id'.
        cfg: Configuration object containing max_hit_input.
    Returns:
        A DataFrame where each event has exactly cfg.max_hit_input rows.
    """
    # Prepare the padding for each event
    print("    Preparing padding for events...")
    
    max_hits = cfg.max_hit_input # p95 is ~5000 for the data, p90: 4373
    data_batch = data_batch.copy()
    
    data_batch["is_padding"] = False
    
    padded_events = [] # Stores one dataframe per event
    num_padding_rows_added = 0 # counts how many padding rows we add in total
    num_real_hits_truncated = 0
    
    unique_events = sorted(data_batch["event_id"].unique())
    
    for event_id in unique_events:
        event_hits = data_batch[data_batch["event_id"] == event_id].copy() # all hits from one event
        num_hits = len(event_hits)
    
        # Remove excess hits
        if num_hits > max_hits:
            num_real_hits_truncated += num_hits - max_hits
            event_hits = event_hits.iloc[:max_hits].copy()
    
        # Add padding if there are fewer than max_hits
        elif num_hits < max_hits:
            num_padding = max_hits - num_hits
            num_padding_rows_added += num_padding
            
            padding_rows = []
            for _ in range(num_padding):
                padding_row = {}
                
                for col in data_batch.columns:
                    if col == "event_id":
                        padding_row[col] = event_id
                    elif col == "particle_id":
                        padding_row[col] = -2 # by convention
                    elif col == "is_padding":
                        padding_row[col] = True
                    else:
                        padding_row[col] = 0
                        
                padding_rows.append(padding_row)
                
            padding_df = pd.DataFrame(padding_rows, columns=data_batch.columns)
            event_hits = pd.concat([event_hits, padding_df], ignore_index=True) 
    
        padded_events.append(event_hits) # at that point event_hits is a df representing one event, with exactly max_hits rows

    out = pd.concat(padded_events, ignore_index=True)
    
    print(f"    Added {num_padding_rows_added} padding hits")
    print(f"    Truncated {num_real_hits_truncated} real hits")

    return out


def _create_padding_mask(
    data_batch: pd.DataFrame,
    cfg: PreprocessingConfig,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Create a padding mask tensor indicating which positions in the hit input are padding.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events,
            must have 'event_id' and 'is_padding' columns.
        cfg: Configuration object containing max_hit_input.
    Returns:
        Tuple of:
        - data_batch with the 'id_padding' column removed
        - Padding mask tensor of shape [num_events, num_bins=1, cfg.max_hit_input]
        where True indicates padding positions.
    """
    
    # Create padding mask tensor [num_events, num_bins=1, max_hit_input]
    print("    Creating padding mask...")
    
    unique_events = sorted(data_batch["event_id"].unique())
    num_events = len(unique_events)
    max_hits = cfg.max_hit_input
    num_bins = 1 

    # Initialize padding mask as all False = real hit
    padding_mask = torch.zeros((num_events, num_bins, max_hits), dtype=torch.bool)

    for event_idx, event_id in enumerate(unique_events):
        event_hits = data_batch[data_batch["event_id"] == event_id]
        event_is_padding = event_hits["is_padding"].to_numpy()
        
        padding_mask[event_idx, 0, :] = torch.tensor(event_is_padding, dtype=torch.bool)
    
    data_batch = data_batch.drop(columns=["is_padding"]).copy()
    
    return  data_batch, padding_mask


def _orphan_hit_removal(data_batch: pd.DataFrame, fraction_to_drop: float, random_state: int = 1993) -> pd.DataFrame:
    """
    Randomly drop a fraction of the hits with no associated particle (orphan hits) from the data batch.

    Args:
        data_batch: DataFrame containing the hit data for a batch of events, must have a 'particle_id' column.
        fraction_to_drop: Fraction of orphan hits to randomly drop (between 0 and 1).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with the specified fraction of orphan hits removed.
    """
    if fraction_to_drop <= 0.0:
        return data_batch

    orphan_hits = data_batch[data_batch["particle_id"] == -1]
    num_orphan_hits = len(orphan_hits)
    num_to_drop = int(num_orphan_hits * fraction_to_drop)

    if num_to_drop > 0:
        drop_indices = orphan_hits.sample(n=num_to_drop, random_state=random_state).index
        data_batch = data_batch.drop(index=drop_indices).reset_index(drop=True)

    return data_batch


def _save_tensor_data(
    file_data: Dict,
    file_path: str,
    tensor_format: str = "pt",
) -> None:
    """
    Save tensor data to file in the specified format.

    Args:
        file_data: Dictionary containing all tensor data to save
        file_path: Path to save the file (without extension)
        tensor_format: Format to save data in ('pt' for PyTorch or 'h5' for HDF5)
    """
    if tensor_format == "pt":
        # Save as PyTorch tensor file
        torch.save(file_data, f"{file_path}.pt")
    elif tensor_format == "h5":
        # Save as compressed HDF5 file
        with h5py.File(f"{file_path}.h5", "w") as f:
            # Save tensor data with gzip compression
            for key, value in file_data.items():
                if isinstance(value, torch.Tensor):
                    # Convert tensor to numpy array for HDF5 storage
                    numpy_value = value.numpy()
                    f.create_dataset(
                        key,
                        data=numpy_value,
                        dtype=numpy_value.dtype,
                        compression="gzip",
                        compression_opts=9,
                    )
                elif isinstance(value, list):
                    # Save lists as attributes or datasets depending on content
                    if key == "batch_events":
                        f.create_dataset(key, data=np.array(value), dtype=np.int64, compression="gzip")
                    else:
                        f.attrs[key] = value
                else:
                    # Save scalars as attributes
                    f.attrs[key] = value
    else:
        raise ValueError(f"Unsupported tensor format: {tensor_format}. Use 'pt' or 'h5'")


def _process_single_batch(args: Tuple) -> Tuple[str, Tuple[int, int], int, int]:
    """
    Process a single batch of events and save the resulting tensors to disk.

    Args:
        args: Tuple of (data_batch, cfg, hit_features, file_id, barcode, start_event, end_event)
    Returns:
        Tuple of (file_path, (start_event, end_event), num_events_processed, nb_bins_max = 1)
    """
    
    data_batch, cfg, hit_features, file_id, barcode, start_event, end_event = args

    print(f"  Processing events {start_event} to {end_event - 1}")

    nb_bins_max = 1 
    
    # Optionally perform orphan hit removal
    data_batch = _orphan_hit_removal(data_batch, cfg.orphan_hit_fraction)

    # Pad or truncate each event to cfg.max_hit_input rows 
    data_batch = _add_padding(data_batch, cfg)
    
    # Build padding mask from the temporary 'is_padding' column
    data_batch, padding_mask = _create_padding_mask(data_batch, cfg)

    # Create hit-to-particle mapping 
    hit_to_particle = data_batch["particle_id"].copy()
    
    # Convert to tensors
    hits_tensor, hit_to_particle_tensor = _to_tensor(
        data_batch=data_batch,
        hit_to_particle=hit_to_particle,
        hit_features=hit_features,
        max_hits_per_event=cfg.max_hit_input,
    )
    
    full_path = f"{cfg.input_tensor_path}/{cfg.dataset_name}_{barcode}"
    path = f"/{cfg.dataset_name}_{barcode}"
    
    # Create output directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    
    # Get the unique event IDs for this batch
    batch_events = sorted(data_batch["event_id"].unique())
    
    # Create a single dictionary with all data
    file_data = {
        "hits_tensor": hits_tensor,
        "hit_to_particle_tensor": hit_to_particle_tensor,
        "padding_mask": padding_mask,
        "start_event": start_event,
        "end_event": end_event,
        "nb_bins": 1,
        "batch_events": batch_events,
    }
    
    # Save tensor file
    file_base = f"{full_path}/tensor_data_{file_id}_{barcode}"
    _save_tensor_data(file_data, file_base, cfg.tensor_format)
    
    # Determine file extension based on tensor format
    file_ext = ".pt" if cfg.tensor_format == "pt" else ".h5"
    
    print(f" Saved tensor data for events {start_event} to {end_event - 1}({cfg.tensor_format}format)")
    
    file_path = f"{path}/tensor_data_{file_id}_{barcode}{file_ext}"
    return file_path, (start_event, end_event), end_event - start_event, nb_bins_max
     

def compute_barcode(cfg: PreprocessingConfig) -> str:
    """
    Compute a barcode string based on the configuration parameters for easy identification of dataset variants.

    Args:
        cfg: Configuration object containing parameters that affect the dataset preparation.
    Returns:
        A string barcode that encodes key configuration parameters : max hits, and orphan hit fraction.
    """
    barcode = f"MH{cfg.max_hit_input}"
    
    if cfg.orphan_hit_fraction > 0:
        barcode += f"_OF{int(cfg.orphan_hit_fraction * 100)}"
        
    return barcode


def prepare_tensor(
    cfg: PreprocessingConfig,
) -> Dict:
    """
    Read DUNE hit files and prepare them for training by converting to PyTorch tensors.

    This function performs the complete data preprocessing pipeline:
    - load hit data from CSV or HDF5 files
    - optionally remove a fraction of orphan hits (hits with no associated particle) via _process_single_batch() function
    - pad or truncate each event to cfg.max_hit_input hits
    - creating a padding mask for attention mechanisms
    - save hit_to_particle_tensor so positive pairs can be sampled during training 
    - convert all data to PyTorch tensors
    - save one tensor file per batch of events

    For each batch of events, a single data file is saved to disk in either PyTorch (.pt) or
    compressed HDF5 (.h5) format containing:
    - `hits_tensor`: Hit data with shape [num_events, num_bins=1, max_hit_input, num_hit_features]
    - `hit_to_particle_tensor`: Hit-to-particle mapping with shape [num_events, num_bins=1, max_hit_input, 1]
    - `padding_mask`: Padding mask with shape [num_events, num_bins=1, max_hit_input]
    - Metadata: start_event, end_event, nb_bins=1, batch_events

    A metadata file is also created containing information about the dataset structure and file paths.
    The output format (PyTorch or HDF5) is controlled by cfg.tensor_format.

    Args:
        cfg: Configuration object containing all parameters including:
            - input_path: Path to input data files
            - input_format: Format of input files ('csv' or 'h5')
            - input_tensor_path: Path where output tensors will be saved
            - events_per_file: Maximum number of events per output tensor file
            - orphan_hit_fraction: Fraction of orphan hits to remove
            - binning_strategy: Binning strategy to use ('global', 'neighbor', 'margin', 'no_bin')
            - bin_width: Width of bins for binning
            - max_hit_input: Maximum number of hits per bin
            - hit_features: List of column names to use as hit features
            - tensor_format: Output tensor file format ('pt' for PyTorch or 'h5' for compressed HDF5)

    Returns:
        Dictionary containing metadata about the processed dataset including:
        - total_events: Total number of events processed
        - events_per_file: Number of events per output file 
        - nb_bins: Maximum number of bins used, here 1 
        - orphan_hit_fraction: Fraction of orphan hits removed
        - file_paths: List of paths to generated tensor files
        - file_event_ranges: List of (start, end) event ranges for each file
    """

    # Use feature lists from config
    hit_features = cfg.hit_features
    needed_columns = list(set(["event_id", "particle_id"] + hit_features)) 
    
    total_events = 0
    nb_bins_max = 1
    file_paths: List[str] = []
    file_event_ranges: List[Tuple[int, int]] = []
    orphan_hit_fraction = cfg.orphan_hit_fraction
    events_per_file = cfg.events_per_file 
    barcode = compute_barcode(cfg)

    # Get file paths based on format
    input_path = cfg.input_path
    input_format = cfg.input_format
    data_files = []
    
    # Collect all data files and prepare for loop
    if input_format == "h5":
        # Get all HDF5 files
        data_files = sorted(glob.glob(f"{input_path}/processed_data*.h5"))
        print(f"Found {len(data_files)} HDF5 file(s)")
        if len(data_files) == 0:
            raise FileNotFoundError(f"No processed_data*.h5 files found in {input_path}")

    elif input_format == "csv":
        data_files = sorted(glob.glob(f"{input_path}/hits_small*.csv"))
        print(f"Found {len(data_files)} hits CSV file(s)")
        if len(data_files) == 0:
            raise FileNotFoundError(f"No hits_small*.csv files found in {input_path}")
        
    else:
        raise ValueError(f"Unsupported input format: {input_format}. Use 'csv' or 'h5")
    
    file_id = 0
    tot_event = 0 
    
    # Processing each file
    for file_idx, data_file in enumerate(data_files):
        if cfg.max_events > 0 and tot_event >= cfg.max_events:
            print(f"Reached max_events limit of {cfg.max_events}. Stopping further processing.")
            break

        print(f"Processing file {file_idx + 1}/{len(data_files)}")

        if input_format == "h5":
            # Read data from HDF5 file
            with pd.HDFStore(data_file, mode="r") as store:
                # Check which key exists in the HDF5 file
                keys = [key.lstrip("/") for key in store.keys()]

                if "hits" not in keys:
                    raise KeyError(f"hits not found in {data_file}. Available keys:{store.keys()}")
                
                data = store.select("hits", columns=needed_columns)
                print(f" Loaded hits data from {data_file}")

        elif input_format == "csv":
            # Read data from CSV files
            data = pd.read_csv(data_file)
            data = data[needed_columns].copy()
            print(f" Loaded hits data from {data_file}")
            
        event_ids_all = sorted(data["event_id"].unique())  
        num_events = len(event_ids_all)
        tot_event += num_events
        
        if cfg.max_events > 0 and tot_event > cfg.max_events:
            allowed_events = cfg.max_events + num_events - tot_event
            print((f" Limiting to {allowed_events} event(s) from this file"))
            event_ids_all = event_ids_all[:allowed_events]
            data = data[data["event_id"].isin(event_ids_all)].reset_index(drop=True)
            num_events = len(event_ids_all)
            
        if num_events == 0: # safety guard
            continue
        
        if num_events % events_per_file != 0: # each file divided into chunks, events_per_file = number of events per chunk
            print(f"Warning: Number of events in {data_file}({num_events})"
                  f"is not divisible by events_per_file ({events_per_file})"
            )
        
        # Build arguments for each batch so they can be processed in parallel
        batch_args = []
        local_file_id = file_id
        
        for batch_start_idx in range(0, num_events, events_per_file):
            batch_end_idx = min(batch_start_idx + events_per_file, num_events)
            
            batch_event_ids = event_ids_all[batch_start_idx:batch_end_idx]
            
            data_batch = data[data["event_id"].isin(batch_event_ids)].reset_index(drop=True)

            start_event = int(batch_event_ids[0])
            end_event = int(batch_event_ids[-1])+1
            
            batch_args.append(
                (
                    data_batch,
                    cfg,
                    hit_features,
                    local_file_id,
                    barcode,
                    start_event,
                    end_event,
                    
                )
            )
            local_file_id += 1
            
        # Process batches
        num_workers = min(cfg.num_workers, len(batch_args))
        if num_workers > 1:
            print(f" Spawning {num_workers} worker processes for {len(batch_args)} batch(es)")
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(_process_single_batch, batch_args)
        else:
            results = [_process_single_batch(args) for args in batch_args]
        
        # Collect results 
        for file_path, event_range, n_events, nb_bins in results:
            file_paths.append(file_path)
            file_event_ranges.append(event_range)
            total_events += n_events
            nb_bins_max = max(nb_bins_max, nb_bins)# needed ? 
            
        file_id = local_file_id
        
        # Save metadata
    metadata = {
        "total_events": total_events,
        "events_per_file": events_per_file,
        "nb_bins": nb_bins_max,
        "orphan_hit_fraction": orphan_hit_fraction,
        "max_hit_input": cfg.max_hit_input,
        "hit_features": hit_features,
        "tensor_format": cfg.tensor_format,
        "file_paths": file_paths,
        "file_event_ranges": file_event_ranges,
    }
    
    os.makedirs(cfg.input_tensor_path, exist_ok=True)
    torch.save(metadata, f"{cfg.input_tensor_path}/metadata_{cfg.dataset_name}_{barcode}.pt")
    
    return metadata


if __name__ == "__main__":
    # Create config object and parse command line arguments
    cfg = PreprocessingConfig()
    cfg.parse_args()

    # Print the configuration
    cfg.print_config()

    # Prepare tensors using config settings
    prepare_tensor(cfg)
