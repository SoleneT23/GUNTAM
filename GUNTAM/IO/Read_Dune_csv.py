import glob
import argparse
from typing import List, Optional
import numpy as np
import pandas as pd


def _process_hits_data(data: pd.DataFrame) -> pd.DataFrame:
    required = {"event", "id", "x", "y", "z"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns:{missing}")
    data = data.rename(columns={"id": "particle_id"}).copy()
    data = data[["event", "particle_id", "x", "y", "z"]].copy()
    data = data.drop_duplicates(subset=["particle_id", "x", "y", "z"], keep="first").copy() # overlapping particles 
    data["hit_id"] = data.index.astype(int)
    
    x_sq = data["x"] ** 2
    y_sq = data["y"] ** 2
    z_sq = data["z"] ** 2

    data["r"] = np.sqrt(x_sq + y_sq)

    rho = np.sqrt(x_sq + y_sq)
    theta = np.arctan2(rho, data["z"])
    data["eta"] = -np.log(np.tan(theta / 2))
    data["phi"] = np.arctan2(data["y"], data["x"])

    return data


def read_hits_csv(args:argparse.Namespace) -> None:
    
    file_suffix = f"_{args.file_number}" if args.file_number is not None else ""
    hit_files = sorted(glob.glob(f"{args.input_path}/event*-hits.csv")) # sorts filenames lexicographically as strings not numerically
    if len(hit_files) == 0:
        raise FileNotFoundError(f"No files matching {args.input_path}/event*-hits.csv")
    total_files = len(hit_files)
    all_data:List[Optional[pd.DataFrame]] = [None]*total_files
    
    for counter, file in enumerate(hit_files): # file is path to each event*-hits.csv, each iteration processes one file = one event file
        if counter % 10 == 0 or counter == total_files - 1:
            print(f"Processing event {counter+1} / {total_files} ({counter / total_files * 100:.1f}%)")

        data = pd.read_csv(file)
        data = _process_hits_data(data)
        data["event_id"] = counter
        particle_counts = data["particle_id"].value_counts() # this counts how many times each particle_id appears, index = particle id, value = number of hits
        valid_particle_ids = particle_counts[particle_counts >= args.min_hits_per_particle].index # keep particles that have only min_hits_per_particle
        
        mask_invalid = ~data["particle_id"].isin(valid_particle_ids)# ~ flips True and False, True = rows with invalid particles
        data.loc[mask_invalid, "particle_id"] = -1 # for rows where mask_invalid is true, replace particle_id with -1
        all_data[counter] = data # None is replaced with a processed Dataframe
        
    print("Concatenating all data…")
    full_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame() # if all_data means non-empty list : True, empty list : False, concat merges dataframes 
    
    print("Concatenation completed")
    print("Final data shape:")
    print(f"Hits:{full_data.shape}")
    print("Writing output files...")
    hits_filename = f"{args.input_path}/hits_small{file_suffix}.csv"
    hdf_filename = f"{args.input_path}/processed_data{file_suffix}.h5"
    
    files_written = []
    
    if "csv" in args.output_format:
        full_data.to_csv(hits_filename, index=False)
        files_written.append(hits_filename)
    
    if "h5" in args.output_format:
        with pd.HDFStore(hdf_filename, mode="w", complevel=9, complib="blosc") as store:
            store.put("hits", full_data, format="table")
        files_written.append(hdf_filename)
    
    print("Files written successfully:")
    for filename in files_written:
        print(f"{filename}")
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Preprocess hit CSV files")
    
    parser.add_argument(
        "--file-number",
        type=int,
        default=None,
        help="Optional number appended to output filenames",
        
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        default="/gpfs/workdir/thibauts/truth_points_all_events",
        help="Path containing event*-hits.csv files", 
    )
    
    parser.add_argument(
        "--min-hits-per-particle",
        type=int,
        default=9,
        help="Minimum hits required for a particle to keep its particle_id"
    )
    
    parser.add_argument(
        "--output-format",
        nargs="+",
        default=["csv"],
        choices=["csv", "h5"],
        help="Output file format(s)"
    )
    
    args = parser.parse_args()
    read_hits_csv(args)