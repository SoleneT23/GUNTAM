import os
import pandas as pd
import argparse

# input_file = "/gpfs/workdir/meleppattun/truth_points_20events.csv"
# output_dir_small = "/gpfs/workdir/thibauts/GUNTAM/tests/data"
# output_dir_full = "/gpfs/workdir/thibauts/truth_points_all_events"

def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into one file per event")

    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir_full", type=str, required=True, help="Directory for all event files")
    parser.add_argument("--output_dir_small", type=str, default=None, help="Directory for subset of events for integration tests")
    parser.add_argument("--num_small_events", type=int, default=0, help="Number of events to save for integration tests")

    args = parser.parse_args()

    os.makedirs(args.output_dir_full, exist_ok=True)
    if args.output_dir_small:
        os.makedirs(args.output_dir_small, exist_ok=True)

    print(f"Loading file: {args.input_file}")
    df = pd.read_csv(args.input_file)

    required = {"event", "id", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns={missing}")

    event_ids = sorted(df["event"].unique())
    print(f"Found {len(event_ids)} events")

    first_events = set(event_ids[:args.num_small_events]) if args.num_small_events > 0 else set()

    for event_id, event_df in df.groupby("event"):
        event_str = f"{int(event_id):09d}"
        event_out = event_df[["event", "id", "x", "y", "z"]].copy()

        full_path = os.path.join(args.output_dir_full, f"event{event_str}-hits.csv")
        event_out.to_csv(full_path, index=False)

        if args.output_dir_small and event_id in first_events:
            small_path = os.path.join(args.output_dir_small, f"event{event_str}-hits.csv")
            event_out.to_csv(small_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()