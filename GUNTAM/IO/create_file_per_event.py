import os
import pandas as pd
import argparse

# Path to dataset : /workdir/meleppattun/guntam_data/truth_particles.csv
# it has 66 million lines

def write_event_file(event_id, event_df, output_dir):
    """
    Save one event dataframe into one eventXXXXXXXXX-hits.csv file.
    """
    event_str = f"{int(event_id):09d}"

    event_out = event_df[
        ["event_id", "particle_id", "x", "y", "z", "charge"]
    ].copy()

    output_path = os.path.join(output_dir, f"event{event_str}-hits.csv")
    event_out.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Split truth_particles.csv into one hits file per event"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="/workdir/meleppattun/guntam_data/truth_particles.csv",
        help="Path to input CSV file",
    )

    parser.add_argument(
        "--output_dir_full",
        type=str,
        required=True,
        help="Directory where all per-event files will be saved",
    )

    parser.add_argument(
        "--output_dir_small",
        type=str,
        default=None,
        help="Optional directory for a small subset of events for tests",
    )

    parser.add_argument(
        "--num_small_events",
        type=int,
        default=0,
        help="Number of first events to also save in output_dir_small",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Number of CSV rows to read at a time",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir_full, exist_ok=True)

    if args.output_dir_small:
        os.makedirs(args.output_dir_small, exist_ok=True)

    required = {"event_id", "x", "y", "z", "charge", "particle_id"}

    print(f"Reading input file: {args.input_file}")
    print(f"Saving full event files to: {args.output_dir_full}")

    if args.output_dir_small:
        print(
            f"Also saving first {args.num_small_events} events to: "
            f"{args.output_dir_small}"
        )

    buffer = pd.DataFrame()
    written_events = set() # to know when we have saved enough small events 
    small_events = set() # stores the IDs of the events copied into the small test directory

    reader = pd.read_csv(args.input_file, chunksize=args.chunksize)

    for chunk_idx, chunk in enumerate(reader):
        print(f"Processing chunk {chunk_idx}")

        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Add the previous incomplete event to the current chunk
        if not buffer.empty:
            chunk = pd.concat([buffer, chunk], ignore_index=True)
            buffer = pd.DataFrame()

        # The last event in a chunk may continue in the next chunk
        # we keep it aside and only write complete events
        last_event_id = chunk["event_id"].iloc[-1]

        complete_chunk = chunk[chunk["event_id"] != last_event_id]
        buffer = chunk[chunk["event_id"] == last_event_id].copy()

        for event_id, event_df in complete_chunk.groupby("event_id", sort=True):
            write_event_file(event_id, event_df, args.output_dir_full)
            written_events.add(event_id)

            if (
                args.output_dir_small
                and args.num_small_events > 0
                and len(small_events) < args.num_small_events
            ):
                write_event_file(event_id, event_df, args.output_dir_small)
                small_events.add(event_id)

    # Write the final buffered event
    if not buffer.empty:
        event_id = buffer["event_id"].iloc[0]
        write_event_file(event_id, buffer, args.output_dir_full)
        written_events.add(event_id)

        if (
            args.output_dir_small
            and args.num_small_events > 0
            and len(small_events) < args.num_small_events
        ):
            write_event_file(event_id, buffer, args.output_dir_small)
            small_events.add(event_id)

    print("Done.")
    print(f"Number of events written: {len(written_events)}")

    if args.output_dir_small:
        print(f"Number of small test events written: {len(small_events)}")


if __name__ == "__main__":
    main()