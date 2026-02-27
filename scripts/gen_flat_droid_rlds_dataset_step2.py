"""
Step 2 (Single Threaded): Gather & Shuffle Phase.
Strategy: Binary Pass-through (No deserialization) for maximum safety.
Reads intermediate buckets, shuffles records in memory, and writes final TFRecords.
"""

import os
import glob
import random
import argparse
import shutil
import logging
from tqdm import tqdm
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_bucket(bucket_path, output_path):
    """
    Reads all TFRecords in a bucket, shuffles them in memory, and writes to output.
    Returns the number of records processed.
    """
    # 1. Find all partial files in the bucket
    part_files = glob.glob(os.path.join(bucket_path, "*.tfrecord"))
    if not part_files:
        return 0

    # 2. Read records (Binary Pass-through)
    # We do NOT parse the example. We just hold the serialized bytes.
    records = []
    try:
        # TFRecordDataset reads the raw bytes of each record
        dataset = tf.data.TFRecordDataset(part_files, buffer_size=100*1024*1024)
        
        for raw_record in dataset:
            records.append(raw_record.numpy())
            
    except Exception as e:
        logging.error(f"Error reading bucket {bucket_path}: {e}")
        raise e

    if not records:
        return 0

    # 3. Memory Shuffle
    random.shuffle(records)

    # 4. Write to final destination
    try:
        with tf.io.TFRecordWriter(output_path) as writer:
            for r in records:
                writer.write(r)
    except Exception as e:
        logging.error(f"Error writing to {output_path}: {e}")
        raise e

    return len(records)

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Gather and Shuffle buckets into final TFRecords.")
    parser.add_argument("--temp_dir", type=str, required=True, help="Input directory containing bucket_xxxxx folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Final output directory for shuffled data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Prepare output directory
    if not os.path.exists(args.temp_dir):
        raise FileNotFoundError(f"Temp directory not found: {args.temp_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir}")

    # Find all bucket directories
    # Expecting structure: temp_dir/bucket_00000, temp_dir/bucket_00001, ...
    bucket_dirs = sorted(glob.glob(os.path.join(args.temp_dir, "bucket_*")))
    
    if not bucket_dirs:
        logging.warning(f"No buckets found in {args.temp_dir}! Check your step 1 output.")
        return

    logging.info(f"Found {len(bucket_dirs)} buckets to process.")

    total_records = 0

    for bucket_dir in bucket_dirs:
        # Extract bucket ID from folder name (assuming bucket_01234 format)
        try:
            folder_name = os.path.basename(bucket_dir)
            bucket_id = folder_name.split('_')[-1]
        except:
            # Fallback if naming is weird
            bucket_id = os.path.basename(bucket_dir)

        # Construct deterministic output filename
        # e.g., droid_shuffled_00001.tfrecord
        output_filename = f"droid_shuffled_{bucket_id}.tfrecord"
        output_path = os.path.join(args.output_dir, output_filename)

        # Process
        count = process_bucket(bucket_dir, output_path)
        total_records += count
        
        # Update progress bar info
        # pbar.set_postfix({"Total Records": total_records})
        logging.info(f"Processed bucket {bucket_id}: {count} records. Total so far: {total_records}")

    logging.info("------------------------------------------------")
    logging.info(f"Global Shuffle Stage 2 Complete.")
    logging.info(f"Total Buckets Processed: {len(bucket_dirs)}")
    logging.info(f"Total Records Generated: {total_records}")
    logging.info(f"Data saved to: {args.output_dir}")
    logging.info("------------------------------------------------")

if __name__ == "__main__":
    main()
