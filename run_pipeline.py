#!/usr/bin/env python3
"""
Main pipeline orchestrator for WGS/WES QC.
Reads configuration, downloads BAMs (optional), runs QC metrics and VerifyBamID2
in parallel, then combines results into a final TSV.
"""

import os
import sys
import argparse
import configparser
import logging
from multiprocessing import Process, cpu_count
import pandas as pd

# Import refactored modules
import qc_metrics
import verify_bamid
import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load and return configuration dictionary."""
    config = configparser.ConfigParser()
    config.read(config_path)
    cfg = {}
    for section in config.sections():
        for key, value in config.items(section):
            cfg[key] = value
    return cfg


def setup_directories(cfg):
    """Create necessary directories if they don't exist."""
    dirs = [cfg['workdir'], cfg['bamdir'], 
            os.path.join(cfg['workdir'], 'QC_metrics'),
            os.path.join(cfg['workdir'], 'res_VerifyBamID2')]
    for d in dirs:
        os.makedirs(d, mode=0o700, exist_ok=True)
    # Set TMPDIR
    os.makedirs(cfg['tmp_dir'], mode=0o700, exist_ok=True)
    os.environ['TMPDIR'] = cfg['tmp_dir']


def download_bams(cfg, sample_df):
    """Download BAM files for all samples (if not already present)."""
    logger.info("Starting BAM downloads...")
    num_threads = min(int(cfg.get('num_threads', 4)), len(sample_df))
    utils.parallel_download(cfg, sample_df, num_threads)
    logger.info("BAM downloads completed.")


def run_qc_metrics(cfg, sample_df):
    """Run QC metrics calculation in parallel chunks."""
    logger.info("Starting QC metrics calculation...")
    qc_metrics.run_qc(cfg, sample_df)
    logger.info("QC metrics calculation completed.")


def run_verify_bamid(cfg, sample_df):
    """Run VerifyBamID2 in parallel chunks."""
    logger.info("Starting VerifyBamID2 analysis...")
    verify_bamid.run_verify(cfg, sample_df)
    logger.info("VerifyBamID2 analysis completed.")


def combine_results(cfg):
    """Combine per-sample QC metrics and VerifyBamID2 results into final TSV."""
    logger.info("Combining results...")
    input_output_dir = os.path.join(cfg['workdir'], 'QC_metrics')
    verify_dir = os.path.join(cfg['workdir'], 'res_VerifyBamID2')

    # Gather all sample directories
    try:
        all_samples = [d for d in os.listdir(input_output_dir)
                       if os.path.isdir(os.path.join(input_output_dir, d))]
    except FileNotFoundError:
        logger.error(f"QC_metrics directory not found: {input_output_dir}")
        return

    final_data = []
    for sample in all_samples:
        sample_path = os.path.join(input_output_dir, sample)
        json_files = utils.list_json_files(sample_path, '.json')
        if not json_files:
            logger.warning(f"No JSON files found for sample {sample}, skipping.")
            continue
        # Merge per-chromosome JSONs
        merged_df = utils.merge_chromosome_qc(json_files, sample_path, input_output_dir, sample)
        # Read VerifyBamID2 results (if available)
        verify_file = os.path.join(verify_dir, f"{sample}.selfSM")
        verify_data = utils.read_verifybamid_results(verify_file)
        # Compute summary statistics
        summary = utils.compute_sample_summary(merged_df, verify_data, sample)
        final_data.append(summary)

    if final_data:
        final_df = pd.DataFrame(final_data)
        # Clean column names (remove stray colons)
        final_df.columns = final_df.columns.str.replace(':', '')
        out_file = os.path.join(input_output_dir, "QC_metricses_data_all_samples.tsv")
        final_df.to_csv(out_file, sep='\t', index=False)
        logger.info(f"Final results written to {out_file}")
    else:
        logger.warning("No data to combine.")


def main():
    parser = argparse.ArgumentParser(description='Run WGS/WES QC pipeline.')
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'data', 'config.ini'),
                        help='Path to configuration file')
    parser.add_argument('--steps', nargs='+', 
                        choices=['download', 'qc', 'verify', 'combine'], 
                        default=['qc', 'verify', 'combine'],
                        help='Pipeline steps to execute (default: qc verify combine)')
    parser.add_argument('--sampleinfo', help='Override sample info file from config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.sampleinfo:
        cfg['sampleinfo'] = args.sampleinfo

    setup_directories(cfg)

    # Read sample information
    sample_df = pd.read_csv(cfg['sampleinfo'], sep='\t')
    # Add full BAM path column (as used in original scripts)
    sample_df['bampath'] = sample_df.apply(
        lambda row: os.path.join(cfg['bam_base_dir'], row['Path'],
                                 f"results_{row['Library_ID']}", "preprocessing", "recalibrated",
                                 row['Library_ID']),
        axis=1
    )
    # Shuffle for load balancing (optional)
    sample_df = sample_df.sample(frac=1).reset_index(drop=True)

    if 'download' in args.steps:
        download_bams(cfg, sample_df)

    # Run QC metrics and VerifyBamID2 in parallel (if both steps are requested)
    processes = []
    if 'qc' in args.steps:
        p1 = Process(target=run_qc_metrics, args=(cfg, sample_df))
        processes.append(p1)
        p1.start()
    if 'verify' in args.steps:
        p2 = Process(target=run_verify_bamid, args=(cfg, sample_df))
        processes.append(p2)
        p2.start()

    # Wait for both to finish
    for p in processes:
        p.join()

    if 'combine' in args.steps:
        combine_results(cfg)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()