#!/usr/bin/env python3
"""
Utility functions used across the pipeline.
"""

import os
import json
import logging
import subprocess
import pandas as pd
import numpy as np
from multiprocessing import Pool
from statistics import median, mean

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def list_json_files(directory, suffix='.json'):
    """Recursively list all JSON files in a directory."""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(suffix):
                json_files.append(os.path.join(root, file))
    return json_files


def merge_chromosome_qc(json_files, sample_path, output_dir, sample_id):
    """
    Merge per-chromosome QC JSON files into a single DataFrame,
    sort by chromosome, and save merged JSON/TSV.
    Returns the merged DataFrame.
    """
    data_frames = []
    for f in json_files:
        df = pd.read_json(f)
        data_frames.append(df)

    if not data_frames:
        raise ValueError(f"No JSON data to merge for sample {sample_id}")

    merged = pd.concat(data_frames, ignore_index=True)
    # Clean chromosome column: remove 'chr' prefix for sorting, then restore
    merged['Chrom'] = merged['Chrom'].str.replace('chr', '')
    merged['Chrom'] = merged['Chrom'].astype(int)
    merged = merged.sort_values(by=['Chrom'], ascending=True)
    merged['Chrom'] = 'chr' + merged['Chrom'].astype(str)
    merged.reset_index(drop=True, inplace=True)

    # Save merged files
    merged.to_json(os.path.join(output_dir, sample_id, f"{sample_id}_merged_all_chrom_qc_metrics.json"),
                   orient='records')
    merged.to_csv(os.path.join(output_dir, sample_id, f"{sample_id}_merged_all_chrom_qc_metrics.tsv"),
                  sep='\t', index=False)
    return merged


def read_verifybamid_results(file_path):
    """Read VerifyBamID2 .selfSM file; return dict with SNPS, FREEMIX, AVG_DP."""
    default = {'#SNPS': 'NA', 'FREEMIX': 'NA', 'AVG_DP': 'NA'}
    if not os.path.exists(file_path):
        return default
    try:
        df = pd.read_csv(file_path, sep='\t', header=0)
        return {
            '#SNPS': df.loc[0, '#SNPS'],
            'FREEMIX': df.loc[0, 'FREEMIX'],
            'AVG_DP': df.loc[0, 'AVG_DP']
        }
    except Exception as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return default


def compute_sample_summary(merged_df, verify_data, sample_id):
    """
    Compute summary statistics from merged per-chromosome QC data.
    Matches the original combine_JSONs logic.
    """
    # Original combined metrics
    total_bases_aligned = merged_df['Total_number_of_bases_aligned'].sum()
    total_bases = merged_df['Total_number_of_bases'].sum()
    # Avoid division by zero
    avg_auto_cov = total_bases_aligned / total_bases if total_bases else 0

    # Median of per-chromosome median coverages
    median_cov_list = merged_df["Median_alignment_coverage_over_autosomal_loci"]
    median_val = round(median(median_cov_list))
    mean_val = mean(median_cov_list)
    mean_by_median = mean_val / median_val if median_val else 0

    # Percentages (mean of per-chromosome percentages)
    pct_5x = round(mean(merged_df["Percent_autosome_coverage_at_5X"]))
    pct_10x = round(mean(merged_df["Percent_autosome_coverage_at_10X"]))
    pct_15x = round(mean(merged_df["Percent_autosome_coverage_at_15X"]))
    pct_20x = round(mean(merged_df["Percent_autosome_coverage_at_20X"]))
    pct_25x = round(mean(merged_df["Percent_autosome_coverage_at_25X"]))
    pct_30x = round(mean(merged_df["Percent_autosome_coverage_at_30X"]))

    summary = {
        "Biosample_id": sample_id,
        "Total_loci_in_Genome": merged_df["Total_loci_in_Genome"].unique()[0] if len(merged_df) > 0 else 'NA',
        "Total_number_of_bases_aligned:": total_bases_aligned,
        "Total_number_of_bases:": total_bases,
        "Average_autosomal_coverage:": avg_auto_cov,
        "Median_alignment_coverage_over_autosomal_loci:": median_val,
        "Mean_alignment_coverage_over_autosomal_loci:": mean_val,
        "Mean_by_Median_autosomal_coverage_ratio_over_region:": mean_by_median,
        "Percent_autosome_coverage_at_5X:": pct_5x,
        "Percent_autosome_coverage_at_10X:": pct_10x,
        "Percent_autosome_coverage_at_15X:": pct_15x,
        "Percent_autosome_coverage_at_20X:": pct_20x,
        "Percent_autosome_coverage_at_25X:": pct_25x,
        "Percent_autosome_coverage_at_30X:": pct_30x,
        "Bases_covered_5x:": merged_df['Bases_covered_5x'].sum(),
        "Bases_covered_10x:": merged_df['Bases_covered_10x'].sum(),
        "Bases_covered_15x:": merged_df['Bases_covered_15x'].sum(),
        "Bases_covered_20x:": merged_df['Bases_covered_20x'].sum(),
        "Bases_covered_25x:": merged_df['Bases_covered_25x'].sum(),
        "Bases_covered_30x:": merged_df['Bases_covered_30x'].sum(),
        "VerifyBamID2_SNPS:": verify_data['#SNPS'],
        "VerifyBamID2_Average_Depth:": verify_data['AVG_DP'],
        "VerifyBamID2_Contamination:": verify_data['FREEMIX']
    }
    return summary


def parallel_download(cfg, sample_df, num_threads):
    """Download BAM files in parallel using a process pool."""
    from functools import partial
    # Prepare list of download tasks
    tasks = []
    for idx, row in sample_df.iterrows():
        # Check if BAM already exists (both .bam and .bai)
        bam_file = os.path.join(cfg['bamdir'], f"{row['Library_ID']}.recal.bam")
        bai_file = bam_file + '.bai'
        if os.path.exists(bam_file) and os.path.exists(bai_file):
            logger.debug(f"BAM files for {row['Library_ID']} already exist, skipping.")
            continue
        tasks.append((cfg['bucketdownload'], cfg['bamdir'], row['Path'], row['Library_ID']))

    if not tasks:
        logger.info("All BAM files already present.")
        return

    # Download function for parallel execution
    def download_one(task):
        bucket, bamdir, source_path, lib_id = task
        cmd1 = f"aws s3 cp s3://{bucket}/{source_path}/results_{lib_id}/preprocessing/recalibrated/{lib_id}/{lib_id}.recal.bam {bamdir}/ --profile enabl-wgs-alignment"
        cmd2 = f"aws s3 cp s3://{bucket}/{source_path}/results_{lib_id}/preprocessing/recalibrated/{lib_id}/{lib_id}.recal.bam.bai {bamdir}/ --profile enabl-wgs-alignment"
        for cmd in [cmd1, cmd2]:
            logger.debug(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

    with Pool(processes=num_threads) as pool:
        pool.map(download_one, tasks)


def chunk_dataframe(df, num_chunks):
    """Split a DataFrame into roughly equal chunks (returns list of DataFrames)."""
    return np.array_split(df, num_chunks)


def process_chunks_parallel(df, num_processes, func, fixed_args):
    """
    Split DataFrame into chunks and process each chunk in parallel using multiprocessing Pool.
    func should accept (chunk_df, *fixed_args) and return something.
    Returns list of results.
    """
    chunks = chunk_dataframe(df, num_processes)
    # Prepare arguments for each chunk: (chunk, *fixed_args)
    args_list = [(chunk,) + fixed_args for chunk in chunks]
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(func, args_list)
    return results