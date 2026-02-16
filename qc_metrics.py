#!/usr/bin/env python3
"""
QC metrics calculation module for WGS/WES pipeline.

Handles:
- Creation of autosomal BED files (WGS: from .fai minus gaps; WES: from GTF)
- Parallel processing of BAM files to compute per‑chromosome coverage statistics
- Writing per‑chromosome JSON and TSV outputs with detailed logs
"""

import os
import logging
import subprocess
import time
import json
from statistics import median, mean

import pandas as pd
import pysam
import numpy as np
from tqdm import tqdm

import utils   # our custom utility module

logger = logging.getLogger(__name__)


def create_autosome_bed(cfg):
    """
    Create a BED file of autosomal regions.

    For WGS:
        - Uses the first 22 chromosomes from the .fai file.
        - Subtracts gap regions (from gap.txt.gz) using bedtools subtract.
    For WES:
        - Extracts exon/intron regions from the GTF file (provided in config).
        - Keeps only autosomal chromosomes.
        - Converts to 0‑based coordinates and sorts.

    The final sorted BED is saved as 'sorted_autosomes.bed' in the working directory.
    """
    workdir = cfg['workdir']
    analysis_type = cfg['analysis_type']
    fai_file = cfg['fai']
    gap_file = cfg['gap']

    if analysis_type == "WES":
        logger.info("Creating autosomal BED from GTF for WES...")
        gtf_file = cfg['gtf']
        # Read GTF, skipping comment lines
        gtf = pd.read_csv(gtf_file, sep='\t', comment='#', header=None,
                          names=['seqname', 'source', 'feature', 'start', 'end',
                                 'score', 'strand', 'frame', 'attribute'])
        # Keep only autosomal chromosomes (1-22, with or without 'chr' prefix)
        autosomes = gtf[gtf['seqname'].str.match(r'^(chr)?[0-9]{1,2}$')]
        bed = autosomes[['seqname', 'start', 'end']].copy()
        bed['start'] = bed['start'] - 1   # GTF is 1‑based, BED is 0‑based
        bed.drop_duplicates(inplace=True)
        bed.to_csv(os.path.join(workdir, 'autosomes.bed'), sep='\t',
                   header=False, index=False)
        # Sort using Linux sort (chr order, then start position)
        cmd = (f"egrep -w '^chr[1-9]|chr[1-2][0-9]' {workdir}/autosomes.bed "
               f"| sort -k1,1V -k2,2n > {workdir}/sorted_autosomes.bed")
        subprocess.run(cmd, shell=True, check=True)

    elif analysis_type == "WGS":
        logger.info("Creating autosomal BED from genome FASTA index...")
        bed_file = os.path.join(workdir, 'autosomes.bed')
        with open(fai_file, 'r') as fai, open(bed_file, 'w') as bed:
            for i in range(22):   # first 22 lines = chr1‑22
                line = fai.readline()
                chrom, length = line.strip().split('\t')[:2]
                bed.write(f"{chrom}\t0\t{length}\n")
        # Extract gap regions, keep only autosomes, sort
        cmd_gap = (f"zcat {gap_file} | cut -f2-4 | egrep -w '^chr[1-9]|chr[1-2][0-9]' "
                   f"| sort -k1,1V -k2,2n > {workdir}/gap_regions.bed")
        subprocess.run(cmd_gap, shell=True, check=True)
        # Subtract gaps from the full autosome BED
        cmd_subtract = (f"bedtools subtract -a {bed_file} -b {workdir}/gap_regions.bed "
                        f"> {workdir}/sorted_autosomes.bed")
        subprocess.run(cmd_subtract, shell=True, check=True)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    logger.info(f"Autosomal BED created: {workdir}/sorted_autosomes.bed")


def _write_chromosome_metrics(out_dir, biosample_id, chrom, total_loci,
                              total_bases_aligned, total_positions, coverage_values,
                              bases_5x, bases_10x, bases_15x, bases_20x,
                              bases_25x, bases_30x):
    """
    Write per‑chromosome QC metrics to JSON and TSV files.

    This is a helper function called after processing each chromosome.
    """
    if total_positions == 0:
        logger.warning(f"No positions covered on {chrom} for {biosample_id}")
        pct_5x = pct_10x = pct_15x = pct_20x = pct_25x = pct_30x = 0
        median_cov = 0
        mean_cov = 0
        min_cov = max_cov = 0
    else:
        pct_5x = (bases_5x / total_positions) * 100
        pct_10x = (bases_10x / total_positions) * 100
        pct_15x = (bases_15x / total_positions) * 100
        pct_20x = (bases_20x / total_positions) * 100
        pct_25x = (bases_25x / total_positions) * 100
        pct_30x = (bases_30x / total_positions) * 100
        median_cov = median(coverage_values)
        mean_cov = mean(coverage_values)
        min_cov = min(coverage_values)
        max_cov = max(coverage_values)

    data = [{
        "Biosample_id": biosample_id,
        "Chrom": chrom,
        "Total_loci_in_Genome": total_loci,
        "Total_number_of_bases_aligned:": total_bases_aligned,
        "Total_number_of_bases:": total_positions,
        "Alignment_coverage_over_autosomal_loci[min_max]:": [min_cov, max_cov],
        "Median_alignment_coverage_over_autosomal_loci:": median_cov,
        "Mean_alignment_coverage_over_autosomal_loci:": mean_cov,
        "Mean_by_Median_autosomal_coverage_ratio_over_region:": (
            mean_cov / median_cov if median_cov else 0
        ),
        "Percent_autosome_coverage_at_5X:": round(pct_5x),
        "Percent_autosome_coverage_at_10X:": round(pct_10x),
        "Percent_autosome_coverage_at_15X:": round(pct_15x),
        "Percent_autosome_coverage_at_20X:": round(pct_20x),
        "Percent_autosome_coverage_at_25X:": round(pct_25x),
        "Percent_autosome_coverage_at_30X:": round(pct_30x),
        "Bases_covered_5x:": bases_5x,
        "Bases_covered_10x:": bases_10x,
        "Bases_covered_15x:": bases_15x,
        "Bases_covered_20x:": bases_20x,
        "Bases_covered_25x:": bases_25x,
        "Bases_covered_30x:": bases_30x
    }]

    json_path = os.path.join(out_dir, f"{chrom}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, cls=utils.NpEncoder)

    tsv_path = os.path.join(out_dir, f"{chrom}.tsv")
    pd.DataFrame(data).to_csv(tsv_path, sep='\t', index=False)


def process_sample_qc(workdir, biosample_id, bamdir, bamfile):
    """
    Process a single sample: iterate over autosomal regions, compute coverage,
    and write per‑chromosome JSON/TSV files.

    Args:
        workdir: Pipeline working directory (contains sorted_autosomes.bed)
        biosample_id: Sample identifier (Library_ID)
        bamdir: Directory containing the BAM file
        bamfile: Name of the BAM file (e.g. 'sample.recal.bam')
    """
    bed_path = os.path.join(workdir, "sorted_autosomes.bed")
    if not os.path.exists(bed_path):
        raise FileNotFoundError(f"BED file not found: {bed_path}")

    bed_df = pd.read_csv(bed_path, header=None, sep='\t',
                         names=['chrom', 'start', 'end'])
    total_loci = (bed_df['end'] - bed_df['start']).sum()

    bam_full_path = os.path.join(bamdir, bamfile)
    bam = pysam.AlignmentFile(bam_full_path, "rb")

    sample_out_dir = os.path.join(workdir, "QC_metrics", biosample_id,
                                   "chromosome_regions")
    os.makedirs(sample_out_dir, exist_ok=True)

    # Timing
    start_time = time.time()
    bam_load_time = start_time   # same as start in original

    current_chrom = None
    total_bases_aligned = 0
    coverage_values = []
    total_positions = 0
    bases_5x = bases_10x = bases_15x = bases_20x = bases_25x = bases_30x = 0

    # Mark start of coverage calculation
    coverage_start_time = time.time()

    for _, row in tqdm(bed_df.iterrows(), total=len(bed_df),
                       desc=f"Processing {biosample_id}"):
        chrom, start, end = row['chrom'], row['start'], row['end']

        if current_chrom is not None and chrom != current_chrom:
            _write_chromosome_metrics(
                sample_out_dir, biosample_id, current_chrom, total_loci,
                total_bases_aligned, total_positions, coverage_values,
                bases_5x, bases_10x, bases_15x, bases_20x, bases_25x, bases_30x
            )
            total_bases_aligned = 0
            coverage_values = []
            total_positions = 0
            bases_5x = bases_10x = bases_15x = bases_20x = bases_25x = bases_30x = 0

        current_chrom = chrom

        for pileupcol in bam.pileup(chrom, start, end, truncate=True):
            cov = pileupcol.n
            total_bases_aligned += cov
            coverage_values.append(cov)
            total_positions += 1
            if cov >= 5:
                bases_5x += 1
            if cov >= 10:
                bases_10x += 1
            if cov >= 15:
                bases_15x += 1
            if cov >= 20:
                bases_20x += 1
            if cov >= 25:
                bases_25x += 1
            if cov >= 30:
                bases_30x += 1

    coverage_end_time = time.time()

    if current_chrom:
        _write_chromosome_metrics(
            sample_out_dir, biosample_id, current_chrom, total_loci,
            total_bases_aligned, total_positions, coverage_values,
            bases_5x, bases_10x, bases_15x, bases_20x, bases_25x, bases_30x
        )

    bam.close()
    end_time = time.time()

    # Write log file with all timings (exactly as original)
    log_path = os.path.join(workdir, "QC_metrics", biosample_id, "log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Program started at: {time.ctime(start_time)}\n")
        f.write(f"BAM file loading time at: {time.ctime(bam_load_time)}\n")
        f.write(f"Coverage calculation started at: {time.ctime(coverage_start_time)}\n")
        f.write(f"Coverage calculation completed at: {time.ctime(coverage_end_time)}\n")
        f.write(f"Total time taken in coverage calculation: {coverage_end_time - coverage_start_time:.4f} seconds\n")
        f.write(f"Program ended at: {time.ctime(end_time)}\n")
        f.write(f"Total execution time: {end_time - start_time:.4f} seconds\n")


def process_chunk(chunk_df, workdir, bamdir):
    """
    Process a chunk of samples (called in parallel by a Pool).

    Args:
        chunk_df: DataFrame subset with columns 'Library_ID' and 'bampath'
        workdir: Pipeline working directory
        bamdir: Directory where BAMs are stored (may be overridden by bampath)

    Returns:
        Number of samples processed in this chunk.
    """
    for _, row in chunk_df.iterrows():
        lib_id = row['Library_ID']
        bam_path = row['bampath']
        bam_file = f"{lib_id}.recal.bam"
        if not os.path.exists(os.path.join(bam_path, bam_file)):
            logger.error(f"BAM file missing for {lib_id} in {bam_path}")
            continue
        logger.info(f"Processing QC metrics for {lib_id}")
        process_sample_qc(workdir, lib_id, bam_path, bam_file)
    return len(chunk_df)


def run_qc(cfg, sample_df):
    """
    Main entry point for QC metrics.

    - Creates autosomal BED if not already present.
    - Splits the sample DataFrame into chunks and processes them in parallel.

    Args:
        cfg: Configuration dictionary
        sample_df: DataFrame with sample information (must contain 'Library_ID' and 'bampath')
    """
    bed_path = os.path.join(cfg['workdir'], 'sorted_autosomes.bed')
    if not os.path.exists(bed_path):
        create_autosome_bed(cfg)
    else:
        logger.info(f"Using existing autosomal BED: {bed_path}")

    num_proc = min(int(cfg.get('num_processes', 4)), len(sample_df))
    utils.process_chunks_parallel(sample_df, num_proc, process_chunk,
                                  (cfg['workdir'], cfg['bamdir']))