#!/usr/bin/env python3
"""
Generate a QC summary table (per‑metric) with observed ranges, pass/fail counts,
and percentages across all samples. Uses configuration file.

Usage:
    python generate_qc_summary.py [config_file]

Configuration file (default: summary_config.ini) must contain:
    [Paths]
    qc_output_dir = /path/to/qc_output
    multiqc_data_dir = /path/to/multiqc_data
    qc_metrics_dir = /path/to/QC_metrics        # for QC_metricses_data_all_samples.tsv
    output_file = /path/to/output.csv
"""

import argparse
import configparser
import csv
import re
import math
import ast
import statistics
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------------------
# Parsers for each file type – now include status fields where available
# ------------------------------------------------------------------------------

def parse_multiqc_general_stats(file_path):
    """
    Parse multiqc_general_stats.txt, group rows by base sample name,
    and return a dict sample -> {metric: value} for duplication and mapping percentages.
    Tries multiple possible column name patterns to accommodate different MultiQC versions.
    Prints a warning if no expected columns are found.
    """
    samples = defaultdict(dict)
    found_any = False

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        # Uncomment next line to see actual column names in your file
        # print("Header:", reader.fieldnames)

        for row in reader:
            sample_raw = row.get('Sample', '')
            is_md = '.md' in sample_raw
            is_recal = '.recal' in sample_raw
            base = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample_raw)

            if base not in samples or (is_md and not samples[base].get('_prefer_md')):
                # --- Picard duplication: try multiple possible column names ---
                dup = None
                possible_dup_cols = [
                    'picard_mark_duplicates-PERCENT_DUPLICATION',
                    'Picard: Mark Duplicates_mqc-generalstats-picard_mark_duplicates-PERCENT_DUPLICATION',
                    'PERCENT_DUPLICATION',
                    'Mark Duplicates_mqc-generalstats-picard_mark_duplicates-PERCENT_DUPLICATION'
                ]
                for col in possible_dup_cols:
                    if col in row and row[col] and row[col].strip():
                        dup = row[col]
                        break
                if dup:
                    try:
                        samples[base]['percent_duplication'] = float(dup)
                        found_any = True
                    except ValueError:
                        pass

                # --- Samtools percent mapped ---
                mapped_pct = None
                possible_mapped_cols = [
                    'samtools_stats-reads_mapped_percent',
                    'Samtools: stats_mqc-generalstats-samtools_stats-reads_mapped_percent',
                    'reads_mapped_percent'
                ]
                for col in possible_mapped_cols:
                    if col in row and row[col] and row[col].strip():
                        mapped_pct = row[col]
                        break
                if mapped_pct:
                    try:
                        samples[base]['percent_mapped'] = float(mapped_pct)
                        found_any = True
                    except ValueError:
                        pass

                # --- Samtools percent properly paired ---
                properly_paired = None
                possible_paired_cols = [
                    'samtools_stats-reads_properly_paired_percent',
                    'Samtools: stats_mqc-generalstats-samtools_stats-reads_properly_paired_percent',
                    'reads_properly_paired_percent'
                ]
                for col in possible_paired_cols:
                    if col in row and row[col] and row[col].strip():
                        properly_paired = row[col]
                        break
                if properly_paired:
                    try:
                        samples[base]['percent_properly_paired'] = float(properly_paired)
                        found_any = True
                    except ValueError:
                        pass

                if is_md:
                    samples[base]['_prefer_md'] = True

    if not found_any:
        print(f"Warning: No expected columns found in {file_path}. Check the column names.")

    return samples


def parse_samtools_alignment_plot(file_path):
    """
    Parse samtools_alignment_plot.txt and return total mapped reads per sample.
    Values may be given as floats (e.g., '908495943.0'), so we convert via float.
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample_raw = row['Sample']
            sample = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample_raw)
            # Use float conversion to handle possible decimal strings, then cast to int
            mapped_mq0_str = row.get('Mapped (with MQ>0)', '0')
            mq0_str = row.get('MQ0', '0')
            try:
                mapped_mq0 = int(float(mapped_mq0_str))
            except ValueError:
                mapped_mq0 = 0
            try:
                mq0 = int(float(mq0_str))
            except ValueError:
                mq0 = 0
            data[sample] = mapped_mq0 + mq0
    return data


def parse_autosomal_coverage_samples(file_path):
    """
    Parse Autosomal_Coverage_Samples_report.tsv and include status fields.
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample = row['Sample']
            data[sample] = {
                'mean_coverage': float(row['Mean_alignment_coverage_over_autosomal_loci']),
                'median_coverage': int(float(row['Median_alignment_coverage_over_autosomal_loci'])),
                'pct_15x': float(row['Percent_autosome_coverage_at_15X']),
                'pct_30x': float(row['Percent_autosome_coverage_at_30X']),
                # Status fields
                'status_pct30': row['Status_pct30'],
            }
    return data


def parse_contamination_report(file_path):
    """Parse Samples_Contamination_report.tsv including statuses."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample = row['Sample']
            data[sample] = {
                'avg_autosomal_coverage': float(row['Average_autosomal_coverage']),
                'freemix': float(row['freemix']),
                'avg_dp': float(row['avg_dp']),
                'status_freemix': row['Status_freemix'],
                'status_avgdp': row['Status_avgdp'],
            }
    return data


def parse_base_quality(file_path):
    """Parse base_quality_report.tsv and average R1/R2 for each sample."""
    sample_quals = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            biosample = row['Biosample_id']
            sample = re.sub(r'(-L001_[12]|-[LR][0-9]+).*', '', biosample)
            mean_q = float(row['mean_quality'])
            sample_quals[sample].append(mean_q)
    return {s: {'mean_base_quality': sum(vals)/len(vals)} for s, vals in sample_quals.items()}


def parse_fastp_insert_size(file_path):
    """
    Parse fastp-insert-size-plot.txt and compute per‑sample mean and std dev.
    """
    sample_stats = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        insert_sizes = [int(h) for h in header[1:] if h.isdigit()]
        for line in f:
            parts = line.strip().split('\t')
            sample_lane = parts[0]
            sample = re.sub(r'(-L001|-[LR][0-9]+).*', '', sample_lane)
            fractions = []
            for i, val in enumerate(parts[1:]):
                if i >= len(insert_sizes):
                    break
                try:
                    frac_str = val.strip('()').split(',')[1].strip()
                    fraction = float(frac_str)
                    fractions.append(fraction)
                except (ValueError, IndexError):
                    fractions.append(0.0)
            if fractions and sum(fractions) > 0:
                total = sum(fractions)
                norm = [f/total for f in fractions]
                mean = sum(isize * f for isize, f in zip(insert_sizes, norm))
                variance = sum(f * (isize - mean)**2 for isize, f in zip(insert_sizes, norm))
                std = math.sqrt(variance)
                sample_stats[sample].append((mean, std))
    result = {}
    for sample, stats_list in sample_stats.items():
        means = [m for m, _ in stats_list]
        stds = [s for _, s in stats_list]
        result[sample] = {
            'insert_size_mean': sum(means) / len(means),
            'insert_size_std': sum(stds) / len(stds)
        }
    return result


# ------------------------------------------------------------------------------
# New parsers for the missing metrics
# ------------------------------------------------------------------------------

def parse_cumulative_coverage(file_path):
    """
    Parse Cumulative_Coverage_Samples_report.tsv.
    Returns dict sample -> {cov5, cov10, cov15, cov20, cov25, cov30}
    Prefers .md over .recal rows.
    """
    samples = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample_raw = row['Biosample_id']
            base = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample_raw)
            is_md = '.md' in sample_raw
            if base not in samples or (is_md and not samples[base].get('_prefer_md')):
                samples[base] = {
                    'cov5': float(row['Coverage_at_5X']),
                    'cov10': float(row['Coverage_at_10X']),
                    'cov15': float(row['Coverage_at_15X']),
                    'cov20': float(row['Coverage_at_20X']),
                    'cov25': float(row['Coverage_at_25X']),
                    'cov30': float(row['Coverage_at_30X']),
                    '_prefer_md': is_md
                }
    return samples


def parse_per_contig_coverage(file_path):
    """
    Parse mosdepth-coverage-per-contig-multi.txt.
    For each sample, extract coverage of autosomes (chr1..chr22) and compute
    coefficient of variation (CV). Prefers .recal rows because .md rows are incomplete.
    Returns dict sample -> {'autosome_cv': cv}
    """
    samples = defaultdict(dict)   # base -> {chrom: coverage}
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()   # skip header
        for line in f:
            parts = line.strip().split('\t')
            sample_raw = parts[0]
            base = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample_raw)
            is_recal = '.recal' in sample_raw
            # Prefer recal over md
            if base not in samples or (is_recal and not samples[base].get('_prefer_recal')):
                chrom_data = {}
                for col in parts[1:]:
                    if not col.strip():
                        continue
                    try:
                        chrom, cov = ast.literal_eval(col)
                        chrom_data[chrom] = cov
                    except:
                        pass
                samples[base] = chrom_data
                samples[base]['_prefer_recal'] = is_recal

    result = {}
    autosomes = {f'chr{i}' for i in range(1, 23)}
    for base, chrom_data in samples.items():
        coverages = []
        for chrom in autosomes:
            if chrom in chrom_data:
                coverages.append(chrom_data[chrom])
        # Expect all 22 autosomes if recal row was used
        if len(coverages) >= 22:
            mean = statistics.mean(coverages)
            stdev = statistics.stdev(coverages)
            cv = stdev / mean
            result[base] = {'autosome_cv': cv}
        # else skip (should not happen for recal rows)
    return result


def parse_qc_metrics_all(file_path):
    """
    Parse QC_metricses_data_all_samples.tsv.
    Returns dict sample -> {'mean_median_ratio': value}
    Sample names are already base names (no suffixes).
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample = row['Biosample_id']
            # Some sample names might have suffixes? We strip to be safe
            sample = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample)
            data[sample] = {
                'mean_median_ratio': float(row['Mean_by_Median_autosomal_coverage_ratio_over_region'])
            }
    return data


# ------------------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate QC summary table with pass/fail statistics.'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='summary_config.ini',
        help='Path to configuration file (default: summary_config.ini)'
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if not Path(args.config_file).is_file():
        raise FileNotFoundError(f"Configuration file '{args.config_file}' not found.")
    config.read(args.config_file)

    try:
        qc_output_dir = config.get('Paths', 'qc_output_dir')
        multiqc_data_dir = config.get('Paths', 'multiqc_data_dir')
        qc_metrics_dir = config.get('Paths', 'qc_metrics_dir')
        output_file = config.get('Paths', 'output_file')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise KeyError("Missing section or key in config file. Ensure [Paths] contains "
                       "qc_output_dir, multiqc_data_dir, qc_metrics_dir, and output_file.") from e

    qc_output_path = Path(qc_output_dir)
    multiqc_path = Path(multiqc_data_dir)
    qc_metrics_path = Path(qc_metrics_dir)
    output_path = Path(output_file)

    # Locate files, warn if missing
    multiqc_stats = multiqc_path / 'multiqc_general_stats.txt'
    if not multiqc_stats.exists():
        print(f"Warning: {multiqc_stats} not found; mapping percentages will be omitted.")
        multiqc_stats = None

    samtools_align = multiqc_path / 'samtools_alignment_plot.txt'
    if not samtools_align.exists():
        print(f"Warning: {samtools_align} not found; reads mapped counts will be omitted.")
        samtools_align = None

    cov_samples = qc_output_path / 'Autosomal_Coverage_Samples_report.tsv'
    if not cov_samples.exists():
        print(f"Warning: {cov_samples} not found; coverage metrics will be omitted.")
        cov_samples = None

    contam = qc_output_path / 'Samples_Contamination_report.tsv'
    if not contam.exists():
        print(f"Warning: {contam} not found; contamination metrics will be omitted.")
        contam = None

    base_qual = qc_output_path / 'base_quality_report.tsv'
    if not base_qual.exists():
        print(f"Warning: {base_qual} not found; base quality metrics will be omitted.")
        base_qual = None

    insert_plot = multiqc_path / 'fastp-insert-size-plot.txt'
    if not insert_plot.exists():
        print(f"Warning: {insert_plot} not found; insert size metrics will be omitted.")
        insert_plot = None

    # New files for missing metrics
    qc_metrics_all = qc_metrics_path / 'QC_metricses_data_all_samples.tsv'
    if not qc_metrics_all.exists():
        print(f"Warning: {qc_metrics_all} not found; mean/median ratio will be omitted.")
        qc_metrics_all = None

    cum_cov_file = qc_output_path / 'Cumulative_Coverage_Samples_report.tsv'
    if not cum_cov_file.exists():
        print(f"Warning: {cum_cov_file} not found; cumulative coverage metrics will be omitted.")
        cum_cov_file = None

    per_contig_file = multiqc_path / 'mosdepth-coverage-per-contig-multi.txt'
    if not per_contig_file.exists():
        print(f"Warning: {per_contig_file} not found; chromosome imbalance metrics will be omitted.")
        per_contig_file = None

    # Parse all files
    multiqc_data = parse_multiqc_general_stats(multiqc_stats) if multiqc_stats else {}
    mapped_reads_data = parse_samtools_alignment_plot(samtools_align) if samtools_align else {}
    cov_data = parse_autosomal_coverage_samples(cov_samples) if cov_samples else {}
    contam_data = parse_contamination_report(contam) if contam else {}
    base_data = parse_base_quality(base_qual) if base_qual else {}
    insert_data = parse_fastp_insert_size(insert_plot) if insert_plot else {}
    # New parsers
    qc_all_data = parse_qc_metrics_all(qc_metrics_all) if qc_metrics_all else {}
    cum_cov_data = parse_cumulative_coverage(cum_cov_file) if cum_cov_file else {}
    per_contig_data = parse_per_contig_coverage(per_contig_file) if per_contig_file else {}

    # Merge all data per sample
    all_samples = set(cov_data.keys()) | set(contam_data.keys()) | set(base_data.keys()) | \
                  set(multiqc_data.keys()) | set(mapped_reads_data.keys()) | set(insert_data.keys()) | \
                  set(qc_all_data.keys()) | set(cum_cov_data.keys()) | set(per_contig_data.keys())
    sample_data = {}
    for sample in all_samples:
        d = {}
        d.update(cov_data.get(sample, {}))
        d.update(contam_data.get(sample, {}))
        d.update(base_data.get(sample, {}))
        d.update(multiqc_data.get(sample, {}))
        d.update(insert_data.get(sample, {}))
        d.update(qc_all_data.get(sample, {}))
        d.update(cum_cov_data.get(sample, {}))
        d.update(per_contig_data.get(sample, {}))
        if sample in mapped_reads_data:
            d['reads_mapped'] = mapped_reads_data[sample]
        sample_data[sample] = d

    total_samples = len(sample_data)

    # --------------------------------------------------------------------------
    # Define metrics with pass/fail evaluation and source file
    # Each entry now includes a "type_category" field for the new Type column.
    # --------------------------------------------------------------------------
    metric_defs = [
        # Depth
        {
            "group": "Depth",
            "type": "Mosdepth Median coverage",
            "type_category": "Average coverage per contig",
            "source": "Median_alignment_coverage_over_autosomal_loci (Autosomal_Coverage_Samples)",
            "source_file": "qc_output/Autosomal_Coverage_Samples_report.tsv",
            "threshold": "Not specified (all pass)",
            "key": "median_coverage",
            "fmt": lambda v: f"{min(v):.0f} – {max(v):.0f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Depth",
            "type": "Mosdepth Mean coverage",
            "type_category": "Average coverage per contig",
            "source": "Mean_alignment_coverage_over_autosomal_loci (Autosomal_Coverage_Samples)",
            "source_file": "qc_output/Autosomal_Coverage_Samples_report.tsv",
            "threshold": "Not specified (all pass)",
            "key": "mean_coverage",
            "fmt": lambda v: f"{min(v):.2f} – {max(v):.2f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Depth",
            "type": "VerifyBAMID average depth",
            "type_category": "Average coverage per contig",
            "source": "avg_dp (Samples_Contamination)",
            "source_file": "qc_output/Samples_Contamination_report.tsv",
            "threshold": "Not specified (all pass)",
            "key": "avg_dp",
            "fmt": lambda v: f"{min(v):.1f} – {max(v):.1f}",
            "pass_eval": lambda val, status_dict: True
        },
        # Coverage
        {
            "group": "Coverage",
            "type": "Mean autosome coverage",
            "type_category": "Average coverage per contig",
            "source": "Average_autosomal_coverage (Samples_Contamination)",
            "source_file": "qc_output/Samples_Contamination_report.tsv",
            "threshold": "Not specified (all pass)",
            "key": "avg_autosomal_coverage",
            "fmt": lambda v: f"{min(v):.2f} – {max(v):.2f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Percent autosomes covered at 30X",
            "type_category": "Percentage autosome coverage",
            "source": "Percent_autosome_coverage_at_30X (Autosomal_Coverage_Samples)",
            "source_file": "qc_output/Autosomal_Coverage_Samples_report.tsv",
            "threshold": ">90% (typical)",
            "key": "pct_30x",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: status_dict.get('status_pct30', '') == 'Pass'
        },
        {
            "group": "Coverage",
            "type": "Percent autosomes covered at 15X",
            "type_category": "Percentage autosome coverage",
            "source": "Percent_autosome_coverage_at_15X (Autosomal_Coverage_Samples)",
            "source_file": "qc_output/Autosomal_Coverage_Samples_report.tsv",
            "threshold": ">98%",
            "key": "pct_15x",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        # Cumulative coverage rows
        {
            "group": "Coverage",
            "type": "Cumulative coverage at 5X (genome-wide)",
            "type_category": "Cumulative coverage distribution",
            "source": "Coverage_at_5X (Cumulative_Coverage_Samples_report.tsv)",
            "source_file": "qc_output/Cumulative_Coverage_Samples_report.tsv",
            "threshold": "Not specified",
            "key": "cov5",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Cumulative coverage at 10X (genome-wide)",
            "type_category": "Cumulative coverage distribution",
            "source": "Coverage_at_10X (Cumulative_Coverage_Samples_report.tsv)",
            "source_file": "qc_output/Cumulative_Coverage_Samples_report.tsv",
            "threshold": "Not specified",
            "key": "cov10",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Cumulative coverage at 15X (genome-wide)",
            "type_category": "Cumulative coverage distribution",
            "source": "Coverage_at_15X (Cumulative_Coverage_Samples_report.tsv)",
            "source_file": "qc_output/Cumulative_Coverage_Samples_report.tsv",
            "threshold": "Not specified",
            "key": "cov15",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Cumulative coverage at 20X (genome-wide)",
            "type_category": "Cumulative coverage distribution",
            "source": "Coverage_at_20X (Cumulative_Coverage_Samples_report.tsv)",
            "source_file": "qc_output/Cumulative_Coverage_Samples_report.tsv",
            "threshold": "Not specified",
            "key": "cov20",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Cumulative coverage at 25X (genome-wide)",
            "type_category": "Cumulative coverage distribution",
            "source": "Coverage_at_25X (Cumulative_Coverage_Samples_report.tsv)",
            "source_file": "qc_output/Cumulative_Coverage_Samples_report.tsv",
            "threshold": "Not specified",
            "key": "cov25",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Cumulative coverage at 30X (genome-wide)",
            "type_category": "Cumulative coverage distribution",
            "source": "Coverage_at_30X (Cumulative_Coverage_Samples_report.tsv)",
            "source_file": "qc_output/Cumulative_Coverage_Samples_report.tsv",
            "threshold": "Not specified",
            "key": "cov30",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Mean/Median autosome coverage ratio",
            "type_category": "Mean median ratio of autosome coverage",
            "source": "Mean_by_Median_autosomal_coverage_ratio_over_region (QC_metricses_data_all_samples.tsv)",
            "source_file": "QC_metrics/QC_metricses_data_all_samples.tsv",
            "threshold": "Not specified",
            "key": "mean_median_ratio",
            "fmt": lambda v: f"{min(v):.4f} – {max(v):.4f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Chromosome coverage imbalance (CV)",
            "type_category": "Chromosome coverage imbalance",
            "source": "mosdepth-coverage-per-contig-multi.txt (autosomes)",
            "source_file": "multiqc_data/mosdepth-coverage-per-contig-multi.txt",
            "threshold": "<0.1",
            "key": "autosome_cv",
            "fmt": lambda v: f"{min(v):.4f} – {max(v):.4f}",
            "pass_eval": lambda val, status_dict: val < 0.1
        },
        # Alignment and preprocessing
        {
            "group": "Alignment and preprocessing",
            "type": "% Duplication",
            "type_category": "Percent reads properly paired",   # grouped under alignment
            "source": "Picard: Mark Duplicates PERCENT_DUPLICATION (multiqc_general_stats.txt)",
            "source_file": "multiqc_data/multiqc_general_stats.txt",
            "threshold": "<20%",
            "key": "percent_duplication",
            "fmt": lambda v: f"{min(v):.1f}% – {max(v):.1f}%",
            "pass_eval": lambda val, status_dict: val < 20.0
        },
        {
            "group": "Alignment and preprocessing",
            "type": "Reads mapped",
            "type_category": "Percent reads properly paired",   # grouped under alignment
            "source": "Samtools alignment plot (Mapped with MQ>0 + MQ0)",
            "source_file": "multiqc_data/samtools_alignment_plot.txt",
            "threshold": "–",
            "key": "reads_mapped",
            "fmt": lambda v: f"{min(v)/1e6:.0f} – {max(v)/1e6:.0f} million reads",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Alignment and preprocessing",
            "type": "% Mapped",
            "type_category": "Percent reads properly paired",   # grouped under alignment
            "source": "Samtools reads_mapped_percent (multiqc_general_stats.txt, .md/.recal)",
            "source_file": "multiqc_data/multiqc_general_stats.txt",
            "threshold": ">95%",
            "key": "percent_mapped",
            "fmt": lambda v: f"{min(v):.2f}% – {max(v):.2f}%",
            "pass_eval": lambda val, status_dict: val > 95.0
        },
        {
            "group": "Alignment and preprocessing",
            "type": "% Reads properly paired",
            "type_category": "Percent reads properly paired",
            "source": "Samtools reads_properly_paired_percent (multiqc_general_stats.txt, .md/.recal)",
            "source_file": "multiqc_data/multiqc_general_stats.txt",
            "threshold": ">90%",
            "key": "percent_properly_paired",
            "fmt": lambda v: f"{min(v):.2f}% – {max(v):.2f}%",
            "pass_eval": lambda val, status_dict: val > 90.0
        },
        {
            "group": "Alignment and preprocessing",
            "type": "Insert size standard deviation",
            "type_category": "Insert size standard deviation",
            "source": "Derived from fastp-insert-size-plot.txt",
            "source_file": "multiqc_data/fastp-insert-size-plot.txt",
            "threshold": "–",
            "key": "insert_size_std",
            "fmt": lambda v: f"{min(v):.1f} – {max(v):.1f} bp",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Alignment and preprocessing",
            "type": "Mean insert size",
            "type_category": "Mean insert size",
            "source": "Derived from fastp-insert-size-plot.txt",
            "source_file": "multiqc_data/fastp-insert-size-plot.txt",
            "threshold": "–",
            "key": "insert_size_mean",
            "fmt": lambda v: f"{min(v):.0f} – {max(v):.0f} bp",
            "pass_eval": lambda val, status_dict: True
        },
        # Cross contamination
        {
            "group": "Cross contamination",
            "type": "Cross contamination (freemix)",
            "type_category": "Sample Contamination",
            "source": "freemix (Samples_Contamination)",
            "source_file": "qc_output/Samples_Contamination_report.tsv",
            "threshold": "<0.05",
            "key": "freemix",
            "fmt": lambda v: f"{min(v):.2e} – {max(v):.2e}",
            "pass_eval": lambda val, status_dict: status_dict.get('status_freemix', '') == 'Pass'
        },
        # Sequence quality
        {
            "group": "Sequence quality",
            "type": "Mean base quality (Phred score)",
            "type_category": "Base quality (Phred score)",
            "source": "mean_quality (base_quality_report)",
            "source_file": "qc_output/base_quality_report.tsv",
            "threshold": ">30",
            "key": "mean_base_quality",
            "fmt": lambda v: f"{min(v):.1f} – {max(v):.1f}",
            "pass_eval": lambda val, status_dict: val > 30.0
        }
    ]

    # Collect values and pass/fail counts for each metric key
    values_by_key = defaultdict(list)
    pass_counts = defaultdict(int)
    fail_counts = defaultdict(int)

    for sample, data in sample_data.items():
        for key in [m['key'] for m in metric_defs if m['key']]:
            if key in data:
                val = data[key]
                values_by_key[key].append(val)
                # Determine pass/fail
                mdef = next((m for m in metric_defs if m.get('key') == key), None)
                if mdef:
                    if mdef['pass_eval'](val, data):
                        pass_counts[key] += 1
                    else:
                        fail_counts[key] += 1

    # Build output rows
    output_rows = []
    for mdef in metric_defs:
        key = mdef.get('key')
        vals = values_by_key.get(key, [])
        if vals:
            observed = mdef['fmt'](vals)
        else:
            observed = 'Not available'

        pass_cnt = pass_counts.get(key, 0)
        fail_cnt = fail_counts.get(key, 0)
        total = pass_cnt + fail_cnt
        if total > 0:
            pass_pct = (pass_cnt / total) * 100
            fail_pct = (fail_cnt / total) * 100
        else:
            pass_pct = fail_pct = 0.0

        row = {
            'QC Group': mdef['group'],
            'Type': mdef['type_category'],          # new column
            'QC Type': mdef['type'],
            'Metric (from report)': mdef['source'],
            'Source File': mdef.get('source_file', ''),
            'Expected Threshold': mdef['threshold'],
            'Observed Value (range)': observed,
            'Pass Count': pass_cnt,
            'Fail Count': fail_cnt,
            'Total Samples': total,
            'Pass %': f"{pass_pct:.1f}%" if pass_cnt > 0 else "0%",
            'Fail %': f"{fail_pct:.1f}%" if fail_cnt > 0 else "0%"
        }
        output_rows.append(row)

    # Add a placeholder row for Per Sequence Quality (already covered by mean quality)
    output_rows.append({
        'QC Group': 'Sequence quality',
        'Type': 'Base quality (Phred score)',
        'QC Type': 'Per Sequence Quality',
        'Metric (from report)': 'Covered by mean quality above',
        'Source File': '',
        'Expected Threshold': '–',
        'Observed Value (range)': '–',
        'Pass Count': '–',
        'Fail Count': '–',
        'Total Samples': '–',
        'Pass %': '–',
        'Fail %': '–'
    })

    # Write CSV with UTF-8 BOM
    fieldnames = [
        'QC Group', 'Type', 'QC Type', 'Metric (from report)', 'Source File',
        'Expected Threshold', 'Observed Value (range)', 'Pass Count', 'Fail Count',
        'Total Samples', 'Pass %', 'Fail %'
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"QC summary table written to {output_path}")
    print(f"Processed {total_samples} samples.")

if __name__ == '__main__':
    main()