#!/usr/bin/env python3
"""
QC Report Generator for WGS/WES Pipeline

This script reads:
- Per‑sample per‑chromosome JSON files (from the pipeline's QC_metrics directory)
- The sample‑level summary TSV (QC_metricses_data_all_samples.tsv)
- MultiQC output directory

and produces:
    - Summary tables (TSV) for samples, chromosomes, contamination, coverage, etc.
    - An interactive HTML report with all plots (using Plotly).

Usage:
    python generate_qc_report.py --qc-dir <path> --multiqc-dir <path> --outdir <path> [--sample-summary <path>] [--config <path>]

Arguments:
    --qc-dir           Path to the QC_metrics directory (contains sample subfolders with merged JSONs)
    --multiqc-dir      Path to the MultiQC output directory (contains mosdepth, samtools, etc. subdirs)
    --outdir           Directory where all reports and plots will be saved
    --sample-summary   Optional path to the sample‑level TSV (if not given, assumed to be QC_metricses_data_all_samples.tsv inside --qc-dir)
    --config           Optional configuration file with thresholds (see example below)
"""

import os
import sys
import argparse
import logging
import configparser
import json
import glob
import re
from statistics import median, mean, variance, stdev
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate QC report from pipeline outputs.')
    parser.add_argument('--qc-dir', required=True,
                        help='Path to the QC_metrics directory (contains sample subfolders)')
    parser.add_argument('--multiqc-dir', required=True,
                        help='MultiQC output directory (contains mosdepth, samtools, etc.)')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for reports and plots')
    parser.add_argument('--sample-summary',
                        help='Optional path to the sample‑level TSV (default: QC_metricses_data_all_samples.tsv inside --qc-dir)')
    parser.add_argument('--config', help='Optional configuration file with thresholds')
    return parser.parse_args()


def load_config(config_path):
    """Load thresholds from a config file (INI format)."""
    defaults = {
        'autosomal_coverage_cutoff': 30,
        'percent_coverage_cutoff': 90,
        'mean_median_ratio_cutoff': 1.5,
        'freemix_cutoff': 0.01,
        'mapped_percent_cutoff': 95,
        'base_quality_cutoff': 30,
        'cv_cutoffs': [5, 10, 15, 20, 25, 30]   # for chromosome coverage imbalance
    }
    if config_path and os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if 'THRESHOLDS' in config:
            for key in defaults:
                if key in config['THRESHOLDS']:
                    try:
                        if key == 'cv_cutoffs':
                            defaults[key] = [int(x.strip()) for x in config['THRESHOLDS'][key].split(',')]
                        else:
                            defaults[key] = float(config['THRESHOLDS'][key])
                    except ValueError:
                        logger.warning(f"Invalid value for {key} in config, using default {defaults[key]}")
    return defaults


def setup_output_dirs(outdir):
    """Create output directory and return path for HTML report."""
    os.makedirs(outdir, exist_ok=True)
    html_path = os.path.join(outdir, 'qc_report.html')
    # If file exists, remove it so we start fresh
    if os.path.exists(html_path):
        os.remove(html_path)
    return html_path


def find_column(df, patterns):
    """
    Find a column in the DataFrame that matches any of the given patterns (case‑insensitive).
    Patterns can be exact strings or substrings. Returns the first matching column name.
    Raises KeyError if no match is found.
    """
    for pattern in patterns:
        # Try exact match (case‑insensitive)
        exact_matches = [col for col in df.columns if col.lower() == pattern.lower()]
        if exact_matches:
            return exact_matches[0]
        # Try substring match
        substring_matches = [col for col in df.columns if pattern.lower() in col.lower()]
        if substring_matches:
            return substring_matches[0]
    raise KeyError(f"None of the patterns {patterns} found in DataFrame columns: {list(df.columns)}")


def load_qc_data(qc_dir, sample_summary_path=None):
    """
    Load all per‑sample merged JSON files and the sample‑level summary TSV,
    then combine them into a single DataFrame with both chromosome‑level and
    sample‑level columns.
    """
    # Find all merged JSON files (one per sample)
    json_pattern = os.path.join(qc_dir, '*', '*_merged_all_chrom_qc_metrics.json')
    json_files = glob.glob(json_pattern)
    if not json_files:
        raise FileNotFoundError(f"No merged JSON files found in {qc_dir} using pattern {json_pattern}")

    logger.info(f"Found {len(json_files)} per‑sample merged JSON files.")

    # Determine sample summary path
    if sample_summary_path is None:
        sample_summary_path = os.path.join(qc_dir, 'QC_metricses_data_all_samples.tsv')
    if not os.path.exists(sample_summary_path):
        raise FileNotFoundError(f"Sample summary file not found: {sample_summary_path}")

    # Load sample‑level summary
    sample_df = pd.read_csv(sample_summary_path, sep='\t')
    # Ensure Biosample_id column is present and named consistently
    if 'Biosample_id' not in sample_df.columns:
        # Try to find a column that might be the sample ID
        possible = ['Sample', 'sample', 'Library_ID']
        for col in possible:
            if col in sample_df.columns:
                sample_df.rename(columns={col: 'Biosample_id'}, inplace=True)
                break
    if 'Biosample_id' not in sample_df.columns:
        raise KeyError("Sample summary file must contain a 'Biosample_id' column.")

    # Load each per‑sample JSON and merge with sample_df
    all_chrom_dfs = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        df_chrom = pd.DataFrame(data)
        # Ensure Biosample_id column exists
        if 'Biosample_id' not in df_chrom.columns:
            # Try to infer from filename or JSON content
            if len(df_chrom) > 0 and 'Biosample_id' in df_chrom.columns:
                pass
            else:
                # Fallback: extract from filename (e.g., WHB14582_merged_all_chrom_qc_metrics.json)
                base = os.path.basename(jf)
                sample_id = base.split('_')[0]
                df_chrom['Biosample_id'] = sample_id
        # Merge with sample_df to add sample‑level columns (e.g., Average_autosomal_coverage, freemix, avg_dp)
        # Use left join to keep all chromosome rows
        merged = df_chrom.merge(sample_df, on='Biosample_id', how='left', suffixes=('', '_sample'))
        all_chrom_dfs.append(merged)

    full_df = pd.concat(all_chrom_dfs, ignore_index=True)

    # Convert Chrom column to integer for sorting (remove 'chr' prefix if present)
    # First, find the chromosome column
    chrom_col = find_column(full_df, ['Chrom', 'Chromosome', 'chr'])
    full_df.rename(columns={chrom_col: 'Chrom'}, inplace=True)
    full_df['Chrom_num'] = full_df['Chrom'].astype(str).str.replace('chr', '').str.extract(r'(\d+)').astype(int)

    # Create a display version with 'chr' prefix
    full_df_display = full_df.copy()
    full_df_display['Chrom'] = 'chr' + full_df_display['Chrom_num'].astype(str)

    return full_df, full_df_display


# ------------------------------------------------------------------------------
# Summary functions (adapted to work with the combined DataFrame)
# ------------------------------------------------------------------------------

def sample_based_summary(df, thresholds):
    """
    Compute per‑sample summary statistics and outlier status.
    Returns a DataFrame with one row per sample.
    """
    # Identify required columns
    mean_cov_col = find_column(df, ['Mean_alignment_coverage_over_autosomal_loci'])
    median_cov_col = find_column(df, ['Median_alignment_coverage_over_autosomal_loci'])
    pct15_col = find_column(df, ['Percent_autosome_coverage_at_15X'])
    pct30_col = find_column(df, ['Percent_autosome_coverage_at_30X'])
    ratio_col = find_column(df, ['Mean_by_Median_autosomal_coverage_ratio_over_region'])

    results = []
    for sample in df['Biosample_id'].unique():
        sub = df[df['Biosample_id'] == sample]
        mean_cov = mean(sub[mean_cov_col])
        median_cov = round(median(sub[median_cov_col]))
        pct_15x = round(mean(sub[pct15_col]))
        pct_30x = round(mean(sub[pct30_col]))
        mean_median_ratio = mean_cov / median_cov if median_cov else 0

        # Outlier detection
        outlier_mean = sub[sub[mean_cov_col] < thresholds['autosomal_coverage_cutoff']]['Biosample_id'].unique()
        outlier_median = sub[sub[median_cov_col] < thresholds['autosomal_coverage_cutoff']]['Biosample_id'].unique()
        outlier_pct15 = sub[sub[pct15_col] < thresholds['percent_coverage_cutoff']]['Biosample_id'].unique()
        outlier_pct30 = sub[sub[pct30_col] < thresholds['percent_coverage_cutoff']]['Biosample_id'].unique()
        outlier_ratio = sub[sub[ratio_col] > thresholds['mean_median_ratio_cutoff']]['Biosample_id'].unique()

        results.append({
            'Sample': sample,
            'Mean_alignment_coverage_over_autosomal_loci': mean_cov,
            'Median_alignment_coverage_over_autosomal_loci': median_cov,
            'Percent_autosome_coverage_at_15X': pct_15x,
            'Percent_autosome_coverage_at_30X': pct_30x,
            'outlier_mean_coverage': list(outlier_mean),
            'outlier_median_coverage': list(outlier_median),
            'outlier_pct15': list(outlier_pct15),
            'outlier_pct30': list(outlier_pct30),
            'outlier_ratio': list(outlier_ratio),
            'Status_mean': 'Failed' if len(outlier_mean) else 'Pass',
            'Status_median': 'Failed' if len(outlier_median) else 'Pass',
            'Status_pct15': 'Failed' if len(outlier_pct15) else 'Pass',
            'Status_pct30': 'Failed' if len(outlier_pct30) else 'Pass',
            'Status_ratio': 'Failed' if len(outlier_ratio) else 'Pass'
        })
    return pd.DataFrame(results)


def chromosome_based_summary(df, thresholds):
    """
    Compute per‑chromosome summary statistics across all samples.
    Returns a DataFrame with one row per chromosome.
    """
    # Identify required columns (same as above)
    mean_cov_col = find_column(df, ['Mean_alignment_coverage_over_autosomal_loci'])
    median_cov_col = find_column(df, ['Median_alignment_coverage_over_autosomal_loci'])
    pct15_col = find_column(df, ['Percent_autosome_coverage_at_15X'])
    pct30_col = find_column(df, ['Percent_autosome_coverage_at_30X'])
    ratio_col = find_column(df, ['Mean_by_Median_autosomal_coverage_ratio_over_region'])

    results = []
    for chrom_num in df['Chrom_num'].unique():
        sub = df[df['Chrom_num'] == chrom_num]
        mean_cov = mean(sub[mean_cov_col])
        median_cov = round(median(sub[median_cov_col]))
        pct_15x = round(mean(sub[pct15_col]))
        pct_30x = round(mean(sub[pct30_col]))
        mean_median_ratio = mean_cov / median_cov if median_cov else 0

        outlier_mean = sub[sub[mean_cov_col] < thresholds['autosomal_coverage_cutoff']]['Chrom'].unique()
        outlier_median = sub[sub[median_cov_col] < thresholds['autosomal_coverage_cutoff']]['Chrom'].unique()
        outlier_pct15 = sub[sub[pct15_col] < thresholds['percent_coverage_cutoff']]['Chrom'].unique()
        outlier_pct30 = sub[sub[pct30_col] < thresholds['percent_coverage_cutoff']]['Chrom'].unique()
        outlier_ratio = sub[sub[ratio_col] > thresholds['mean_median_ratio_cutoff']]['Chrom'].unique()

        results.append({
            'Chromosome': f'chr{chrom_num}',
            'Mean_alignment_coverage_over_autosomal_loci': mean_cov,
            'Median_alignment_coverage_over_autosomal_loci': median_cov,
            'Percent_autosome_coverage_at_15X': pct_15x,
            'Percent_autosome_coverage_at_30X': pct_30x,
            'outlier_mean_coverage': list(outlier_mean),
            'outlier_median_coverage': list(outlier_median),
            'outlier_pct15': list(outlier_pct15),
            'outlier_pct30': list(outlier_pct30),
            'outlier_ratio': list(outlier_ratio),
            'Status_mean': 'Failed' if len(outlier_mean) else 'Pass',
            'Status_median': 'Failed' if len(outlier_median) else 'Pass',
            'Status_pct15': 'Failed' if len(outlier_pct15) else 'Pass',
            'Status_pct30': 'Failed' if len(outlier_pct30) else 'Pass',
            'Status_ratio': 'Failed' if len(outlier_ratio) else 'Pass'
        })
    return pd.DataFrame(results)


def contamination_summary(df, thresholds):
    """
    Compute per‑sample contamination metrics (Average_autosomal_coverage, freemix, avg_dp).
    Uses the sample‑level columns that were merged in.
    """
    # Check which sample-level columns exist
    possible_aac = find_column(df, ['Average_autosomal_coverage'])
    possible_freemix = find_column(df, ['VerifyBamID2_Contamination', 'freemix', 'FREEMIX'])
    possible_avgdp = find_column(df, ['VerifyBamID2_Average_Depth', 'avg_dp', 'AVG_DP'])

    if not possible_aac and not possible_freemix and not possible_avgdp:
        logger.warning("No contamination-related columns found in merged data. Skipping contamination summary.")
        return pd.DataFrame()

    # Deduplicate by sample
    df_unique = df.drop_duplicates(subset=['Biosample_id']).copy()
    results = []
    for _, row in df_unique.iterrows():
        sample = row['Biosample_id']
        aac = row[possible_aac] if possible_aac else np.nan
        freemix = row[possible_freemix] if possible_freemix else np.nan
        avg_dp = row[possible_avgdp] if possible_avgdp else np.nan

        outlier_aac = aac < thresholds['autosomal_coverage_cutoff'] if not np.isnan(aac) else False
        outlier_freemix = freemix > thresholds['freemix_cutoff'] if not np.isnan(freemix) else False
        outlier_avgdp = avg_dp < thresholds['autosomal_coverage_cutoff'] if not np.isnan(avg_dp) else False

        results.append({
            'Sample': sample,
            'Average_autosomal_coverage': aac,
            'freemix': freemix,
            'avg_dp': avg_dp,
            'outlier_AAC': sample if outlier_aac else None,
            'outlier_freemix': sample if outlier_freemix else None,
            'outlier_avgdp': sample if outlier_avgdp else None,
            'Status_AAC': 'Failed' if outlier_aac else 'Pass',
            'Status_freemix': 'Failed' if outlier_freemix else 'Pass',
            'Status_avgdp': 'Failed' if outlier_avgdp else 'Pass'
        })
    contam_df = pd.DataFrame(results)
    if not contam_df.empty:
        # Convert outlier columns to lists (for consistency)
        contam_df['outlier_AAC'] = contam_df['outlier_AAC'].apply(lambda x: [x] if x else [])
        contam_df['outlier_freemix'] = contam_df['outlier_freemix'].apply(lambda x: [x] if x else [])
        contam_df['outlier_avgdp'] = contam_df['outlier_avgdp'].apply(lambda x: [x] if x else [])
    return contam_df


# ------------------------------------------------------------------------------
# MultiQC data processing functions (updated to handle multiple rows)
# ------------------------------------------------------------------------------

def find_multiqc_files(multiqc_dir, suffix):
    """Recursively find all files with given suffix in multiqc_dir."""
    matches = []
    for root, dirs, files in os.walk(multiqc_dir):
        for f in files:
            if f.endswith(suffix):
                matches.append(os.path.join(root, f))
    return matches


def safe_read_multiqc_file(filepath, expected_cols=None):
    """
    Safely read a MultiQC TSV file, returning a DataFrame or None if error.
    expected_cols: optional number of expected columns for validation.
    """
    try:
        df = pd.read_csv(filepath, sep='\t', header=0)
        if len(df) < 1:
            logger.warning(f"File {filepath} has no data rows, skipping.")
            return None
        if expected_cols and df.shape[1] != expected_cols:
            logger.warning(f"File {filepath} has {df.shape[1]} columns, expected {expected_cols}, skipping.")
            return None
        return df
    except Exception as e:
        logger.warning(f"Could not read {filepath}: {e}")
        return None


def parse_multiqc_value(val):
    """
    Convert a MultiQC value to float, handling strings like "(0, 100.0)".
    Returns float or np.nan if parsing fails.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Try direct conversion
    try:
        return float(s)
    except ValueError:
        pass
    # Try to extract the second number from a tuple string like "(0, 100.0)"
    match = re.search(r'\([^,]+,\s*([^)]+)\)', s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    # If all fails, return NaN
    logger.warning(f"Could not parse value: {s}")
    return np.nan


def process_cumulative_coverage(multiqc_dir, thresholds, html_path):
    """Process mosdepth cumulative coverage files (mosdepth-cumcoverage-dist-id.txt)."""
    files = find_multiqc_files(multiqc_dir, 'mosdepth-cumcoverage-dist-id.txt')
    if not files:
        logger.warning("No cumulative coverage files found (mosdepth-cumcoverage-dist-id.txt).")
        return None, None

    fig = go.Figure()
    report_data = []

    for f in files:
        df = safe_read_multiqc_file(f)
        if df is None:
            continue
        # Iterate over each sample row (skip header)
        for idx, row in df.iterrows():
            sample_name = str(row['Sample'])
            # Clean sample name (remove .recal if present)
            sample_name = sample_name.split('.')[0]
            # Extract values from columns after 'Sample'
            values = [parse_multiqc_value(row[col]) for col in df.columns[1:]]
            clean_values = [v for v in values if not np.isnan(v)]
            if not clean_values:
                logger.warning(f"No valid numeric values for {sample_name} in {f}, skipping.")
                continue
            # Create plot data (coverage from 1 to len(clean_values))
            plot_df = pd.DataFrame({
                'Coverage': np.arange(1, len(clean_values)+1),
                'Percent': clean_values,
                'Sample': sample_name
            })
            fig.add_trace(go.Scatter(
                x=plot_df['Coverage'], y=plot_df['Percent'],
                mode='markers', marker=dict(size=3), name=sample_name
            ))

            # Extract coverage at specific thresholds
            cov_dict = {'Biosample_id': sample_name}
            for level in [5,10,15,20,25,30]:
                # Find row where Coverage == level
                row_match = plot_df[plot_df['Coverage'] == level]
                val = row_match['Percent'].values[0] if not row_match.empty else np.nan
                cov_dict[f'Coverage_at_{level}X'] = val
            report_data.append(cov_dict)

    if not report_data:
        logger.warning("No valid cumulative coverage data found.")
        return None, None

    # Add horizontal line at 90%
    fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90% threshold")

    fig.update_layout(
        title='Cumulative Coverage Distribution',
        xaxis_title='Coverage (X)',
        yaxis_title='Percentage of bases covered',
        plot_bgcolor='white',
        hovermode='closest'
    )
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Create report DataFrame
    report_df = pd.DataFrame(report_data)
    for col in report_df.columns[1:]:
        report_df[f'Pass_at_{col}'] = report_df[col] >= thresholds['percent_coverage_cutoff']
        report_df[f'Pass_at_{col}'] = report_df[f'Pass_at_{col}'].map({True: 'Pass', False: 'Failed'})
    return report_df, fig


def process_coverage_distribution(multiqc_dir, html_path):
    """Process mosdepth coverage distribution files (mosdepth-coverage-dist-id.txt)."""
    files = find_multiqc_files(multiqc_dir, 'mosdepth-coverage-dist-id.txt')
    if not files:
        logger.warning("No coverage distribution files found (mosdepth-coverage-dist-id.txt).")
        return

    fig = go.Figure()
    for f in files:
        df = safe_read_multiqc_file(f)
        if df is None:
            continue
        for idx, row in df.iterrows():
            sample_name = str(row['Sample']).split('.')[0]
            values = [parse_multiqc_value(row[col]) for col in df.columns[1:]]
            clean_values = [v for v in values if not np.isnan(v)]
            if not clean_values:
                logger.warning(f"No valid numeric values for {sample_name} in {f}, skipping.")
                continue
            plot_df = pd.DataFrame({
                'Coverage': np.arange(1, len(clean_values)+1),
                'Percent': clean_values,
                'Sample': sample_name
            })
            fig.add_trace(go.Scatter(
                x=plot_df['Coverage'], y=plot_df['Percent'],
                mode='lines', name=sample_name, line=dict(shape='spline', smoothing=1.3)
            ))

    if len(fig.data) == 0:
        logger.warning("No valid coverage distribution data found.")
        return

    fig.update_layout(
        title='Coverage Distribution',
        xaxis_title='Coverage (X)',
        yaxis_title='Percentage of bases covered',
        plot_bgcolor='white'
    )
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def process_per_contig_coverage(multiqc_dir, thresholds, html_path):
    """Process per‑contig coverage files (mosdepth-coverage-per-contig-multi.txt)."""
    files = find_multiqc_files(multiqc_dir, 'mosdepth-coverage-per-contig-multi.txt')
    if not files:
        logger.warning("No per‑contig coverage files found (mosdepth-coverage-per-contig-multi.txt).")
        return None, None

    # Combine all data into one DataFrame for box plot
    all_data = []
    for f in files:
        df = safe_read_multiqc_file(f)
        if df is None:
            continue
        # Melt the DataFrame to long format: each row is a sample-contig combination
        try:
            # The file should have 'Sample' column and then contig columns
            id_vars = ['Sample']
            # All other columns are contigs
            value_vars = [col for col in df.columns if col != 'Sample']
            melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                             var_name='Contig', value_name='Coverage')
            # Convert coverage to numeric, coercing errors
            melted['Coverage'] = pd.to_numeric(melted['Coverage'], errors='coerce')
            melted.dropna(subset=['Coverage'], inplace=True)
            for _, row in melted.iterrows():
                sample_name = str(row['Sample']).split('.')[0]
                all_data.append({'BioSample_id': sample_name,
                                 'Contig': row['Contig'],
                                 'Coverage': row['Coverage']})
        except Exception as e:
            logger.warning(f"Could not parse {f}: {e}")
            continue

    if not all_data:
        logger.warning("No valid per‑contig coverage data found.")
        return None, None

    box_df = pd.DataFrame(all_data)

    # Box plot by contig
    fig = px.box(box_df, x='Contig', y='Coverage', color='Contig',
                 title='Average Contig Coverage per Chromosome')
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Generate report: per sample, exclude chrX/chrY, compute CV
    report = []
    for sample in box_df['BioSample_id'].unique():
        sub = box_df[(box_df['BioSample_id'] == sample) &
                     (~box_df['Contig'].isin(['chrX', 'chrY']))]
        if len(sub) == 0:
            continue
        mean_cov = sub['Coverage'].mean()
        std_cov = sub['Coverage'].std()
        cv = (std_cov / mean_cov) * 100 if mean_cov else np.nan
        status_dict = {}
        for cutoff in thresholds['cv_cutoffs']:
            status_dict[f'cv≤{cutoff}%'] = 'Pass' if cv <= cutoff else 'Failed'
        report.append({
            'Biosample_id': sample,
            'mean_coverage': mean_cov,
            'std_coverage': std_cov,
            'cv_percent': cv,
            **status_dict
        })
    report_df = pd.DataFrame(report)
    return report_df, fig


def process_mapping_stats(multiqc_dir, thresholds, html_path):
    """Process samtools alignment stats files (samtools_alignment_plot.txt)."""
    files = find_multiqc_files(multiqc_dir, 'samtools_alignment_plot.txt')
    if not files:
        logger.warning("No mapping stats files found (samtools_alignment_plot.txt).")
        return None

    records = []
    for f in files:
        df = safe_read_multiqc_file(f, expected_cols=4)  # Expect Sample, mapped, mq0, unmapped
        if df is None:
            continue
        for idx, row in df.iterrows():
            sample_name = str(row['Sample']).split('.')[0]
            try:
                mapped = parse_multiqc_value(row.iloc[1])
                mq0 = parse_multiqc_value(row.iloc[2])
                unmapped = parse_multiqc_value(row.iloc[3])
                if np.isnan(mapped) or np.isnan(mq0) or np.isnan(unmapped):
                    logger.warning(f"Missing values for {sample_name} in {f}, skipping.")
                    continue
                total = mapped + mq0 + unmapped
                records.append({
                    'Biosample_id': sample_name,
                    'mapped_percent': (mapped / total) * 100,
                    'mq0_percent': (mq0 / total) * 100,
                    'unmapped_percent': (unmapped / total) * 100
                })
            except Exception as e:
                logger.warning(f"Error parsing mapping stats from {f}: {e}")
                continue

    if not records:
        logger.warning("No valid mapping stats found.")
        return None

    df_stats = pd.DataFrame(records)

    # Stacked bar chart
    fig = go.Figure()
    for status in ['mapped_percent', 'mq0_percent', 'unmapped_percent']:
        fig.add_trace(go.Bar(
            x=df_stats['Biosample_id'],
            y=df_stats[status],
            name=status.replace('_percent', '')
        ))
    fig.update_layout(barmode='stack', title='Mapping Statistics',
                      xaxis_title='Sample', yaxis_title='Percentage',
                      plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Add pass/fail based on mapped percent
    df_stats['Status'] = df_stats['mapped_percent'].apply(
        lambda x: 'Pass' if x >= thresholds['mapped_percent_cutoff'] else 'Failed'
    )
    return df_stats


def process_insert_stats(multiqc_dir, html_path):
    """
    Process insert size stats – originally used samtools_stats.txt, but that file is not present.
    Instead, we look for fastp-insert-size-plot.txt and parse it if available.
    If not found, skip and log a warning.
    """
    files = find_multiqc_files(multiqc_dir, 'fastp-insert-size-plot.txt')
    if not files:
        logger.warning("No insert size files found (fastp-insert-size-plot.txt). Skipping insert stats.")
        return None

    # We'll attempt to parse fastp insert size plot (format may vary). For now, skip.
    logger.info("fastp-insert-size-plot.txt found but parsing not implemented. Skipping insert stats.")
    return None


def process_base_quality(multiqc_dir, thresholds, html_path):
    """Process per‑base sequence quality files (fastqc_per_base_sequence_quality_plot.txt)."""
    files = find_multiqc_files(multiqc_dir, 'fastqc_per_base_sequence_quality_plot.txt')
    if not files:
        logger.warning("No base quality files found (fastqc_per_base_sequence_quality_plot.txt).")
        return None

    records = []
    for f in files:
        df = safe_read_multiqc_file(f)
        if df is None:
            continue
        for idx, row in df.iterrows():
            sample_name = str(row['Sample']).split('.')[0]
            values = [parse_multiqc_value(row[col]) for col in df.columns[1:]]
            clean_values = [v for v in values if not np.isnan(v)]
            if not clean_values:
                logger.warning(f"No valid numeric values for {sample_name} in {f}, skipping.")
                continue
            records.append({
                'Biosample_id': sample_name,
                'min_quality': np.min(clean_values),
                'max_quality': np.max(clean_values),
                'mean_quality': np.mean(clean_values),
                'median_quality': np.median(clean_values)
            })

    if not records:
        logger.warning("No valid base quality data found.")
        return None

    df_qual = pd.DataFrame(records)

    # Bar plots
    for col in df_qual.columns[1:]:
        fig = px.bar(df_qual, x='Biosample_id', y=col, title=col)
        fig.add_hline(y=30, line_dash="dash", line_color="red")
        fig.update_layout(plot_bgcolor='white')
        with open(html_path, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

    # Status based on min quality >= 30
    df_qual['Status'] = df_qual['min_quality'].apply(
        lambda x: 'Pass' if x >= thresholds['base_quality_cutoff'] else 'Failed'
    )
    return df_qual


# ------------------------------------------------------------------------------
# Plotting functions for the main QC metrics (using the combined DataFrame)
# ------------------------------------------------------------------------------

def plot_coverage_by_chromosome(df_display, html_path):
    """Line plots: Percent coverage at 5X, 15X, 30X per chromosome across all samples."""
    # Find column names
    pct5_col = find_column(df_display, ['Percent_autosome_coverage_at_5X'])
    pct15_col = find_column(df_display, ['Percent_autosome_coverage_at_15X'])
    pct30_col = find_column(df_display, ['Percent_autosome_coverage_at_30X'])

    fig = go.Figure()
    for cov, col in [(5, pct5_col), (15, pct15_col), (30, pct30_col)]:
        # One line per sample
        for sample in df_display['Biosample_id'].unique():
            sub = df_display[df_display['Biosample_id'] == sample]
            fig.add_trace(go.Scatter(
                x=sub['Chrom'], y=sub[col],
                mode='lines+markers', name=f"{sample} {cov}X", legendgroup=sample,
                showlegend=(cov == 5)   # show only once per sample to avoid clutter
            ))
    fig.add_hrect(y0=90, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(
        title='Autosome Coverage per Chromosome',
        xaxis_title='Chromosome',
        yaxis_title='Percentage',
        plot_bgcolor='white'
    )
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def plot_boxplots(df_display, html_path):
    """Box plots for 15X and 30X coverage per sample and per chromosome."""
    pct15_col = find_column(df_display, ['Percent_autosome_coverage_at_15X'])
    pct30_col = find_column(df_display, ['Percent_autosome_coverage_at_30X'])

    # Sample box at 15X
    fig = px.box(df_display, x='Biosample_id', y=pct15_col,
                 color='Biosample_id', title='Sample vs 15X Coverage')
    fig.add_hrect(y0=90, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Sample box at 30X
    fig = px.box(df_display, x='Biosample_id', y=pct30_col,
                 color='Biosample_id', title='Sample vs 30X Coverage')
    fig.add_hrect(y0=74, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Chromosome box at 15X
    fig = px.box(df_display, x='Chrom', y=pct15_col,
                 color='Chrom', title='Chromosome vs 15X Coverage')
    fig.add_hrect(y0=90, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Chromosome box at 30X
    fig = px.box(df_display, x='Chrom', y=pct30_col,
                 color='Chrom', title='Chromosome vs 30X Coverage')
    fig.add_hrect(y0=74, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def plot_other_metrics(df_display, html_path):
    """Line plots for Bases_covered_15x, Mean/Median ratio, Median/Mean coverage."""
    # Bases covered at 15X
    bases15_col = find_column(df_display, ['Bases_covered_15x', 'Bases_covered_15X'])
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub[bases15_col],
                                  mode='lines+markers', name=sample))
    fig.update_layout(title='Bases Covered at 15X per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Bases', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Mean/Median ratio
    ratio_col = find_column(df_display, ['Mean_by_Median_autosomal_coverage_ratio_over_region'])
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub[ratio_col],
                                  mode='lines+markers', name=sample))
    fig.add_hrect(y0=0.95, y1=1.2, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.add_hrect(y0=1.2, y1=1.25, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.update_layout(title='Mean/Median Coverage Ratio',
                      xaxis_title='Chromosome', yaxis_title='Ratio', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Median coverage
    median_col = find_column(df_display, ['Median_alignment_coverage_over_autosomal_loci'])
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub[median_col],
                                  mode='lines+markers', name=sample))
    fig.add_hrect(y0=32, y1=40, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(title='Median Coverage per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Coverage (X)', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Mean coverage
    mean_col = find_column(df_display, ['Mean_alignment_coverage_over_autosomal_loci'])
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub[mean_col],
                                  mode='lines+markers', name=sample))
    fig.add_hrect(y0=32, y1=40, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=40, y1=67, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(title='Mean Coverage per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Coverage (X)', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Percent coverage at various thresholds (5X,10X,15X,20X,25X,30X) – one plot per threshold
    for cov in [5,10,15,20,25,30]:
        col = find_column(df_display, [f'Percent_autosome_coverage_at_{cov}X'])
        fig = go.Figure()
        for sample in df_display['Biosample_id'].unique():
            sub = df_display[df_display['Biosample_id'] == sample]
            fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub[col],
                                      mode='lines+markers', name=sample))
        # Adjust threshold color based on cov
        if cov <= 15:
            fig.add_hrect(y0=90, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
            fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
        else:
            fig.add_hrect(y0=85, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
            fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
        fig.update_layout(title=f'Percent Coverage at {cov}X',
                          xaxis_title='Chromosome', yaxis_title='Percentage', plot_bgcolor='white')
        with open(html_path, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def plot_contamination(contam_df, html_path):
    """Line plots for AAC, freemix, avg_dp (with scaling)."""
    if contam_df.empty:
        logger.warning("Contamination DataFrame empty, skipping contamination plots.")
        return
    # AAC vs avg_dp vs freemix*1e5
    fig = go.Figure()
    for sample in contam_df['Sample']:
        row = contam_df[contam_df['Sample'] == sample].iloc[0]
        fig.add_trace(go.Scatter(x=[sample], y=[row['Average_autosomal_coverage']],
                                  mode='markers', name=f"{sample} AAC", marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=[sample], y=[row['avg_dp']],
                                  mode='markers', name=f"{sample} avg_dp", marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=[sample], y=[row['freemix']*1e5],
                                  mode='markers', name=f"{sample} freemix*1e5", marker=dict(size=8)))
    fig.update_layout(title='AAC, avg_dp, freemix (scaled)',
                      xaxis_title='Sample', yaxis_title='Value', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # AAC vs avg_dp only
    fig = go.Figure()
    for sample in contam_df['Sample']:
        row = contam_df[contam_df['Sample'] == sample].iloc[0]
        fig.add_trace(go.Scatter(x=[sample], y=[row['Average_autosomal_coverage']],
                                  mode='markers', name=f"{sample} AAC", marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=[sample], y=[row['avg_dp']],
                                  mode='markers', name=f"{sample} avg_dp", marker=dict(size=8)))
    fig.update_layout(title='AAC vs avg_dp', xaxis_title='Sample', yaxis_title='Value',
                      plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def plot_total_bases_aligned(df_display, html_path):
    """Line plot of total bases aligned per chromosome."""
    # Find the column for total bases aligned
    bases_col = find_column(df_display, ['Total_number_of_bases_aligned', 'Total_number_of_bases_aligned:'])
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub[bases_col],
                                  mode='lines+markers', name=sample))
    fig.update_layout(title='Total Bases Aligned per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Bases', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()
    thresholds = load_config(args.config)
    html_path = setup_output_dirs(args.outdir)

    # Load combined QC data (chromosome-level + sample-level)
    logger.info("Loading QC data...")
    df, df_display = load_qc_data(args.qc_dir, args.sample_summary)
    logger.info(f"Loaded {len(df)} rows for {df['Biosample_id'].nunique()} samples.")

    # --------------------------------------------------------------------------
    # Generate summaries and reports (TSV)
    # --------------------------------------------------------------------------
    logger.info("Generating sample-based summary...")
    sample_summary = sample_based_summary(df, thresholds)
    sample_summary.to_csv(os.path.join(args.outdir, 'Autosomal_Coverage_Samples_report.tsv'), sep='\t', index=False)

    logger.info("Generating chromosome-based summary...")
    chrom_summary = chromosome_based_summary(df, thresholds)
    chrom_summary.to_csv(os.path.join(args.outdir, 'Autosomal_Coverage_Chromosomes_report.tsv'), sep='\t', index=False)

    logger.info("Generating contamination summary...")
    contam_summary = contamination_summary(df, thresholds)
    if not contam_summary.empty:
        contam_summary.to_csv(os.path.join(args.outdir, 'Samples_Contamination_report.tsv'), sep='\t', index=False)
    else:
        logger.warning("Contamination summary not generated (missing columns).")

    # --------------------------------------------------------------------------
    # Process MultiQC data
    # --------------------------------------------------------------------------
    logger.info("Processing MultiQC cumulative coverage...")
    cumcov_report, _ = process_cumulative_coverage(args.multiqc_dir, thresholds, html_path)
    if cumcov_report is not None:
        cumcov_report.to_csv(os.path.join(args.outdir, 'Cumulative_Coverage_Samples_report.tsv'), sep='\t', index=False)

    logger.info("Processing MultiQC coverage distribution...")
    process_coverage_distribution(args.multiqc_dir, html_path)

    logger.info("Processing per-contig coverage...")
    contig_report, _ = process_per_contig_coverage(args.multiqc_dir, thresholds, html_path)
    if contig_report is not None:
        contig_report.to_csv(os.path.join(args.outdir, 'Coverage_per_contig_Samples_Chromosome_report.tsv'), sep='\t', index=False)
        # Also save chromosome imbalance report (same as above, but we already have CV)
        contig_report.to_csv(os.path.join(args.outdir, 'Chromosome_coverage_imbalance_report.tsv'), sep='\t', index=False)

    logger.info("Processing mapping statistics...")
    map_stats = process_mapping_stats(args.multiqc_dir, thresholds, html_path)
    if map_stats is not None:
        map_stats.to_csv(os.path.join(args.outdir, 'Percent_mapped_reads_report.tsv'), sep='\t', index=False)

    logger.info("Processing insert statistics...")
    insert_stats = process_insert_stats(args.multiqc_dir, html_path)
    if insert_stats is not None:
        insert_stats.to_csv(os.path.join(args.outdir, 'insert_quality_report.tsv'), sep='\t', index=False)

    logger.info("Processing base quality...")
    base_qual = process_base_quality(args.multiqc_dir, thresholds, html_path)
    if base_qual is not None:
        base_qual.to_csv(os.path.join(args.outdir, 'base_quality_report.tsv'), sep='\t', index=False)

    # --------------------------------------------------------------------------
    # Generate plots from main QC data
    # --------------------------------------------------------------------------
    logger.info("Generating main QC plots...")
    plot_coverage_by_chromosome(df_display, html_path)
    plot_boxplots(df_display, html_path)
    plot_other_metrics(df_display, html_path)
    plot_total_bases_aligned(df_display, html_path)
    if not contam_summary.empty:
        plot_contamination(contam_summary, html_path)

    logger.info(f"All reports and plots saved to {args.outdir}")
    logger.info(f"Combined HTML report: {html_path}")


if __name__ == '__main__':
    main()