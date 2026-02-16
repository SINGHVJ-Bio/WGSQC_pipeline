#!/usr/bin/env python3
"""
QC Report Generator for WGS/WES Pipeline

This script reads the per‑sample per‑chromosome QC metrics (from the pipeline)
and the MultiQC output directory, then produces:
    - Summary tables (TSV) for samples, chromosomes, contamination, coverage, etc.
    - An interactive HTML report with all plots (using Plotly).

Usage:
    python generate_qc_report.py --qc-json <path> --multiqc-dir <path> --outdir <path> [--config <path>]

Arguments:
    --qc-json       Path to the merged QC JSON file (e.g., QC_metricses_data_all_samples_all_chrom.json)
    --multiqc-dir   Path to the MultiQC output directory (contains mosdepth, samtools, etc. subdirs)
    --outdir        Directory where all reports and plots will be saved
    --config        (Optional) Configuration file with thresholds (see example below)
"""

import os
import sys
import argparse
import logging
import configparser
import json
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
    parser.add_argument('--qc-json', required=True,
                        help='Path to the merged QC JSON file (QC_metricses_data_all_samples_all_chrom.json)')
    parser.add_argument('--multiqc-dir', required=True,
                        help='MultiQC output directory (contains mosdepth, samtools, etc.)')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for reports and plots')
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


def load_qc_data(qc_json):
    """Load the merged QC JSON and return a DataFrame with proper types."""
    df = pd.read_json(qc_json)
    # Remove 'chr' prefix from Chrom column and convert to int for sorting
    df['Chrom'] = df['Chrom'].str.replace('chr', '').astype(int)
    # Create a copy with 'chr' prefix for display
    df_display = df.copy()
    df_display['Chrom'] = 'chr' + df_display['Chrom'].astype(str)
    return df, df_display


def sample_based_summary(df, thresholds):
    """
    Compute per‑sample summary statistics and outlier status.
    Returns a DataFrame with one row per sample.
    """
    results = []
    for sample in df['Biosample_id'].unique():
        sub = df[df['Biosample_id'] == sample]
        mean_cov = mean(sub['Mean_alignment_coverage_over_autosomal_loci'])
        median_cov = round(median(sub['Median_alignment_coverage_over_autosomal_loci']))
        pct_15x = round(mean(sub['Percent_autosome_coverage_at_15X']))
        pct_30x = round(mean(sub['Percent_autosome_coverage_at_30X']))
        mean_median_ratio = mean_cov / median_cov if median_cov else 0

        # Outlier detection
        outlier_mean = sub[sub['Mean_alignment_coverage_over_autosomal_loci'] < thresholds['autosomal_coverage_cutoff']]['Biosample_id'].unique()
        outlier_median = sub[sub['Median_alignment_coverage_over_autosomal_loci'] < thresholds['autosomal_coverage_cutoff']]['Biosample_id'].unique()
        outlier_pct15 = sub[sub['Percent_autosome_coverage_at_15X'] < thresholds['percent_coverage_cutoff']]['Biosample_id'].unique()
        outlier_pct30 = sub[sub['Percent_autosome_coverage_at_30X'] < thresholds['percent_coverage_cutoff']]['Biosample_id'].unique()
        outlier_ratio = sub[sub['Mean_by_Median_autosomal_coverage_ratio_over_region'] > thresholds['mean_median_ratio_cutoff']]['Biosample_id'].unique()

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
    results = []
    for chrom in df['Chrom'].unique():
        sub = df[df['Chrom'] == chrom]
        mean_cov = mean(sub['Mean_alignment_coverage_over_autosomal_loci'])
        median_cov = round(median(sub['Median_alignment_coverage_over_autosomal_loci']))
        pct_15x = round(mean(sub['Percent_autosome_coverage_at_15X']))
        pct_30x = round(mean(sub['Percent_autosome_coverage_at_30X']))
        mean_median_ratio = mean_cov / median_cov if median_cov else 0

        # Outlier detection (chromosomes where any sample fails)
        outlier_mean = sub[sub['Mean_alignment_coverage_over_autosomal_loci'] < thresholds['autosomal_coverage_cutoff']]['Chrom'].unique()
        outlier_median = sub[sub['Median_alignment_coverage_over_autosomal_loci'] < thresholds['autosomal_coverage_cutoff']]['Chrom'].unique()
        outlier_pct15 = sub[sub['Percent_autosome_coverage_at_15X'] < thresholds['percent_coverage_cutoff']]['Chrom'].unique()
        outlier_pct30 = sub[sub['Percent_autosome_coverage_at_30X'] < thresholds['percent_coverage_cutoff']]['Chrom'].unique()
        outlier_ratio = sub[sub['Mean_by_Median_autosomal_coverage_ratio_over_region'] > thresholds['mean_median_ratio_cutoff']]['Chrom'].unique()

        results.append({
            'Chromosome': f'chr{chrom}',
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
    Assumes these columns exist in the merged DataFrame.
    """
    # Deduplicate by sample (there may be multiple rows per sample)
    df_unique = df.drop_duplicates(subset=['Biosample_id']).copy()
    results = []
    for _, row in df_unique.iterrows():
        sample = row['Biosample_id']
        aac = row['Average_autosomal_coverage']
        freemix = row['freemix'] if 'freemix' in row else np.nan
        avg_dp = row['avg_dp'] if 'avg_dp' in row else np.nan

        outlier_aac = aac < thresholds['autosomal_coverage_cutoff']
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
    # Convert outlier columns to lists (for consistency)
    contam_df['outlier_AAC'] = contam_df['outlier_AAC'].apply(lambda x: [x] if x else [])
    contam_df['outlier_freemix'] = contam_df['outlier_freemix'].apply(lambda x: [x] if x else [])
    contam_df['outlier_avgdp'] = contam_df['outlier_avgdp'].apply(lambda x: [x] if x else [])
    return contam_df


# ------------------------------------------------------------------------------
# MultiQC data processing functions
# ------------------------------------------------------------------------------

def find_multiqc_files(multiqc_dir, suffix):
    """Recursively find all files with given suffix in multiqc_dir."""
    matches = []
    for root, dirs, files in os.walk(multiqc_dir):
        for f in files:
            if f.endswith(suffix):
                matches.append(os.path.join(root, f))
    return matches


def process_cumulative_coverage(multiqc_dir, thresholds, html_path):
    """Process mosdepth cumulative coverage files (*mosdepth-cumcoverage-dist-id_1.txt)."""
    files = find_multiqc_files(multiqc_dir, 'mosdepth-cumcoverage-dist-id_1.txt')
    if not files:
        logger.warning("No cumulative coverage files found.")
        return None, None

    fig = go.Figure()
    extracted_data = []
    report_data = []

    for f in files:
        df = pd.read_csv(f, sep='\t')
        sample_name = df['Sample'].iloc[1].split('.')[0]   # e.g., "WHB13294.recal" -> "WHB13294"
        # The second row contains the actual data
        values = df.iloc[1, 1:].values.astype(float)
        plot_df = pd.DataFrame({
            'Coverage': np.arange(1, len(values)+1),
            'Percent': values,
            'Sample': sample_name
        })
        fig.add_trace(go.Scatter(
            x=plot_df['Coverage'], y=plot_df['Percent'],
            mode='markers', marker=dict(size=3), name=sample_name
        ))

        # Extract coverage at specific thresholds
        cov_dict = {'Biosample_id': sample_name}
        for level in [5,10,15,20,25,30]:
            val = plot_df.loc[plot_df['Coverage'] == level, 'Percent'].values
            cov_dict[f'Coverage_at_{level}X'] = val[0] if len(val) else np.nan
        report_data.append(cov_dict)

    # Add horizontal line at 90%
    fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="90% threshold")

    fig.update_layout(
        title='Cumulative Coverage Distribution',
        xaxis_title='Coverage (X)',
        yaxis_title='Percentage of bases covered',
        plot_bgcolor='white',
        hovermode='closest'
    )
    # Save to HTML
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Create report DataFrame
    report_df = pd.DataFrame(report_data)
    for col in report_df.columns[1:]:
        report_df[f'Pass_at_{col}'] = report_df[col] >= thresholds['percent_coverage_cutoff']
        report_df[f'Pass_at_{col}'] = report_df[f'Pass_at_{col}'].map({True: 'Pass', False: 'Failed'})
    return report_df, fig


def process_coverage_distribution(multiqc_dir, html_path):
    """Process mosdepth coverage distribution files (*mosdepth-coverage-dist-id_1.txt)."""
    files = find_multiqc_files(multiqc_dir, 'mosdepth-coverage-dist-id_1.txt')
    if not files:
        logger.warning("No coverage distribution files found.")
        return

    fig = go.Figure()
    for f in files:
        df = pd.read_csv(f, sep='\t')
        sample_name = df['Sample'].iloc[1].split('.')[0]
        values = df.iloc[1, 1:].values.astype(float)
        plot_df = pd.DataFrame({
            'Coverage': np.arange(1, len(values)+1),
            'Percent': values,
            'Sample': sample_name
        })
        fig.add_trace(go.Scatter(
            x=plot_df['Coverage'], y=plot_df['Percent'],
            mode='lines', name=sample_name, line=dict(shape='spline', smoothing=1.3)
        ))

    fig.update_layout(
        title='Coverage Distribution',
        xaxis_title='Coverage (X)',
        yaxis_title='Percentage of bases covered',
        plot_bgcolor='white'
    )
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def process_per_contig_coverage(multiqc_dir, thresholds, html_path):
    """Process per‑contig coverage files (*coverage-per-contig_1.txt)."""
    files = find_multiqc_files(multiqc_dir, 'coverage-per-contig_1.txt')
    if not files:
        logger.warning("No per‑contig coverage files found.")
        return None, None

    # Combine all data into one DataFrame for box plot
    all_data = []
    for f in files:
        df = pd.read_csv(f, sep='\t')
        sample_name = df['Sample'].iloc[1].split('.')[0]
        # Transpose: first column is contig names, rest are values
        contigs = df.iloc[1:, 0].values
        values = df.iloc[1:, 1].values.astype(float)
        for contig, val in zip(contigs, values):
            all_data.append({'BioSample_id': sample_name, 'Contig': contig, 'Coverage': val})

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
    """Process samtools alignment stats files (*samtools_alignment_plot_1.txt)."""
    files = find_multiqc_files(multiqc_dir, 'samtools_alignment_plot_1.txt')
    if not files:
        logger.warning("No mapping stats files found.")
        return None

    records = []
    for f in files:
        df = pd.read_csv(f, sep='\t')
        sample_name = df['Sample'].iloc[1].split('.')[0]
        # Extract values: row 1 (index 1) has mapped, MQ0, unmapped
        mapped = df.iloc[1, 1]
        mq0 = df.iloc[1, 2]
        unmapped = df.iloc[1, 3]
        total = mapped + mq0 + unmapped
        records.append({
            'Biosample_id': sample_name,
            'mapped_percent': (mapped / total) * 100,
            'mq0_percent': (mq0 / total) * 100,
            'unmapped_percent': (unmapped / total) * 100
        })

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
    """Process samtools stats files (*samtools_stats.txt) for insert metrics."""
    files = find_multiqc_files(multiqc_dir, 'samtools_stats.txt')
    if not files:
        logger.warning("No samtools stats files found.")
        return None

    records = []
    for f in files:
        df = pd.read_csv(f, sep='\t')
        sample_name = df['Sample'].iloc[1].split('.')[0]
        # Extract needed rows
        mapped_paired = df.loc[df['#category'] == 'reads_mapped_and_paired_percent', 'sample'].values[0]
        properly_paired = df.loc[df['#category'] == 'reads_properly_paired_percent', 'sample'].values[0]
        mapped_percent = df.loc[df['#category'] == 'reads_mapped_percent', 'sample'].values[0]
        insert_avg = df.loc[df['#category'] == 'insert_size_average', 'sample'].values[0]
        insert_std = df.loc[df['#category'] == 'insert_size_standard_deviation', 'sample'].values[0]
        records.append({
            'Biosample_id': sample_name,
            'reads_mapped_and_paired_percent': float(mapped_paired),
            'reads_properly_paired_percent': float(properly_paired),
            'reads_mapped_percent': float(mapped_percent),
            'insert_size_average': float(insert_avg),
            'insert_size_standard_deviation': float(insert_std)
        })

    df_insert = pd.DataFrame(records)

    # Create bar charts for each metric
    for col in df_insert.columns[1:]:
        fig = px.bar(df_insert, x='Biosample_id', y=col, title=col)
        fig.update_layout(plot_bgcolor='white')
        with open(html_path, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

    # Combined grouped bar chart
    fig = px.bar(df_insert, x='Biosample_id',
                 y=['reads_mapped_and_paired_percent', 'reads_properly_paired_percent',
                    'reads_mapped_percent', 'insert_size_average', 'insert_size_standard_deviation'],
                 title='Insert & Mapping Metrics', barmode='group')
    fig.update_layout(plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Add pass/fail based on first three columns >=95%
    def pass_fail(row):
        return 'Pass' if all(row[1:4] >= 95) else 'Failed'
    df_insert['Status'] = df_insert.apply(pass_fail, axis=1)
    return df_insert


def process_base_quality(multiqc_dir, thresholds, html_path):
    """Process per‑base sequence quality files (*per_base_sequence_quality_plot_1.txt)."""
    files = find_multiqc_files(multiqc_dir, 'per_base_sequence_quality_plot_1.txt')
    if not files:
        logger.warning("No base quality files found.")
        return None

    records = []
    for f in files:
        df = pd.read_csv(f, sep='\t')
        sample_name = df['Sample'].iloc[1].split('.')[0]
        values = df.iloc[1, 1:].values.astype(float)
        records.append({
            'Biosample_id': sample_name,
            'min_quality': np.min(values),
            'max_quality': np.max(values),
            'mean_quality': np.mean(values),
            'median_quality': np.median(values)
        })

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
# Plotting functions for the main QC metrics (from the JSON)
# ------------------------------------------------------------------------------

def plot_coverage_by_chromosome(df_display, html_path):
    """Line plots: Percent coverage at 5X, 15X, 30X per chromosome across all samples."""
    fig = go.Figure()
    for cov in [5, 15, 30]:
        col = f'Percent_autosome_coverage_at_{cov}X'
        if col in df_display.columns:
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
    # Sample box at 15X
    fig = px.box(df_display, x='Biosample_id', y='Percent_autosome_coverage_at_15X',
                 color='Biosample_id', title='Sample vs 15X Coverage')
    fig.add_hrect(y0=90, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Sample box at 30X
    fig = px.box(df_display, x='Biosample_id', y='Percent_autosome_coverage_at_30X',
                 color='Biosample_id', title='Sample vs 30X Coverage')
    fig.add_hrect(y0=74, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Chromosome box at 15X
    fig = px.box(df_display, x='Chrom', y='Percent_autosome_coverage_at_15X',
                 color='Chrom', title='Chromosome vs 15X Coverage')
    fig.add_hrect(y0=90, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Chromosome box at 30X
    fig = px.box(df_display, x='Chrom', y='Percent_autosome_coverage_at_30X',
                 color='Chrom', title='Chromosome vs 30X Coverage')
    fig.add_hrect(y0=74, y1=95, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=95, y1=101, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(plot_bgcolor='white', showlegend=False)
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def plot_other_metrics(df_display, html_path):
    """Line plots for Bases_covered_15x, Mean/Median ratio, Median/Mean coverage."""
    # Bases covered at 15X
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub['Bases_covered_15x'],
                                  mode='lines+markers', name=sample))
    fig.update_layout(title='Bases Covered at 15X per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Bases', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Mean/Median ratio
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub['Mean_by_Median_autosomal_coverage_ratio_over_region'],
                                  mode='lines+markers', name=sample))
    fig.add_hrect(y0=0.95, y1=1.2, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.add_hrect(y0=1.2, y1=1.25, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.update_layout(title='Mean/Median Coverage Ratio',
                      xaxis_title='Chromosome', yaxis_title='Ratio', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Median coverage
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub['Median_alignment_coverage_over_autosomal_loci'],
                                  mode='lines+markers', name=sample))
    fig.add_hrect(y0=32, y1=40, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(title='Median Coverage per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Coverage (X)', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Mean coverage
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub['Mean_alignment_coverage_over_autosomal_loci'],
                                  mode='lines+markers', name=sample))
    fig.add_hrect(y0=32, y1=40, fillcolor='rgba(255,0,0,0.2)', line_width=0)
    fig.add_hrect(y0=40, y1=67, fillcolor='rgba(0,255,0,0.2)', line_width=0)
    fig.update_layout(title='Mean Coverage per Chromosome',
                      xaxis_title='Chromosome', yaxis_title='Coverage (X)', plot_bgcolor='white')
    with open(html_path, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Percent coverage at various thresholds (5X,10X,15X,20X,25X,30X) – one plot per threshold
    for cov in [5,10,15,20,25,30]:
        col = f'Percent_autosome_coverage_at_{cov}X'
        if col not in df_display.columns:
            continue
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
    fig = go.Figure()
    for sample in df_display['Biosample_id'].unique():
        sub = df_display[df_display['Biosample_id'] == sample]
        fig.add_trace(go.Scatter(x=sub['Chrom'], y=sub['Total_number_of_bases_aligned:'],
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

    # Load main QC data
    logger.info("Loading QC JSON...")
    df, df_display = load_qc_data(args.qc_json)

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
    contam_summary.to_csv(os.path.join(args.outdir, 'Samples_Contamination_report.tsv'), sep='\t', index=False)

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
    plot_contamination(contam_summary, html_path)

    logger.info(f"All reports and plots saved to {args.outdir}")
    logger.info(f"Combined HTML report: {html_path}")


if __name__ == '__main__':
    main()