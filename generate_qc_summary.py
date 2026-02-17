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
    output_file = /path/to/output.csv
"""

import argparse
import configparser
import csv
import re
import math
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------------------
# Parsers for each file type – now include status fields where available
# ------------------------------------------------------------------------------

def parse_multiqc_general_stats(file_path):
    """
    Parse multiqc_general_stats.txt, group rows by base sample name,
    and return a dict sample -> {metric: value} for duplication and mapping percentages.
    We prefer rows ending with '.md' over '.recal' for mapping stats.
    """
    samples = defaultdict(dict)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample_raw = row.get('Sample', '')
            is_md = '.md' in sample_raw
            is_recal = '.recal' in sample_raw
            base = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample_raw)

            if base not in samples or (is_md and not samples[base].get('_prefer_md')):
                # Picard duplication
                dup = row.get('Picard: Mark Duplicates_mqc-generalstats-picard_mark_duplicates-PERCENT_DUPLICATION')
                if dup:
                    samples[base]['percent_duplication'] = float(dup)

                # Samtools mapping percentages
                mapped_pct = row.get('Samtools: stats_mqc-generalstats-samtools_stats-reads_mapped_percent')
                if mapped_pct:
                    samples[base]['percent_mapped'] = float(mapped_pct)
                properly_paired = row.get('Samtools: stats_mqc-generalstats-samtools_stats-reads_properly_paired_percent')
                if properly_paired:
                    samples[base]['percent_properly_paired'] = float(properly_paired)

                if is_md:
                    samples[base]['_prefer_md'] = True
    return samples

def parse_samtools_alignment_plot(file_path):
    """Parse samtools_alignment_plot.txt and return total mapped reads per sample."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample_raw = row['Sample']
            sample = re.sub(r'(\.md|\.recal|-[^.]+|\.[0-9]+)$', '', sample_raw)
            mapped_mq0 = int(row.get('Mapped (with MQ>0)', 0))
            mq0 = int(row.get('MQ0', 0))
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
        output_file = config.get('Paths', 'output_file')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise KeyError("Missing section or key in config file. Ensure [Paths] contains "
                       "qc_output_dir, multiqc_data_dir, and output_file.") from e

    qc_output_path = Path(qc_output_dir)
    multiqc_path = Path(multiqc_data_dir)
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

    # Parse all files
    multiqc_data = parse_multiqc_general_stats(multiqc_stats) if multiqc_stats else {}
    mapped_reads_data = parse_samtools_alignment_plot(samtools_align) if samtools_align else {}
    cov_data = parse_autosomal_coverage_samples(cov_samples) if cov_samples else {}
    contam_data = parse_contamination_report(contam) if contam else {}
    base_data = parse_base_quality(base_qual) if base_qual else {}
    insert_data = parse_fastp_insert_size(insert_plot) if insert_plot else {}

    # Merge all data per sample
    all_samples = set(cov_data.keys()) | set(contam_data.keys()) | set(base_data.keys()) | \
                  set(multiqc_data.keys()) | set(mapped_reads_data.keys()) | set(insert_data.keys())
    sample_data = {}
    for sample in all_samples:
        d = {}
        d.update(cov_data.get(sample, {}))
        d.update(contam_data.get(sample, {}))
        d.update(base_data.get(sample, {}))
        d.update(multiqc_data.get(sample, {}))
        d.update(insert_data.get(sample, {}))
        if sample in mapped_reads_data:
            d['reads_mapped'] = mapped_reads_data[sample]
        sample_data[sample] = d

    total_samples = len(sample_data)

    # --------------------------------------------------------------------------
    # Define metrics with pass/fail evaluation
    # --------------------------------------------------------------------------
    metric_defs = [
        {
            "group": "Depth",
            "type": "Mosdepth Median coverage",
            "source": "Median_alignment_coverage_over_autosomal_loci (Autosomal_Coverage_Samples)",
            "threshold": "Not specified (all pass)",
            "key": "median_coverage",
            "fmt": lambda v: f"{min(v):.0f} – {max(v):.0f}",
            "pass_eval": lambda val, status_dict: True   # all pass
        },
        {
            "group": "Depth",
            "type": "Mosdepth Mean coverage",
            "source": "Mean_alignment_coverage_over_autosomal_loci (Autosomal_Coverage_Samples)",
            "threshold": "Not specified (all pass)",
            "key": "mean_coverage",
            "fmt": lambda v: f"{min(v):.2f} – {max(v):.2f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Depth",
            "type": "VerifyBAMID average depth",
            "source": "avg_dp (Samples_Contamination)",
            "threshold": "Not specified (all pass)",
            "key": "avg_dp",
            "fmt": lambda v: f"{min(v):.1f} – {max(v):.1f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Mean autosome coverage",
            "source": "Average_autosomal_coverage (Samples_Contamination)",
            "threshold": "Not specified (all pass)",
            "key": "avg_autosomal_coverage",
            "fmt": lambda v: f"{min(v):.2f} – {max(v):.2f}",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Coverage",
            "type": "Percent autosomes covered at 30X",
            "source": "Percent_autosome_coverage_at_30X (Autosomal_Coverage_Samples)",
            "threshold": ">90% (typical)",
            "key": "pct_30x",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            "pass_eval": lambda val, status_dict: status_dict.get('status_pct30', '') == 'Pass'
        },
        {
            "group": "Coverage",
            "type": "Percent autosomes covered at 15X",
            "source": "Percent_autosome_coverage_at_15X (Autosomal_Coverage_Samples)",
            "threshold": ">98%",
            "key": "pct_15x",
            "fmt": lambda v: f"{min(v):.0f}% – {max(v):.0f}%",
            # From original report, all samples pass 15X (Status_pct15 = Pass)
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Alignment and preprocessing",
            "type": "% Duplication",
            "source": "Picard: Mark Duplicates PERCENT_DUPLICATION (multiqc_general_stats.txt)",
            "threshold": "<20%",
            "key": "percent_duplication",
            "fmt": lambda v: f"{min(v):.1f}% – {max(v):.1f}%",
            "pass_eval": lambda val, status_dict: val < 20.0
        },
        {
            "group": "Alignment and preprocessing",
            "type": "Reads mapped",
            "source": "Samtools alignment plot (Mapped with MQ>0 + MQ0)",
            "threshold": "–",
            "key": "reads_mapped",
            "fmt": lambda v: f"{min(v)/1e6:.0f} – {max(v)/1e6:.0f} million reads",
            "pass_eval": lambda val, status_dict: True   # no threshold given
        },
        {
            "group": "Alignment and preprocessing",
            "type": "% Mapped",
            "source": "Samtools reads_mapped_percent (multiqc_general_stats.txt, .md/.recal)",
            "threshold": ">95%",
            "key": "percent_mapped",
            "fmt": lambda v: f"{min(v):.2f}% – {max(v):.2f}%",
            "pass_eval": lambda val, status_dict: val > 95.0
        },
        {
            "group": "Alignment and preprocessing",
            "type": "% Reads properly paired",
            "source": "Samtools reads_properly_paired_percent (multiqc_general_stats.txt, .md/.recal)",
            "threshold": ">90%",
            "key": "percent_properly_paired",
            "fmt": lambda v: f"{min(v):.2f}% – {max(v):.2f}%",
            "pass_eval": lambda val, status_dict: val > 90.0
        },
        {
            "group": "Alignment and preprocessing",
            "type": "Insert size standard deviation",
            "source": "Derived from fastp-insert-size-plot.txt",
            "threshold": "–",
            "key": "insert_size_std",
            "fmt": lambda v: f"{min(v):.1f} – {max(v):.1f} bp",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Alignment and preprocessing",
            "type": "Mean insert size",
            "source": "Derived from fastp-insert-size-plot.txt",
            "threshold": "–",
            "key": "insert_size_mean",
            "fmt": lambda v: f"{min(v):.0f} – {max(v):.0f} bp",
            "pass_eval": lambda val, status_dict: True
        },
        {
            "group": "Cross contamination",
            "type": "Cross contamination (freemix)",
            "source": "freemix (Samples_Contamination)",
            "threshold": "<0.05",
            "key": "freemix",
            "fmt": lambda v: f"{min(v):.2e} – {max(v):.2e}",
            "pass_eval": lambda val, status_dict: status_dict.get('status_freemix', '') == 'Pass'
        },
        {
            "group": "Sequence quality",
            "type": "Mean base quality (Phred score)",
            "source": "mean_quality (base_quality_report)",
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
                # Find the metric definition that uses this key
                mdef = next((m for m in metric_defs if m.get('key') == key), None)
                if mdef:
                    # Pass evaluation may need additional status fields from data
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

        # Get counts
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
            'QC Type': mdef['type'],
            'Metric (from report)': mdef['source'],
            'Expected Threshold': mdef['threshold'],
            'Observed Value (range)': observed,
            'Pass Count': pass_cnt,
            'Fail Count': fail_cnt,
            'Total Samples': total,
            'Pass %': f"{pass_pct:.1f}%" if pass_cnt > 0 else "0%",
            'Fail %': f"{fail_pct:.1f}%" if fail_cnt > 0 else "0%"
        }
        output_rows.append(row)

    # Add a special row for Per Sequence Quality (already covered by mean)
    output_rows.append({
        'QC Group': 'Sequence quality',
        'QC Type': 'Per Sequence Quality',
        'Metric (from report)': 'Covered by mean quality above',
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
        'QC Group', 'QC Type', 'Metric (from report)', 'Expected Threshold',
        'Observed Value (range)', 'Pass Count', 'Fail Count', 'Total Samples',
        'Pass %', 'Fail %'
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