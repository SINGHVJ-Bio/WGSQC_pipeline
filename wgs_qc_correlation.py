#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WGS QC Correlation Analysis Script – Publication Quality
with chromosome‑level metrics, sex check, sample information integration,
and highlighting of low‑coverage samples (using configurable cutoff for 30X).

Additionally, if group_cuttoff_30X is provided in the config file,
a separate analysis is run that splits samples into High/Low groups
based on that threshold, tests all available QC metrics and metadata
for association with the group, and creates boxplots/bar charts.

The sex check scatter plot now labels the three samples with the lowest
and three with the highest Y‑chromosome coverage within each sex group:
- On the inferred subplot, extremes are based on inferred sex.
- On the known sex subplot (if available), extremes are based on known sex.
Male labels are dark blue, female labels dark red. Labels are adjusted
to avoid overlap using adjustText (if installed), with arrows and shrinkage.

Enhancements in the coverage group analysis (per suffix):
- Boxplots show Mann‑Whitney U, p‑value, and group sizes.
- Titles include the cutoff value.
- Bar charts display counts inside bars and chi‑square test results.
- Larger correlation heatmap for continuous variables.
- Mixed‑type association matrix (Pearson / Kruskal‑Wallis / Cramér’s V).
- Separate heatmap for categorical‑categorical associations (Cramér’s V).
- Diagnostic sample count checks to identify missing samples.

Enhancements in the main analysis (outside coverage groups):
- Boxplots for Sex (2 groups) now include Mann‑Whitney U and p‑value.
- Boxplots for Race (multiple groups) now include Kruskal‑Wallis H and p‑value.

Usage:
    python wgs_qc_correlation.py --config /path/to/summary_config.ini

The configuration file must contain a [Paths] section with the following keys:
    qc_output_dir      = directory with Autosomal_Coverage_Samples_report.tsv, etc.
    multiqc_data_dir   = directory with multiqc_general_stats.txt
    qc_metrics_dir     = directory containing sample subdirectories with chromosome files
    output_file        = path used to derive plot output directory (plots/ will be created alongside)
    sex_info           = (optional) path to TSV file with columns Library_ID, Age, Sex, Race, Raw_Data_Size, ...
    group_cuttoff_30X  = (optional) threshold for Percent_autosome_coverage_at_30X to define High/Low groups
                         (also used as the low‑coverage threshold for 30X in scatter plots)

Plots are saved in both .png (300 dpi) and .pdf formats.
"""

import argparse
import configparser
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, mannwhitneyu, chi2_contingency, kruskal
from scipy.stats.contingency import association

# Optional adjustText for label repelling
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("Note: adjustText not installed. Label overlapping may occur in sex plot.")

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def clean_sample_id(s):
    """
    Clean a sample ID by:
    1. Converting to string and stripping whitespace.
    2. Removing everything after the first underscore, hyphen, or dot.
    3. Taking the first 8 characters (assumes IDs are 8 chars long).
    """
    if pd.isna(s):
        return s
    s = str(s).strip()
    # Remove suffixes: anything after _, -, or .
    s = re.sub(r'[_.-].*$', '', s)
    # Take first 8 characters
    return s[:8]

def extract_sample_name(name):
    """Extract base sample name from various MultiQC suffixes (e.g., .md, .recal, _1-L001)."""
    return clean_sample_id(name)

def safe_pearsonr(x, y):
    """Compute Pearson correlation, return NaN if insufficient data."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]

def set_publication_style():
    """Apply seaborn and matplotlib settings for publication‑quality plots."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['pdf.fonttype'] = 42      # Ensure text is editable in PDF
    plt.rcParams['ps.fonttype'] = 42

def clean_column_names(df):
    """Strip whitespace and remove trailing colons from column names."""
    df.columns = df.columns.str.strip().str.rstrip(':')
    return df

def is_main_chromosome(chrom):
    """Return True if chromosome is a standard autosome or sex chromosome,
       handling both 'chr'-prefixed and plain names."""
    if pd.isna(chrom):
        return False
    chrom_str = str(chrom)
    if chrom_str.startswith('chr'):
        chrom_str = chrom_str[3:]
    if chrom_str.isdigit() and 1 <= int(chrom_str) <= 22:
        return True
    if chrom_str in ['X', 'Y']:
        return True
    return False

def natural_sort_chromosomes(chrom_list):
    """Sort chromosome names naturally, accepting both 'chr'‑prefixed and plain."""
    def chrom_key(chrom):
        c = str(chrom)
        if c.startswith('chr'):
            c = c[3:]
        if c.isdigit():
            return (0, int(c))
        elif c == 'X':
            return (1, 23)
        elif c == 'Y':
            return (2, 24)
        else:
            return (3, c)
    return sorted(chrom_list, key=chrom_key)

def infer_sex(chrX_cov, chrY_cov, auto_cov, male_threshold=0.7, y_min=2):
    """
    Infer sex from coverage:
    - Male: chrX/auto < male_threshold and chrY_cov > y_min
    - Female: otherwise (chrX/auto ~1, chrY near 0)
    """
    if chrX_cov is None or chrY_cov is None or auto_cov is None:
        return 'Unknown'
    ratio = chrX_cov / auto_cov
    if ratio < male_threshold and chrY_cov > y_min:
        return 'Male'
    else:
        return 'Female'

# -----------------------------------------------------------------------------
# Coverage group analysis (split by cutoff)
# -----------------------------------------------------------------------------
def run_coverage_group_analysis(config, results_base_dir, cutoff):
    """
    Perform separate analyses for .md and .recal samples:
    - Merge sample metadata, multiqc stats, and autosomal coverage.
    - Split into High/Low groups based on cutoff for Percent_autosome_coverage_at_30X.
    - Identify categorical/continuous variables.
    - Run Mann‑Whitney U (continuous) or chi‑square (categorical) tests.
    - Save statistical results CSV and generate boxplots/bar charts.
    - Also create a categorical‑only association matrix (Cramér's V).
    - Diagnostic prints to identify missing samples.
    """
    print("\n" + "="*60)
    print("Starting coverage group analysis (split by cutoff = {}%)".format(cutoff))
    print("="*60)

    # Retrieve paths from config
    qc_output_dir      = config['Paths']['qc_output_dir']
    multiqc_data_dir   = config['Paths']['multiqc_data_dir']
    sex_info_file      = config.get('Paths', 'sex_info', fallback=None)

    autosomal_cov_file = os.path.join(qc_output_dir, 'Autosomal_Coverage_Samples_report.tsv')
    multiqc_stats_file = os.path.join(multiqc_data_dir, 'multiqc_general_stats.txt')

    # Create output directory for this analysis
    analysis_out_dir = os.path.join(results_base_dir, 'coverage_analysis')
    os.makedirs(analysis_out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Read sex info (if available) and clean sample IDs
    # -------------------------------------------------------------------------
    if sex_info_file and os.path.exists(sex_info_file):
        sex_info = pd.read_csv(sex_info_file, sep='\t')
        # Rename first column to SampleID_base
        sex_info.rename(columns={sex_info.columns[0]: 'SampleID_base'}, inplace=True)
        # Clean IDs
        sex_info['SampleID_base'] = sex_info['SampleID_base'].apply(clean_sample_id)
        print(f"Loaded sex info for {len(sex_info)} samples.")
    else:
        sex_info = pd.DataFrame()
        if sex_info_file:
            print(f"Warning: sex_info file {sex_info_file} not found – proceeding without it.")

    # -------------------------------------------------------------------------
    # Read autosomal coverage report and clean sample IDs
    # -------------------------------------------------------------------------
    autosomal_cov = pd.read_csv(autosomal_cov_file, sep='\t')
    autosomal_cov.rename(columns={autosomal_cov.columns[0]: 'SampleID_base'}, inplace=True)
    autosomal_cov['SampleID_base'] = autosomal_cov['SampleID_base'].apply(clean_sample_id)
    cov_col = next((col for col in autosomal_cov.columns if 'Percent_autosome_coverage_at_30X' in col), None)
    if cov_col is None:
        raise KeyError("Could not find 'Percent_autosome_coverage_at_30X' in autosomal coverage file.")
    autosomal_cov = autosomal_cov[['SampleID_base', cov_col]].copy()
    autosomal_cov.rename(columns={cov_col: 'Percent_autosome_coverage_at_30X'}, inplace=True)
    print("Loaded autosomal coverage.")

    # -------------------------------------------------------------------------
    # Read and filter MultiQC general stats, clean sample IDs
    # -------------------------------------------------------------------------
    multiqc_raw = pd.read_csv(multiqc_stats_file, sep='\t', low_memory=False)
    multiqc_raw.rename(columns={multiqc_raw.columns[0]: 'Sample'}, inplace=True)
    # Clean the sample name to base ID
    multiqc_raw['base_id'] = multiqc_raw['Sample'].apply(clean_sample_id)
    multiqc_filtered = multiqc_raw[multiqc_raw['base_id'].notna()].copy()

    # Remove columns that are completely empty
    def is_mostly_empty(col):
        return col.isnull().all() or (col.astype(str).str.strip() == '').all()
    empty_cols = [col for col in multiqc_filtered.columns if is_mostly_empty(multiqc_filtered[col])]
    multiqc_filtered.drop(columns=empty_cols, inplace=True)

    print(f"MultiQC: kept {len(multiqc_filtered)} rows (.md or .recal samples).")
    print(f"Dropped {len(empty_cols)} empty columns.")

    # Separate into .md and .recal based on original Sample column
    md_df = multiqc_filtered[multiqc_filtered['Sample'].str.lower().str.endswith('.md')].copy()
    recal_df = multiqc_filtered[multiqc_filtered['Sample'].str.lower().str.endswith('.recal')].copy()
    md_df.drop(columns=['Sample'], inplace=True)
    recal_df.drop(columns=['Sample'], inplace=True)
    # Ensure unique base_id (take first if duplicates)
    md_df = md_df.groupby('base_id').first().reset_index()
    recal_df = recal_df.groupby('base_id').first().reset_index()

    print(f"Unique .md samples: {len(md_df)}")
    print(f"Unique .recal samples: {len(recal_df)}")

    # -------------------------------------------------------------------------
    # Diagnostic: Check for missing samples before merging
    # -------------------------------------------------------------------------
    if not sex_info.empty:
        sex_ids = set(sex_info['SampleID_base'])
        md_ids = set(md_df['base_id'])
        recal_ids = set(recal_df['base_id'])
        cov_ids = set(autosomal_cov['SampleID_base'])

        missing_in_md = sex_ids - md_ids
        missing_in_recal = sex_ids - recal_ids
        missing_in_cov = sex_ids - cov_ids

        if missing_in_md:
            print(f"WARNING: {len(missing_in_md)} samples from sex_info not found in .md MultiQC data:")
            print(sorted(missing_in_md))
        if missing_in_recal:
            print(f"WARNING: {len(missing_in_recal)} samples from sex_info not found in .recal MultiQC data:")
            print(sorted(missing_in_recal))
        if missing_in_cov:
            print(f"WARNING: {len(missing_in_cov)} samples from sex_info not found in autosomal coverage file:")
            print(sorted(missing_in_cov))

    # -------------------------------------------------------------------------
    # Define analysis function for one suffix
    # -------------------------------------------------------------------------
    def analyze_one(df_multiqc, suffix_name, out_parent, cutoff):
        out_dir = os.path.join(out_parent, f'analysis_{suffix_name}')
        os.makedirs(out_dir, exist_ok=True)

        # Merge with sex_info and autosomal coverage
        merged = df_multiqc.merge(sex_info, left_on='base_id', right_on='SampleID_base', how='inner')
        merged = merged.merge(autosomal_cov, on='SampleID_base', how='inner')

        if merged.empty:
            print(f"No merged data for {suffix_name}. Skipping.")
            return

        print(f"\n--- Analyzing {suffix_name.upper()} samples ---")
        print(f"Merged data shape: {merged.shape}")

        # Define groups
        merged['Coverage_group'] = np.where(merged['Percent_autosome_coverage_at_30X'] > cutoff,
                                            'High', 'Low')
        group_counts = merged['Coverage_group'].value_counts()
        n_high_total = group_counts.get('High', 0)
        n_low_total = group_counts.get('Low', 0)
        print(f"Group sizes: High = {n_high_total}, Low = {n_low_total}")

        # Identify categorical/continuous variables
        exclude_cols = ['base_id', 'SampleID_base', 'Percent_autosome_coverage_at_30X', 'Coverage_group']
        categorical_vars = []
        continuous_vars = []
        for col in merged.columns:
            if col in exclude_cols:
                continue
            if merged[col].dtype == 'object' or merged[col].dtype.name == 'category':
                categorical_vars.append(col)
            elif pd.api.types.is_numeric_dtype(merged[col]):
                if merged[col].nunique() < 10:
                    categorical_vars.append(col)
                else:
                    continuous_vars.append(col)
            else:
                categorical_vars.append(col)

        print(f"Continuous variables: {len(continuous_vars)}")
        print(f"Categorical variables: {len(categorical_vars)}")

        # Statistical tests results table
        results = []

        # Continuous: Mann-Whitney U
        for var in continuous_vars:
            group_high = merged[merged['Coverage_group'] == 'High'][var].dropna()
            group_low  = merged[merged['Coverage_group'] == 'Low'][var].dropna()
            n_high = len(group_high)
            n_low = len(group_low)
            if n_high < 2 or n_low < 2:
                p_val = np.nan
                stat = np.nan
            else:
                stat, p_val = mannwhitneyu(group_high, group_low, alternative='two-sided')
            med_high = group_high.median() if n_high > 0 else np.nan
            q1_high = group_high.quantile(0.25) if n_high > 0 else np.nan
            q3_high = group_high.quantile(0.75) if n_high > 0 else np.nan
            med_low = group_low.median() if n_low > 0 else np.nan
            q1_low = group_low.quantile(0.25) if n_low > 0 else np.nan
            q3_low = group_low.quantile(0.75) if n_low > 0 else np.nan
            results.append({
                'Variable': var,
                'Type': 'continuous',
                'Test': 'Mann-Whitney U',
                'Statistic': stat,
                'P-value': p_val,
                'High_median (IQR)': f"{med_high:.2f} ({q1_high:.2f}-{q3_high:.2f})" if not pd.isna(med_high) else '',
                'Low_median (IQR)': f"{med_low:.2f} ({q1_low:.2f}-{q3_low:.2f})" if not pd.isna(med_low) else '',
                'n_High': n_high,
                'n_Low': n_low
            })

        # Categorical: Chi-square
        for var in categorical_vars:
            cont_table = pd.crosstab(merged[var], merged['Coverage_group'])
            if cont_table.shape[0] < 2 or cont_table.shape[1] < 2:
                p_val = np.nan
                stat = np.nan
            else:
                chi2, p_val, dof, expected = chi2_contingency(cont_table)
                stat = chi2
            total_high = merged['Coverage_group'].value_counts().get('High', 0)
            total_low  = merged['Coverage_group'].value_counts().get('Low', 0)
            summary_parts = []
            for cat in cont_table.index:
                count_high = cont_table.loc[cat, 'High'] if 'High' in cont_table.columns else 0
                count_low  = cont_table.loc[cat, 'Low'] if 'Low' in cont_table.columns else 0
                pct_high = 100 * count_high / total_high if total_high > 0 else 0
                pct_low  = 100 * count_low / total_low if total_low > 0 else 0
                summary_parts.append(f"{cat}: High {count_high} ({pct_high:.1f}%), Low {count_low} ({pct_low:.1f}%)")
            results.append({
                'Variable': var,
                'Type': 'categorical',
                'Test': 'Chi-square',
                'Statistic': stat,
                'P-value': p_val,
                'High_median (IQR)': '',
                'Low_median (IQR)': '',
                'Group_summary': '; '.join(summary_parts),
                'n_High': total_high,
                'n_Low': total_low
            })

        # Save results
        results_df = pd.DataFrame(results).sort_values('P-value')
        results_csv = os.path.join(out_dir, 'statistical_tests.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"Statistical results saved to {results_csv}")

        # Generate plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (8, 6)

        # --- Boxplots for continuous variables with annotations ---
        for var in continuous_vars:
            data = merged[[var, 'Coverage_group']].dropna()
            if data.empty:
                continue
            # Get the row from results_df for this variable
            row = results_df[results_df['Variable'] == var].iloc[0]
            stat_val = row['Statistic']
            p_val = row['P-value']
            n_high = row['n_High']
            n_low = row['n_Low']
            # Create plot
            plt.figure()
            ax = sns.boxplot(x='Coverage_group', y=var, data=data, palette='Set2')
            # Annotate
            textstr = f"U = {stat_val:.2f}\np = {p_val:.2e}\nn_High = {n_high}\nn_Low = {n_low}"
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.title(f'Distribution of {var} by Coverage Group (cutoff = {cutoff}%)')
            plt.tight_layout()
            outfile = os.path.join(out_dir, f'boxplot_{var}.png')
            plt.savefig(outfile, dpi=150)
            plt.close()

        # --- Stacked bar charts for categorical variables with counts and test annotation ---
        for var in categorical_vars:
            ct = pd.crosstab(merged[var], merged['Coverage_group'])
            if ct.empty:
                continue
            # Normalize by column for percentages
            ct_norm = ct.div(ct.sum(axis=0), axis=1) * 100
            # Get test result
            row = results_df[results_df['Variable'] == var].iloc[0]
            stat_val = row['Statistic']
            p_val = row['P-value']
            # Create plot
            fig, ax = plt.subplots()
            bars = ct_norm.plot(kind='bar', stacked=True, color=['#FF9999', '#66B2FF'],
                                edgecolor='black', ax=ax, legend=False)
            # Add counts as text inside bars
            for i, (cat, row_data) in enumerate(ct.iterrows()):
                cum_height = 0
                for j, (group, count) in enumerate(row_data.items()):
                    if count > 0:
                        height = ct_norm.loc[cat, group]  # percentage
                        # Place text at center of segment
                        ax.text(i, cum_height + height/2, str(int(count)),
                                ha='center', va='center', fontsize=8, color='white', weight='bold')
                        cum_height += height
            # Add test annotation
            ax.text(0.98, 0.98, f"χ² = {stat_val:.2f}\np = {p_val:.2e}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_ylabel('Percentage (%)')
            ax.set_xlabel(var)
            ax.set_title(f'Proportion of Coverage Groups by {var} (cutoff = {cutoff}%)')
            ax.legend(title='Coverage Group', labels=['High', 'Low'])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            outfile = os.path.join(out_dir, f'barplot_{var}.png')
            plt.savefig(outfile, dpi=150)
            plt.close()

        # --- Larger correlation heatmap for continuous variables ---
        numeric_cols = continuous_vars + ['Percent_autosome_coverage_at_30X']
        numeric_cols = [c for c in numeric_cols if c in merged.columns]
        if len(numeric_cols) > 1:
            corr = merged[numeric_cols].corr()
            plt.figure(figsize=(16,14))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True,
                        linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title(f'Correlation Matrix of Continuous Variables ({suffix_name}, cutoff={cutoff}%)')
            plt.tight_layout()
            outfile = os.path.join(out_dir, 'correlation_heatmap_continuous.png')
            plt.savefig(outfile, dpi=150)
            plt.close()

        # --- Mixed-type association matrix (all variables) ---
        all_vars = continuous_vars + categorical_vars + ['Percent_autosome_coverage_at_30X']
        all_vars = [v for v in all_vars if v in merged.columns]
        n_vars = len(all_vars)
        assoc_matrix = pd.DataFrame(index=all_vars, columns=all_vars, dtype=float)

        for i, var1 in enumerate(all_vars):
            for j, var2 in enumerate(all_vars):
                if i == j:
                    assoc_matrix.loc[var1, var2] = 1.0  # self-correlation
                    continue
                # Extract non-missing pairs
                temp = merged[[var1, var2]].dropna()
                if len(temp) < 3:
                    assoc_matrix.loc[var1, var2] = np.nan
                    continue
                # Determine types
                type1 = 'continuous' if var1 in continuous_vars + ['Percent_autosome_coverage_at_30X'] else 'categorical'
                type2 = 'continuous' if var2 in continuous_vars + ['Percent_autosome_coverage_at_30X'] else 'categorical'

                if type1 == 'continuous' and type2 == 'continuous':
                    # Pearson correlation
                    r, _ = pearsonr(temp[var1], temp[var2])
                    assoc_matrix.loc[var1, var2] = r
                elif type1 == 'continuous' and type2 == 'categorical':
                    # Kruskal-Wallis H statistic (use as measure of association)
                    groups = temp.groupby(var2)[var1].apply(list)
                    if len(groups) < 2:
                        assoc_matrix.loc[var1, var2] = np.nan
                    else:
                        h, _ = kruskal(*groups)
                        assoc_matrix.loc[var1, var2] = h
                elif type1 == 'categorical' and type2 == 'continuous':
                    # Same as above, symmetric
                    groups = temp.groupby(var1)[var2].apply(list)
                    if len(groups) < 2:
                        assoc_matrix.loc[var1, var2] = np.nan
                    else:
                        h, _ = kruskal(*groups)
                        assoc_matrix.loc[var1, var2] = h
                else:  # both categorical
                    # Cramér's V
                    cont_table = pd.crosstab(temp[var1], temp[var2])
                    if cont_table.shape[0] < 2 or cont_table.shape[1] < 2:
                        assoc_matrix.loc[var1, var2] = np.nan
                    else:
                        cramer_v = association(cont_table, method='cramer')
                        assoc_matrix.loc[var1, var2] = cramer_v

        # Plot the association matrix
        plt.figure(figsize=(18,16))
        sns.heatmap(assoc_matrix, annot=True, fmt='.2f', cmap='viridis',
                    cbar_kws={'label': 'Association measure'}, square=True,
                    linewidths=0.5)
        plt.title(f'Mixed-type Association Matrix ({suffix_name}, cutoff={cutoff}%)\n'
                  '(Pearson r / Kruskal-Wallis H / Cramér\'s V)')
        plt.tight_layout()
        outfile = os.path.join(out_dir, 'association_matrix_all.png')
        plt.savefig(outfile, dpi=150)
        plt.close()

        # --- NEW: Categorical-only association matrix (Cramér's V) ---
        if len(categorical_vars) >= 2:
            cat_matrix = pd.DataFrame(index=categorical_vars, columns=categorical_vars, dtype=float)
            for i, var1 in enumerate(categorical_vars):
                for j, var2 in enumerate(categorical_vars):
                    if i == j:
                        cat_matrix.loc[var1, var2] = 1.0
                        continue
                    temp = merged[[var1, var2]].dropna()
                    if len(temp) < 3:
                        cat_matrix.loc[var1, var2] = np.nan
                        continue
                    cont_table = pd.crosstab(temp[var1], temp[var2])
                    if cont_table.shape[0] < 2 or cont_table.shape[1] < 2:
                        cat_matrix.loc[var1, var2] = np.nan
                    else:
                        cat_matrix.loc[var1, var2] = association(cont_table, method='cramer')
            plt.figure(figsize=(12,10))
            sns.heatmap(cat_matrix, annot=True, fmt='.2f', cmap='Blues',
                        cbar_kws={'label': "Cramér's V"}, square=True,
                        linewidths=0.5)
            plt.title(f'Categorical‑Categorical Association (Cramér\'s V) – {suffix_name} (cutoff={cutoff}%)')
            plt.tight_layout()
            outfile = os.path.join(out_dir, 'categorical_association_matrix.png')
            plt.savefig(outfile, dpi=150)
            plt.close()
            print("Categorical association matrix saved.")

        print(f"Plots saved in {out_dir}")

    # Run for md and recal
    if not md_df.empty:
        analyze_one(md_df, 'md', analysis_out_dir, cutoff)
    else:
        print("No .md samples found – skipping md analysis.")
    if not recal_df.empty:
        analyze_one(recal_df, 'recal', analysis_out_dir, cutoff)
    else:
        print("No .recal samples found – skipping recal analysis.")

    print("\nCoverage group analysis completed.\n")

# -----------------------------------------------------------------------------
# Command‑line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='WGS QC correlation analysis with high‑resolution plots.')
parser.add_argument('--config', required=True, help='Path to summary_config.ini file')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Read configuration
# -----------------------------------------------------------------------------
config = configparser.ConfigParser()
config.read(args.config)

qc_output_dir = config['Paths']['qc_output_dir']
multiqc_data_dir = config['Paths']['multiqc_data_dir']
qc_metrics_dir = config['Paths']['qc_metrics_dir']
output_base = os.path.dirname(config['Paths']['output_file'])
plot_dir = os.path.join(output_base, 'plots')
boxplot_dir = os.path.join(plot_dir, 'boxplots')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(boxplot_dir, exist_ok=True)

# Optional sex info file
sex_info_file = config.get('Paths', 'sex_info', fallback=None)

# Optional group cutoff for 30X coverage
cutoff_30X = config.get('Paths', 'group_cuttoff_30X', fallback=None)
if cutoff_30X is not None:
    cutoff_30X = float(cutoff_30X)

# -----------------------------------------------------------------------------
# Original WGS QC correlation analysis (with enhanced statistical annotations)
# -----------------------------------------------------------------------------

# Read MultiQC general stats
multiqc_file = os.path.join(multiqc_data_dir, 'multiqc_general_stats.txt')
if not os.path.exists(multiqc_file):
    raise FileNotFoundError(f'MultiQC general stats not found: {multiqc_file}')

df_multiqc = pd.read_csv(multiqc_file, sep='\t', low_memory=False)
df_multiqc['sample_base'] = df_multiqc['Sample'].apply(clean_sample_id)

# Split into .recal, .md, and read‑level rows
recal_rows = df_multiqc[df_multiqc['Sample'].str.endswith('.recal')].copy()
md_rows = df_multiqc[df_multiqc['Sample'].str.endswith('.md')].copy()
read_rows = df_multiqc[df_multiqc['Sample'].str.contains(r'-L00[0-9]_[12]', regex=True)].copy()

# --- Process samtools stats (from .recal) ---
samtools_cols = [col for col in recal_rows.columns if 'samtools_stats' in col]
if samtools_cols:
    samtools_df = recal_rows[['sample_base'] + samtools_cols].drop_duplicates('sample_base')
    samtools_df.set_index('sample_base', inplace=True)
else:
    samtools_df = pd.DataFrame()

# --- Process Picard duplicate stats (from .md) ---
picard_cols = [col for col in md_rows.columns if 'picard_mark_duplicates' in col]
if picard_cols:
    picard_df = md_rows[['sample_base'] + picard_cols].drop_duplicates('sample_base')
    picard_df.set_index('sample_base', inplace=True)
else:
    picard_df = pd.DataFrame()

# --- Process fastp and FastQC per‑read stats (robust column matching) ---
agg_list = []
for sample, group in read_rows.groupby('sample_base'):
    agg = {'sample_base': sample}

    # ----- fastp passed filter reads (total reads) -----
    reads_col = next((col for col in group.columns if 'passed_filter_reads' in col), None)
    total_reads = group[reads_col].sum() if reads_col is not None else np.nan
    agg['total_reads'] = total_reads

    # ----- fastp Q30 bases -----
    q30_bases_col = next((col for col in group.columns if 'q30_bases' in col), None)
    q30_bases = group[q30_bases_col].sum() if q30_bases_col is not None else np.nan
    agg['q30_bases'] = q30_bases   # store for later calculation

    # ----- fastp Q30 rate -----
    q30_rate_col = next((col for col in group.columns if 'q30_rate' in col), None)
    q30_rate_mean = group[q30_rate_col].mean() if q30_rate_col is not None else np.nan
    agg['q30_rate'] = q30_rate_mean

    # ----- fastp GC content -----
    gc_col = next((col for col in group.columns if 'gc_content' in col), None)
    gc_content = group[gc_col].mean() if gc_col is not None else np.nan
    agg['gc_content'] = gc_content

    # ----- FastQC percent duplicates -----
    dup_col = next((col for col in group.columns 
                    if 'percent_duplicates' in col and 'fastqc' in col.lower()), None)
    fastqc_dup = group[dup_col].mean() if dup_col is not None else np.nan
    agg['fastqc_dup_pct'] = fastqc_dup

    # ----- Compute total bases (if both q30_bases and q30_rate are available) -----
    if not np.isnan(q30_bases) and not np.isnan(q30_rate_mean) and q30_rate_mean > 0:
        total_bases = q30_bases / q30_rate_mean
    else:
        total_bases = np.nan
    agg['total_bases'] = total_bases

    agg_list.append(agg)

fastp_agg_df = pd.DataFrame(agg_list)
fastp_agg_df.set_index('sample_base', inplace=True)

# -----------------------------------------------------------------------------
# Read autosomal coverage report and clean IDs
# -----------------------------------------------------------------------------
cov_file = os.path.join(qc_output_dir, 'Autosomal_Coverage_Samples_report.tsv')
df_cov = pd.read_csv(cov_file, sep='\t')
df_cov.set_index('Sample', inplace=True)
# Clean the index using .map()
df_cov.index = df_cov.index.astype(str).map(clean_sample_id)
cov_cols = ['Percent_autosome_coverage_at_15X', 'Percent_autosome_coverage_at_30X']
df_cov = df_cov[cov_cols].copy()

# -----------------------------------------------------------------------------
# Read contamination report (VerifyBamID2) and clean IDs
# -----------------------------------------------------------------------------
contam_file = os.path.join(qc_output_dir, 'Samples_Contamination_report.tsv')
df_contam = pd.read_csv(contam_file, sep='\t')
df_contam.set_index('Sample', inplace=True)
df_contam.index = df_contam.index.astype(str).map(clean_sample_id)
df_contam = df_contam[['freemix']].copy()

# -----------------------------------------------------------------------------
# Read comprehensive QC metrics file (optional) and clean IDs
# -----------------------------------------------------------------------------
qc_metrics_file = os.path.join(qc_metrics_dir, 'QC_metricses_data_all_samples.tsv')
if os.path.exists(qc_metrics_file):
    df_qcmetrics = pd.read_csv(qc_metrics_file, sep='\t')
    df_qcmetrics.set_index('Biosample_id', inplace=True)
    df_qcmetrics.index = df_qcmetrics.index.astype(str).map(clean_sample_id)
    # Keep contamination and average coverage for cross‑check
    df_qcmetrics = df_qcmetrics[['Average_autosomal_coverage', 'VerifyBamID2_Contamination']].copy()
else:
    df_qcmetrics = pd.DataFrame()

# -----------------------------------------------------------------------------
# Read mosdepth XY coverage (for sex check) and clean IDs
# -----------------------------------------------------------------------------
xy_file = os.path.join(multiqc_data_dir, 'mosdepth-xy-coverage-plot.txt')
if os.path.exists(xy_file):
    df_xy = pd.read_csv(xy_file, sep='\t')
    # The file likely has columns: Sample, Chromosome X, Chromosome Y
    df_xy['sample_base'] = df_xy['Sample'].apply(clean_sample_id)
    # Keep only the relevant columns, remove duplicates
    df_xy = df_xy[['sample_base', 'Chromosome X', 'Chromosome Y']].drop_duplicates('sample_base')
    df_xy.set_index('sample_base', inplace=True)
else:
    print("Note: mosdepth-xy-coverage-plot.txt not found – sex check plot will be skipped.")
    df_xy = pd.DataFrame()

# -----------------------------------------------------------------------------
# Read sample information (sex, age, race) if provided and clean IDs
# -----------------------------------------------------------------------------
if sex_info_file and os.path.exists(sex_info_file):
    df_info = pd.read_csv(sex_info_file, sep='\t')
    # Map M/F to Male/Female
    if 'Sex' in df_info.columns:
        df_info['Sex'] = df_info['Sex'].map({'M': 'Male', 'F': 'Female'}).fillna(df_info['Sex'])
    # Set index to Library_ID and clean
    df_info.set_index('Library_ID', inplace=True)
    df_info.index = df_info.index.astype(str).map(clean_sample_id)
    print(f"Loaded sample information for {len(df_info)} samples.")
else:
    df_info = pd.DataFrame()
    if sex_info_file:
        print(f"Warning: sex_info file {sex_info_file} not found – proceeding without it.")

# -----------------------------------------------------------------------------
# Merge all dataframes (with inner joins to keep only common samples)
# -----------------------------------------------------------------------------
df_merged = df_cov.copy()
df_merged = df_merged.join(df_contam, how='left')
if not samtools_df.empty:
    df_merged = df_merged.join(samtools_df, how='left')
if not picard_df.empty:
    df_merged = df_merged.join(picard_df, how='left')
df_merged = df_merged.join(fastp_agg_df, how='left')
if not df_qcmetrics.empty:
    df_merged = df_merged.join(df_qcmetrics, how='left', rsuffix='_qcmetrics')
if not df_xy.empty:
    df_merged = df_merged.join(df_xy, how='left')
if not df_info.empty:
    df_merged = df_merged.join(df_info, how='left')

# Drop rows with missing coverage data
initial_n = len(df_merged)
df_merged.dropna(subset=['Percent_autosome_coverage_at_15X', 'Percent_autosome_coverage_at_30X'], inplace=True)
print(f"Dropped {initial_n - len(df_merged)} samples missing coverage data.")

# -----------------------------------------------------------------------------
# Rename columns for clarity – only rename columns that exist
# -----------------------------------------------------------------------------
rename_dict = {
    'Percent_autosome_coverage_at_15X': 'cov_15X_pct',
    'Percent_autosome_coverage_at_30X': 'cov_30X_pct',
    'freemix': 'contamination_freemix',
    'Samtools: stats_mqc-generalstats-samtools_stats-reads_mapped_percent': 'mapped_pct',
    'Samtools: stats_mqc-generalstats-samtools_stats-reads_properly_paired_percent': 'proper_paired_pct',
    'Samtools: stats_mqc-generalstats-samtools_stats-reads_MQ0_percent': 'mq0_pct',
    'Samtools: stats_mqc-generalstats-samtools_stats-error_rate': 'error_rate',
    'Picard: Mark Duplicates_mqc-generalstats-picard_mark_duplicates-PERCENT_DUPLICATION': 'dup_pct',
    'fastp_mqc-generalstats-fastp-pct_duplication': 'fastp_dup_pct',
    'FastQC_mqc-generalstats-fastqc-percent_duplicates': 'fastqc_dup_pct',
    'total_reads': 'total_reads',
    'total_bases': 'total_bases',
    'q30_rate': 'q30_rate',
    'gc_content': 'gc_content',
    'Average_autosomal_coverage': 'mean_cov',
    'VerifyBamID2_Contamination': 'contamination_verify',
    'Chromosome X': 'chrX_cov',
    'Chromosome Y': 'chrY_cov'
}
# Only keep keys that actually exist in the dataframe
rename_dict = {k: v for k, v in rename_dict.items() if k in df_merged.columns}
df_merged.rename(columns=rename_dict, inplace=True)

# -----------------------------------------------------------------------------
# Define low‑coverage flags: 15X uses fixed 90%, 30X uses cutoff from config (if available)
# -----------------------------------------------------------------------------
df_merged['low_15X'] = df_merged['cov_15X_pct'] < 90
if cutoff_30X is not None:
    df_merged['low_30X'] = df_merged['cov_30X_pct'] < cutoff_30X
else:
    # Fallback to 90% if no cutoff provided (shouldn't happen if group analysis runs)
    df_merged['low_30X'] = df_merged['cov_30X_pct'] < 90

# -----------------------------------------------------------------------------
# SEX CHECK PLOT – separate extreme samples for inferred and known sex
# -----------------------------------------------------------------------------
if 'chrX_cov' in df_merged.columns and 'chrY_cov' in df_merged.columns:
    set_publication_style()
    # Use mean_cov as autosomal coverage
    if 'mean_cov' in df_merged.columns:
        auto_cov = df_merged['mean_cov']
    else:
        auto_cov = None

    # Create a copy with non‑null XY values
    sex_df = df_merged[['chrX_cov', 'chrY_cov', 'mean_cov']].dropna().copy()
    if not sex_df.empty:
        # Infer sex
        sex_df['inferred_sex'] = sex_df.apply(lambda row: infer_sex(row['chrX_cov'], row['chrY_cov'], row['mean_cov']), axis=1)
        # Add known sex if available
        if 'Sex' in df_merged.columns:
            sex_df['known_sex'] = df_merged.loc[sex_df.index, 'Sex']
        else:
            sex_df['known_sex'] = 'Unknown'

        # --- Identify samples with extreme Y coverage in each inferred sex group ---
        inferred_label_samples = set()
        for sex_group in ['Male', 'Female']:
            group_df = sex_df[sex_df['inferred_sex'] == sex_group]
            if len(group_df) == 0:
                continue
            sorted_group = group_df.sort_values('chrY_cov')
            # Smallest 3 (lowest Y)
            for idx in sorted_group.index[:3]:
                inferred_label_samples.add(idx)
            # Largest 3 (highest Y)
            for idx in sorted_group.index[-3:]:
                inferred_label_samples.add(idx)

        # --- Identify samples with extreme Y coverage in each known sex group ---
        known_label_samples = set()
        for sex_group in ['Male', 'Female']:
            group_df = sex_df[sex_df['known_sex'] == sex_group]
            if len(group_df) == 0:
                continue
            sorted_group = group_df.sort_values('chrY_cov')
            # Smallest 3 (lowest Y)
            for idx in sorted_group.index[:3]:
                known_label_samples.add(idx)
            # Largest 3 (highest Y)
            for idx in sorted_group.index[-3:]:
                known_label_samples.add(idx)

        # Create a figure with two subplots: inferred and known (if available)
        fig, axes = plt.subplots(1, 2 if 'known_sex' in sex_df.columns else 1, figsize=(16,7) if 'known_sex' in sex_df.columns else (9,7))
        if 'known_sex' in sex_df.columns:
            ax1, ax2 = axes
        else:
            ax1 = axes

        # --- Inferred sex subplot ---
        sns.scatterplot(data=sex_df, x='chrX_cov', y='chrY_cov', hue='inferred_sex',
                        palette={'Male': 'blue', 'Female': 'pink', 'Unknown': 'gray'},
                        s=80, alpha=0.8, edgecolor='k', ax=ax1)
        ax1.set_xlabel('Chromosome X coverage')
        ax1.set_ylabel('Chromosome Y coverage')
        ax1.set_title('Inferred sex from coverage')

        # Add labels for extreme Y samples on inferred plot (color by inferred sex)
        texts_inferred = []
        for idx in inferred_label_samples:
            row = sex_df.loc[idx]
            # Choose text color based on inferred sex
            text_color = 'darkblue' if row['inferred_sex'] == 'Male' else 'darkred'
            texts_inferred.append(ax1.text(row['chrX_cov'], row['chrY_cov'], idx,
                                            fontsize=8, color=text_color, weight='bold', alpha=0.9))
        if ADJUST_TEXT_AVAILABLE and texts_inferred:
            adjust_text(texts_inferred, ax=ax1,
                        arrowprops=dict(arrowstyle='-', color='black', lw=1.5, shrinkA=5, shrinkB=5),
                        expand_points=(1.5, 1.5),
                        expand_text=(1.5, 1.5))

        # --- Known sex subplot (if available) ---
        if 'known_sex' in sex_df.columns:
            sns.scatterplot(data=sex_df, x='chrX_cov', y='chrY_cov', hue='known_sex',
                            palette={'Male': 'blue', 'Female': 'pink', 'Unknown': 'gray'},
                            s=80, alpha=0.8, edgecolor='k', ax=ax2)
            ax2.set_xlabel('Chromosome X coverage')
            ax2.set_ylabel('Chromosome Y coverage')
            ax2.set_title('Known sex (from sample info)')

            # Add labels for extreme Y samples on known plot (color by known sex)
            texts_known = []
            for idx in known_label_samples:
                row = sex_df.loc[idx]
                # Choose text color based on known sex
                text_color = 'darkblue' if row['known_sex'] == 'Male' else 'darkred'
                texts_known.append(ax2.text(row['chrX_cov'], row['chrY_cov'], idx,
                                             fontsize=8, color=text_color, weight='bold', alpha=0.9))
            if ADJUST_TEXT_AVAILABLE and texts_known:
                adjust_text(texts_known, ax=ax2,
                            arrowprops=dict(arrowstyle='-', color='black', lw=1.5, shrinkA=5, shrinkB=5),
                            expand_points=(1.5, 1.5),
                            expand_text=(1.5, 1.5))

            # Add a consistency check
            consistent = (sex_df['inferred_sex'] == sex_df['known_sex']).sum()
            total = len(sex_df)
            fig.suptitle(f'Sex check: {consistent}/{total} consistent', fontsize=14)

        plt.tight_layout()
        png_path = os.path.join(plot_dir, 'sex_check_scatter.png')
        pdf_path = os.path.join(plot_dir, 'sex_check_scatter.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Created sex check plot: {png_path}')
    else:
        print("Note: No valid XY data after dropping NaNs – sex check plot skipped.")
else:
    print("Note: chrX_cov or chrY_cov columns not available – sex check plot skipped.")

# -----------------------------------------------------------------------------
# Additional plots using sample information (if available) – now with stats
# -----------------------------------------------------------------------------
if not df_info.empty:
    set_publication_style()

    # Boxplot of 15X/30X coverage by Sex (with Mann-Whitney U)
    if 'Sex' in df_merged.columns and df_merged['Sex'].notna().any():
        for target in ['15X', '30X']:
            col = f'cov_{target}_pct'
            plot_df = df_merged[[col, 'Sex']].dropna()
            if len(plot_df) < 3:
                continue
            # Mann-Whitney U test
            groups = [plot_df[plot_df['Sex'] == s][col] for s in plot_df['Sex'].unique() if len(plot_df[plot_df['Sex'] == s]) >= 2]
            if len(groups) == 2:
                stat, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                stat, p_val = np.nan, np.nan
            fig, ax = plt.subplots(figsize=(6,5))
            sns.boxplot(data=plot_df, x='Sex', y=col, palette={'Male': 'lightblue', 'Female': 'lightpink'})
            sns.stripplot(data=plot_df, x='Sex', y=col, color='black', size=3, alpha=0.5, jitter=True)
            ax.set_ylabel(f'Genome covered at {target} (%)')
            ax.set_title(f'{target} coverage by sex')
            # Annotate with test result
            if not np.isnan(stat):
                ax.text(0.98, 0.98, f"U = {stat:.2f}\np = {p_val:.2e}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            png_path = os.path.join(plot_dir, f'{target}_coverage_by_sex.png')
            pdf_path = os.path.join(plot_dir, f'{target}_coverage_by_sex.pdf')
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Created {target} coverage by sex plot with statistics.')

    # Boxplot of 15X/30X coverage by Race (with Kruskal-Wallis)
    if 'Race' in df_merged.columns and df_merged['Race'].notna().any():
        race_counts = df_merged['Race'].value_counts()
        valid_races = race_counts[race_counts >= 3].index.tolist()
        if len(valid_races) >= 2:
            plot_df = df_merged[df_merged['Race'].isin(valid_races)]
            for target in ['15X', '30X']:
                col = f'cov_{target}_pct'
                # Kruskal-Wallis H test
                groups = [plot_df[plot_df['Race'] == r][col].dropna() for r in valid_races]
                if all(len(g) >= 2 for g in groups):
                    h_stat, p_val = kruskal(*groups)
                else:
                    h_stat, p_val = np.nan, np.nan
                fig, ax = plt.subplots(figsize=(max(6, len(valid_races)*0.8),5))
                sns.boxplot(data=plot_df, x='Race', y=col)
                sns.stripplot(data=plot_df, x='Race', y=col, color='black', size=3, alpha=0.5, jitter=True)
                ax.set_ylabel(f'Genome covered at {target} (%)')
                ax.set_title(f'{target} coverage by race')
                if not np.isnan(h_stat):
                    ax.text(0.98, 0.98, f"H = {h_stat:.2f}\np = {p_val:.2e}",
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.tight_layout()
                png_path = os.path.join(plot_dir, f'{target}_coverage_by_race.png')
                pdf_path = os.path.join(plot_dir, f'{target}_coverage_by_race.pdf')
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                fig.savefig(pdf_path, bbox_inches='tight')
                plt.close(fig)
                print(f'Created {target} coverage by race plot with statistics.')

    # Scatter plot of Age vs coverage (if Age is numeric) – already had correlation
    if 'Age' in df_merged.columns:
        # Try to convert Age to numeric, coercing errors to NaN
        df_merged['Age_numeric'] = pd.to_numeric(df_merged['Age'], errors='coerce')
        if df_merged['Age_numeric'].notna().sum() >= 3:
            for target in ['15X', '30X']:
                col = f'cov_{target}_pct'
                plot_df = df_merged[[col, 'Age_numeric']].dropna()
                if len(plot_df) >= 3:
                    fig, ax = plt.subplots(figsize=(6,5))
                    sns.scatterplot(data=plot_df, x='Age_numeric', y=col, s=60, alpha=0.7, edgecolor='k')
                    sns.regplot(data=plot_df, x='Age_numeric', y=col, scatter=False, color='red',
                                line_kws={'linestyle':'--'})
                    corr = safe_pearsonr(plot_df['Age_numeric'], plot_df[col])
                    ax.set_xlabel('Age (years)')
                    ax.set_ylabel(f'Genome covered at {target} (%)')
                    ax.set_title(f'{target} coverage vs age (r = {corr:.3f})')
                    plt.tight_layout()
                    png_path = os.path.join(plot_dir, f'{target}_coverage_vs_age.png')
                    pdf_path = os.path.join(plot_dir, f'{target}_coverage_vs_age.pdf')
                    fig.savefig(png_path, dpi=300, bbox_inches='tight')
                    fig.savefig(pdf_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Created {target} coverage vs age plot: {png_path}')

    # Scatter plot of Raw_Data_Size vs coverage (if present)
    if 'Raw_Data_Size' in df_merged.columns:
        df_merged['Raw_Data_Size'] = pd.to_numeric(df_merged['Raw_Data_Size'], errors='coerce')
        if df_merged['Raw_Data_Size'].notna().sum() >= 3:
            for target in ['15X', '30X']:
                col = f'cov_{target}_pct'
                plot_df = df_merged[[col, 'Raw_Data_Size']].dropna()
                if len(plot_df) >= 3:
                    fig, ax = plt.subplots(figsize=(6,5))
                    sns.scatterplot(data=plot_df, x='Raw_Data_Size', y=col, s=60, alpha=0.7, edgecolor='k')
                    sns.regplot(data=plot_df, x='Raw_Data_Size', y=col, scatter=False, color='red',
                                line_kws={'linestyle':'--'})
                    corr = safe_pearsonr(plot_df['Raw_Data_Size'], plot_df[col])
                    ax.set_xlabel('Raw Data Size (GB?)')
                    ax.set_ylabel(f'Genome covered at {target} (%)')
                    ax.set_title(f'{target} coverage vs Raw Data Size (r = {corr:.3f})')
                    plt.tight_layout()
                    png_path = os.path.join(plot_dir, f'{target}_coverage_vs_raw_size.png')
                    pdf_path = os.path.join(plot_dir, f'{target}_coverage_vs_raw_size.pdf')
                    fig.savefig(png_path, dpi=300, bbox_inches='tight')
                    fig.savefig(pdf_path, bbox_inches='tight')
                    plt.close(fig)
                    print(f'Created {target} coverage vs raw data size plot.')

# -----------------------------------------------------------------------------
# Read chromosome‑level metrics (if available) with robust handling
# -----------------------------------------------------------------------------
chrom_data = []
skipped_samples = []  # to summarize warnings
essential_chrom_cols = ['Percent_autosome_coverage_at_15X', 'Percent_autosome_coverage_at_30X']

for sample_id in df_merged.index:
    sample_dir = os.path.join(qc_metrics_dir, sample_id)
    chrom_file = os.path.join(sample_dir, f"{sample_id}_merged_all_chrom_qc_metrics.tsv")
    if not os.path.exists(chrom_file):
        continue

    try:
        df_chrom = pd.read_csv(chrom_file, sep='\t')
        # Clean column names: strip whitespace and remove trailing colons
        df_chrom = clean_column_names(df_chrom)
        
        # Remove duplicate columns (keep first occurrence)
        df_chrom = df_chrom.loc[:, ~df_chrom.columns.duplicated()]
        
        # Check if required columns exist
        required = ['Biosample_id', 'Chrom'] + essential_chrom_cols
        if not all(col in df_chrom.columns for col in required):
            skipped_samples.append(sample_id)
            continue
        
        # Subset to essential columns and drop rows where any of them is NaN
        df_sub = df_chrom[required].dropna(subset=essential_chrom_cols)
        if df_sub.empty:
            skipped_samples.append(sample_id)
            continue
        
        # Keep only main chromosomes
        df_sub = df_sub[df_sub['Chrom'].apply(is_main_chromosome)]
        if df_sub.empty:
            skipped_samples.append(sample_id)
            continue
        
        chrom_data.append(df_sub)
    except Exception as e:
        print(f"Error reading {chrom_file}: {e}")
        skipped_samples.append(sample_id)

if skipped_samples:
    print(f"Note: {len(skipped_samples)} samples had no usable chromosome data (skipped for chromosome plots).")

if chrom_data:
    df_chrom_all = pd.concat(chrom_data, ignore_index=True)

    # In case any duplicate (sample, chromosome) rows remain, aggregate by mean
    df_chrom_all = df_chrom_all.groupby(['Biosample_id', 'Chrom'], as_index=False).mean(numeric_only=True)

    # Merge low‑coverage flags from the main dataframe
    df_chrom_all = df_chrom_all.merge(df_merged[['low_15X', 'low_30X']],
                                       left_on='Biosample_id', right_index=True, how='left')

    # -------------------------------------------------------------------------
    # 1. Individual sample boxplots (saved in plots/boxplots/)
    # -------------------------------------------------------------------------
    set_publication_style()
    for sample_id in df_chrom_all['Biosample_id'].unique():
        sample_data = df_chrom_all[df_chrom_all['Biosample_id'] == sample_id]
        if sample_data.empty:
            continue

        # Create a figure with two subplots: 15X and 30X
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, target in zip(axes, ['15X', '30X']):
            col = f'Percent_autosome_coverage_at_{target}'
            if col not in sample_data.columns:
                ax.set_visible(False)
                continue
            # Sort chromosomes naturally
            chroms = sample_data['Chrom'].unique()
            chroms_sorted = natural_sort_chromosomes(chroms)
            plot_data = sample_data.set_index('Chrom')[col].loc[chroms_sorted].reset_index()
            sns.boxplot(data=plot_data, x='Chrom', y=col, ax=ax, color='skyblue')
            ax.tick_params(axis='x', rotation=90)
            ax.set_title(f'{sample_id} – {target} coverage per chromosome')
            ax.set_ylabel(f'Percent genome covered at {target}')
            ax.set_xlabel('Chromosome')
        plt.tight_layout()
        # Save
        png_path = os.path.join(boxplot_dir, f'{sample_id}_chrom_boxplots.png')
        pdf_path = os.path.join(boxplot_dir, f'{sample_id}_chrom_boxplots.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Created individual boxplots for {sample_id}')

    # -------------------------------------------------------------------------
    # 2. Combined heatmap of coverage for all samples
    # -------------------------------------------------------------------------
    for target in ['15X', '30X']:
        col = f'Percent_autosome_coverage_at_{target}'
        if col not in df_chrom_all.columns:
            print(f"Column {col} not found in chromosome data, skipping heatmap.")
            continue

        # Pivot table: rows = samples, columns = chromosomes, values = coverage
        pivot = df_chrom_all.pivot(index='Biosample_id', columns='Chrom', values=col)
        # Sort chromosomes naturally
        chroms_sorted = natural_sort_chromosomes(pivot.columns.tolist())
        pivot = pivot[chroms_sorted]

        # Sort samples by overall coverage (from df_merged)
        sample_order = df_merged.sort_values(f'cov_{target}_pct').index
        pivot = pivot.loc[sample_order.intersection(pivot.index)]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(chroms_sorted)*0.4), max(8, len(pivot)*0.3)))
        sns.heatmap(pivot, annot=False, cmap='RdYlGn_r', cbar_kws={'label': f'% coverage at {target}'},
                    linewidths=0.5, linecolor='gray', ax=ax)
        ax.set_title(f'All samples – chromosome‑wise {target} coverage')
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('Sample')
        plt.tight_layout()

        png_path = os.path.join(plot_dir, f'chrom_{target}_heatmap.png')
        pdf_path = os.path.join(plot_dir, f'chrom_{target}_heatmap.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        print(f'Created combined heatmap: {png_path}')

    # -------------------------------------------------------------------------
    # 3. Group boxplot comparing low vs high coverage samples
    # -------------------------------------------------------------------------
    set_publication_style()
    for target, low_flag in [('15X', 'low_15X'), ('30X', 'low_30X')]:
        col = f'Percent_autosome_coverage_at_{target}'
        if col not in df_chrom_all.columns:
            continue

        plt.figure(figsize=(max(12, len(df_chrom_all['Chrom'].unique())*0.3), 6))
        ax = sns.boxplot(data=df_chrom_all, x='Chrom', y=col, hue=low_flag,
                         palette={True: 'red', False: 'blue'}, showfliers=False)
        # Add strip plot for individual points
        sns.stripplot(data=df_chrom_all, x='Chrom', y=col, hue=low_flag,
                      dodge=True, color='black', size=2, alpha=0.5, ax=ax, legend=False)
        ax.tick_params(axis='x', rotation=90)
        ax.set_title(f'Per‑chromosome {target} coverage by overall sample quality')
        ax.set_ylabel(f'Percent genome covered at {target}')
        ax.set_xlabel('Chromosome')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Use the appropriate threshold in legend
            if target == '30X' and cutoff_30X is not None:
                low_label = f'<{cutoff_30X}%'
            else:
                low_label = '<90%'
            ax.legend(handles[:2], [f'≥{90 if target=="15X" else cutoff_30X}%', low_label], title='Overall coverage')
        plt.tight_layout()
        png_path = os.path.join(plot_dir, f'chrom_{target}_group_boxplot.png')
        pdf_path = os.path.join(plot_dir, f'chrom_{target}_group_boxplot.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        print(f'Created group boxplot: {png_path}')

else:
    print("No chromosome‑level data found – skipping chromosome plots.")

# -----------------------------------------------------------------------------
# Define metrics to correlate (sample‑level)
# -----------------------------------------------------------------------------
metrics_of_interest = [
    'total_reads', 'total_bases', 'mapped_pct', 'proper_paired_pct',
    'dup_pct', 'fastp_dup_pct', 'fastqc_dup_pct',
    'contamination_freemix', 'contamination_verify',
    'q30_rate', 'gc_content', 'mq0_pct', 'error_rate', 'mean_cov',
    'chrX_cov', 'chrY_cov', 'Raw_Data_Size'
]
available_metrics = [m for m in metrics_of_interest if m in df_merged.columns]

# -----------------------------------------------------------------------------
# Generate publication‑quality scatter plots with highlighting
# -----------------------------------------------------------------------------
set_publication_style()

# Option to label low‑coverage points (disable if too many)
MAX_LABELS = 15   # only label if number of low samples <= this

for cov_target, low_flag in [('cov_15X_pct', 'low_15X'), ('cov_30X_pct', 'low_30X')]:
    target_label = '15X' if cov_target == 'cov_15X_pct' else '30X'
    low_samples = df_merged[df_merged[low_flag]].index.tolist()
    num_low = len(low_samples)
    # Determine threshold for legend
    if target_label == '30X' and cutoff_30X is not None:
        low_threshold = cutoff_30X
    else:
        low_threshold = 90

    for metric in available_metrics:
        # Prepare data, dropping missing values for this metric
        plot_df = df_merged[[cov_target, metric, low_flag]].dropna(subset=[cov_target, metric])
        if len(plot_df) < 3:
            print(f'Skipping {metric} vs {target_label}: insufficient data')
            continue

        x = plot_df[metric]
        y = plot_df[cov_target]
        low = plot_df[low_flag]

        corr = safe_pearsonr(x, y)

        # Create figure
        fig, ax = plt.subplots(figsize=(6,5))

        # Scatter plot with two colors
        sns.scatterplot(x=x, y=y, hue=low, palette={True: 'red', False: 'blue'},
                        s=60, alpha=0.7, edgecolor='k', ax=ax, legend='brief')

        # Add regression line (overall, not per group)
        sns.regplot(x=x, y=y, ax=ax, scatter=False, color='gray',
                    line_kws={'linestyle':'--', 'linewidth':1.5, 'alpha':0.6})

        # Optionally label low‑coverage points if not too many
        if num_low <= MAX_LABELS and num_low > 0:
            low_points_df = plot_df[low]  # subset where low is True
            for idx in low_points_df.index:
                ax.text(plot_df.loc[idx, metric] + 0.02 * x.max(),
                        plot_df.loc[idx, cov_target],
                        idx, fontsize=8, alpha=0.7)

        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(f'Genome covered at {target_label} (%)', fontsize=12)
        ax.set_title(f'{metric} vs {target_label} coverage', fontsize=14, weight='bold')

        # Add correlation annotation
        if not np.isnan(corr):
            ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Adjust legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Rename legend labels using the appropriate threshold
            new_labels = [f'Coverage ≥{low_threshold}%', f'Coverage <{low_threshold}% ({num_low} samples)']
            ax.legend(handles, new_labels, title=target_label, loc='best')

        plt.tight_layout()

        # Save both PNG (300 dpi) and PDF
        safe_metric = metric.replace(' ', '_').replace('%', 'pct')
        png_path = os.path.join(plot_dir, f'corr_{target_label}_{safe_metric}.png')
        pdf_path = os.path.join(plot_dir, f'corr_{target_label}_{safe_metric}.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)

        print(f'Created plot: {png_path}')

# -----------------------------------------------------------------------------
# Create a correlation matrix heatmap (optional)
# -----------------------------------------------------------------------------
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
# Exclude boolean low flags from correlation matrix
numeric_cols = [c for c in numeric_cols if c not in ['low_15X', 'low_30X']]
corr_data = df_merged[numeric_cols].dropna(axis=1, how='all').dropna()
if len(corr_data) > 1:
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True,
                cbar_kws={'shrink': 0.8}, ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Correlation matrix of QC metrics', fontsize=16, weight='bold')
    plt.tight_layout()

    png_path = os.path.join(plot_dir, 'correlation_matrix.png')
    pdf_path = os.path.join(plot_dir, 'correlation_matrix.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Created correlation matrix: {png_path}')

# -----------------------------------------------------------------------------
# Diagnostic: Compare sample counts with expected total from sex_info
# -----------------------------------------------------------------------------
if not df_info.empty:
    expected_samples = set(df_info.index)
    present_samples = set(df_merged.index)
    missing = expected_samples - present_samples
    if missing:
        print(f"\nWARNING: {len(missing)} samples from sample info are missing in merged data:")
        print(sorted(missing))
    else:
        print("\nAll samples from sample info are present in merged data.")

# -----------------------------------------------------------------------------
# Run the additional coverage group analysis if cutoff is provided
# -----------------------------------------------------------------------------
if cutoff_30X is not None:
    run_coverage_group_analysis(config, output_base, cutoff_30X)
else:
    print("group_cuttoff_30X not found in config; skipping coverage group analysis.")

print(f'\nAll plots saved to: {plot_dir}')