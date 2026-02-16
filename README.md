# WGSQC_pipeline (WGS/WES QC Pipeline)

A Python pipeline to compute coverage QC metrics and run VerifyBamID2 on WGS or WES samples.

## Features
- Per‑chromosome coverage statistics (median, mean, % bases at 5X, 10X, …, 30X).
- Contamination estimation with VerifyBamID2.
- Automatic BED creation for autosomes (WGS: from .fai minus gaps; WES: from GTF).
- Parallel processing for speed.
- Optional BAM download from S3.
- Post‑processing report with interactive HTML and TSV summaries.

## Requirements
- Python 3.8+ with: `pandas`, `numpy`, `pysam`, `plotly`, `tqdm`
- `bedtools`, `verifybamid2`, `awscli` (if downloading)
- Linux utilities: `zcat`, `cut`, `egrep`, `sort`
- Reference files: FASTA with .fai, gap.txt.gz, GTF (for WES), VerifyBamID2 resource files

## Installation
git clone <repository> /path/to/WGSQC_pipeline
cd /path/to/WGSQC_pipeline


## Configuration
Edit `data/config.ini` to set all paths and parameters (see example in the file).

Prepare a sample sheet (TSV) with columns `Library_ID` and `Path` (subdirectory under `bam_base_dir` that leads to the recalibrated BAM).

## Usage
Run the full pipeline:

python run_pipeline.py --config data/config.ini --steps download qc verify combine

Steps can be selected individually; default is qc verify combine.
After the pipeline finishes, generate an interactive report (requires MultiQC outputs):

python generate_qc_report.py \
    --qc-json /path/to/QC_metricses_data_all_samples_all_chrom.json \
    --multiqc-dir /path/to/multiqc_data \
    --outdir /path/to/report_output \
    --config data/thresholds.ini   # optional


## Outputs

workdir/QC_metrics/ – per‑sample per‑chromosome JSON/TSV, merged sample files, final summary TSV.
workdir/res_VerifyBamID2/ – VerifyBamID2 output files.
Report outputs: HTML file and multiple TSV summaries.

