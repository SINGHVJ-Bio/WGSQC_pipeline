#!/usr/bin/env python3
"""
VerifyBamID2 analysis module.
Runs VerifyBamID2 on each sample in parallel chunks.
"""

import os
import logging
import subprocess

import utils

logger = logging.getLogger(__name__)


def run_verifybamid2_sample(cfg, lib_id, bam_path, bam_file):
    """
    Run VerifyBamID2 for a single sample using the shell script.
    """
    script = os.path.join(cfg['pipeline_path'], 'shellScript', 'runVerifyBamID2.sh')
    cmd = [
        'bash', script,
        cfg['workdir'],
        bam_file,
        cfg['verifybamid_resource_path'],
        cfg['references'],
        lib_id,
        bam_path
    ]
    logger.debug(f"Running VerifyBamID2 for {lib_id}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def process_chunk(chunk_df, cfg):
    """
    Process a chunk of samples for VerifyBamID2.
    """
    for _, row in chunk_df.iterrows():
        lib_id = row['Library_ID']
        bam_path = row['bampath']
        bam_file = f"{lib_id}.recal.bam"
        # Check if BAM exists
        if not os.path.exists(os.path.join(bam_path, bam_file)):
            logger.error(f"BAM file missing for {lib_id} in {bam_path}")
            continue
        logger.info(f"Running VerifyBamID2 for {lib_id}")
        run_verifybamid2_sample(cfg, lib_id, bam_path, bam_file)
    return len(chunk_df)


def run_verify(cfg, sample_df):
    """
    Main entry for VerifyBamID2: run on samples in parallel.
    """
    num_proc = min(int(cfg.get('num_processes', 4)), len(sample_df))
    utils.process_chunks_parallel(sample_df, num_proc, process_chunk, (cfg,))