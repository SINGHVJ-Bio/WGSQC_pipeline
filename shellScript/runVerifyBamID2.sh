#!/bin/bash

workdir=$1
bam=$2
resource_path=$3
ref_fasta=$4
sample=$5
bampath=$6

mkdir -p $workdir/res_VerifyBamID2
PC_num=4
threads=4

start=$(date +%s)
echo "${start}"

verifybamid2 \
  --SVDPrefix $resource_path/1000g.phase3.100k.b38.vcf.gz.dat \
  --Reference $ref_fasta \
  --BamFile $bampath/$bam \
  --Output $workdir/res_VerifyBamID2/$sample \
  --WithinAncestry \
  --NumPC $PC_num \
  --NumThread $threads

echo "Total_Secs_Process $(($(date +%s) - ${start}))"