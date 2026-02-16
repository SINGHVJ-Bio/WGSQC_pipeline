#!/bin/bash
# Download BAM and index from S3 for a given sample

bucketdownload=$1
bamdir=$2
p_id=$3          # Not used directly, kept for compatibility
S_id=$4
source_path=$5

cmd="aws s3 cp s3://${bucketdownload}/${source_path}/results_${S_id}/preprocessing/recalibrated/${S_id}/${S_id}.recal.bam ${bamdir}/ --profile enabl-wgs-alignment"
echo $cmd
$cmd

cmd="aws s3 cp s3://${bucketdownload}/${source_path}/results_${S_id}/preprocessing/recalibrated/${S_id}/${S_id}.recal.bam.bai ${bamdir}/ --profile enabl-wgs-alignment"
echo $cmd
$cmd