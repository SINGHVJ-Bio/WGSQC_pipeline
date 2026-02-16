#!/bin/bash

autosomes_bed=$1
gap=$2
workdir=$3

zcat $gap | cut -f2-4 | egrep -w '^chr[1-9]|chr[1-2][0-9]' | sort -k1,1V -k2,2n > $workdir/gap_regions.bed
bedtools subtract -a $workdir/$autosomes_bed -b $workdir/gap_regions.bed > $workdir/sorted_autosomes.bed