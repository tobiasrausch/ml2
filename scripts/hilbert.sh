#!/bin/bash

# Fetch covid lineages
MINNUM=500
ROOT=${1}
if [ -f ${ROOT}/summary.tsv ]
then
    for LIN in `cut -f 5,6 ${ROOT}/summary.tsv | grep "^pass" | cut -f 2 | sort | uniq -c | awk '$1>'${MINNUM}' {print $2;}' | grep -v "Unassigned"`
    do
	for SAMPLE in `cut -f 1,5,6 ${ROOT}/summary.tsv | grep "pass" | grep -P "\t${LIN}$" | head -n ${MINNUM} | cut -f 1`
	do
	    if [ -f ${ROOT}/../../../results/${SAMPLE}/${SAMPLE}.srt.bam ]
	    then
		/opt/dev/wally/src/wally hilbert -g /opt/dev/covid19/ref/NC_045512.2.fa -r NC_045512.2:1-30000:${LIN}_${SAMPLE} ${ROOT}/../../../results/${SAMPLE}/${SAMPLE}.srt.bam
	    fi
	done
    done
fi
