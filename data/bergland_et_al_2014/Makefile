all: 6d_v7.3_output_fixed.vcf.gz 6d_v7.3_output_fixed_filtered.vcf.gz

6d_v7.3_output_fixed.vcf.gz: 6d_v7.3_output.vcf.gz
	zcat 6d_v7.3_output.vcf.gz| sed 's/#CHR/#CHROM/' | gzip > 6d_v7.3_output_fixed.vcf.gz


6d_v7.3_output_fixed_filtered.vcf.gz: 6d_v7.3_output.vcf.gz
	(zcat 6d_v7.3_output.vcf.gz | sed 's/#CHR/#CHROM/' | grep "^#"; zcat 6d_v7.3_output.vcf.gz | grep -v "^#" | grep "USED=TRUE") | gzip > 6d_v7.3_output_fixed_filtered.vcf.gz
