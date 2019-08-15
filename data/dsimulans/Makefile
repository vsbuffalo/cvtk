data: dsim-all-chromosome-M252_draft_4-chrnamesok.fa dsim-M252-popgen-ann-r1.1.sorted.gtf dsim-M252-seqlens.txt

dsim-all-chromosome-M252_draft_4-chrnamesok.fa:
	wget https://datadryad.org/bitstream/handle/10255/dryad.97303/dsim-all-chromosome-M252_draft_4-chrnamesok.fa

dsim-M252-popgen-ann-r1.1.gtf:
	wget https://datadryad.org/bitstream/handle/10255/dryad.97304/dsim-M252-popgen-ann-r1.1.gtf

dsim-M252-popgen-ann-r1.1.sorted.gtf: dsim-M252-popgen-ann-r1.1.gtf
	sort -k1,1 -k4,4n dsim-M252-popgen-ann-r1.1.gtf > dsim-M252-popgen-ann-r1.1.sorted.gtf


dsim-M252-seqlens.txt: ./dsim-all-chromosome-M252_draft_4-chrnamesok.fa
	bioawk -c fastx '{print $$name, length($$seq)}' $< > $@
