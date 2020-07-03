## Barghi et al. (2019) Data

- `Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID.sync.gz.orig`: Gzipped sync file
  of all samples, downloaded from
  https://datadryad.org/resource/doi:10.5061/dryad.rr137kn. Then, I remove
  columns the authors have added to the Sync file:

      zcat Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID.sync.gz.orig | cut -f 1-73 | gzip > Dsim_F0-F60_Q20_polymorphic_CMH_FET_blockID.sync.gz

- `barghi_et_al_2019_design.txt`: These are the sample names, scraped from the README on 
   the datadryad link above. They contain all metadata (after reshaping done in notebook).

