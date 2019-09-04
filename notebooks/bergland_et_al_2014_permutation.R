library(tidyverse)
library(feather)


read_vcf <- function(file) {
  vcf_cols <- cols(.default=col_character())
  res <- read_tsv(file, col_types=vcf_cols, comment="##")
  colnames(res) <- tolower(sub('^#', '', colnames(res)))
  mutate(res, pos=as.integer(pos))
}

split_info <- function(x) {
  keyvals <- lapply(strsplit(x, ';', fixed=TRUE)[[1]], 
                    function(y) strsplit(y, '=', fixed=TRUE)[[1]])
  out <- do.call(rbind, keyvals)
  row <- as.list(out[, 2])
  names(row) <- out[, 1]
  as_tibble(row)
}

# from VCF header
CHROM_COUNTS <- c(fl1 = 78, fl2 = 96, ga = 102, sc = 96, nc = 92, me1 = 172, 
                  me2 = 150, pa_7_2009 = 110, pa_11_2009 = 148, pa_7_2010 = 232, 
                  pa_11_2010 = 66, pa_7_2011 = 150, pa_10_2011 = 94, pa_11_2011 = 100)

FEATHER_FILE <- '../data/bergland_et_al_2014/bergland_et_al_2014.feather'

# there's a bug in feather, so we regenerate every time :( 
if (TRUE || !file.exists(FEATHER_FILE)) {
  vcf <- read_vcf('../data/bergland_et_al_2014/6d_v7.3_output.vcf.gz') %>%
           rename(format=formart) 
  vcf <- vcf %>% mutate(info=map(info, split_info)) %>% unnest(info)
  vcf <- vcf %>% gather(sample, genotype, fl1:pa_11_2011) %>% 
          # rename across-sample read depth
          rename(aveDP = DP) %>% 
          separate(genotype, into=c('AD', 'DP'), sep=':', convert=TRUE)
  write_feather(vcf, FEATHER_FILE)
} else {
  vcf <- read_feather(FEATHER_FILE)
}

# use only the SNPs they use
vcf_used <- vcf %>% filter(as.logical(USED))
vcf_used <- vcf_used %>% unite(locus, chr, pos, remove=FALSE)

# nest the seasonal data from PA orchard
CACHED_SEASONAL_NESTED <- '../data/bergland_et_al_2014/bergland_et_al_2014_nested.Rdata'
if (!file.exists(CACHED_SEASONAL_NESTED)) {
  pa <- vcf_used %>% select(-(id:format), -EFF) %>% filter(str_detect(sample, "^pa")) %>% 
    # remove the odd timepoint -- the paper uses all 7 and 11 pairs except for 2011; 
    # there it uses pa_10_2011
    filter(sample != 'pa_11_2011') %>%
    # classify as fall/spring
    extract(sample, regex="pa_(\\d+)_.*", into='month', remove=FALSE) %>%
    mutate(season=ifelse(month == 7, 'spring', 'fall')) %>%
    nest(sample:season) 
  save(pa, file=CACHED_SEASONAL_NESTED)
} else {
  load(CACHED_SEASONAL_NESTED)
}

pa <- pa %>% mutate(SP=as.numeric(SP))


check_data <- function(x, min_nonzero=0.5) {
  (mean(x$AD == 0) < min_nonzero) & (mean(x$DP == 0) < min_nonzero)
}

extract_pval <- function(x) {
 s = summary(x)$coefficients
 if (nrow(s) == 1) return(NA)
 s[2, 4]
}

fit_seasonal <- function(x, chrom_counts, pvalue_only=TRUE) {
  neff <- with(x, floor((DP*chrom_counts[sample]-1)/(DP+chrom_counts[sample])))
  fit <- glm(AD/DP ~ season, weights=neff, data=x, family=binomial())
  if (!pvalue_only)
    return(fit)
  return(extract_pval(fit))
}

fit_seasonal_permuted <- function(x, chrom_counts, pvalue_only=TRUE) {
  neff <- with(x, floor((DP*chrom_counts[sample]-1)/(DP+chrom_counts[sample])))
  permuted_season = sample(x$season)
  fit <- glm(AD/DP ~ permuted_season, weights=neff, data=x, family=binomial())
  if (!pvalue_only)
    return(fit)
  return(extract_pval(fit))
}

CACHED_SEASONAL_MODELS <- '../data/bergland_et_al_2014/bergland_et_al_2014_models.Rdata'
if (!file.exists(CACHED_SEASONAL_MODELS)) {
  pa_res <- pa %>% 
              mutate(valid_locus=map_lgl(data, check_data)) %>% 
      	      mutate(seasonal_pvalue=map_dbl(data, fit_seasonal, chrom_counts=CHROM_COUNTS)) %>%
      	      mutate(permuted_pvalue=map_dbl(data, fit_seasonal_permuted, chrom_counts=CHROM_COUNTS)) 
  
  save(pa_res, file=CACHED_SEASONAL_MODELS)
} else {
  load(CACHED_SEASONAL_MODELS)
}

dres <- pa_res %>% select(locus:FallF, valid_locus, seasonal_pvalue, permuted_pvalue)

write_feather(dres, '../data/bergland_et_al_2014/bergland_et_al_2014_permutation.feather')
write_csv(dres, '../data/bergland_et_al_2014/bergland_et_al_2014_permutation.csv')

# figure of p-value scatter
pdf('../data/bergland_et_al_2014/bergland_pvalue_scatter_R.pdf', height=3.42, width=3.42)
plot(pa_res$SP, pa_res$seasonal_pvalue, pch='.', col=alpha('black', 0.1), 
     ylab='seasonal p-value', xlab="original study's p-value", bty='n')
text(0.05, 0.9, paste0("ρ = ", round(cor(pa_res$SP, pa_res$seasonal_pvalue), 2)))
dev.off()

# figure of histogram comparison
pdf('../data/bergland_et_al_2014/bergland_pvalue_10bin_hist_R.pdf', height=3.42, width=3.42)
nbins <- 10
permuted_hist <- hist(dres$permuted_pvalue, nbins, plot=FALSE)
seasonal_hist <- hist(dres$seasonal_pvalue, nbins, plot=FALSE)

histdat <- rbind(permuted_hist$counts, seasonal_hist$counts)
labels <- apply(cbind(permuted_hist$breaks[1:nbins], permuted_hist$breaks[2:(nbins+1)]), 1, function(x) sprintf("(%.1f, %.1f]", x[1], x[2]))
colnames(histdat) <- labels
rownames(histdat) <- c('permuted', 'seasonal')
cols <- c('gray22', 'gray60')
barplot(histdat, beside=TRUE, border=0, angle=90, axes=FALSE, col=cols)
legend(x='topright', legend=rownames(histdat), fill=cols, bty='n', border=FALSE)
axis(2)
dev.off()

pdf('../data/bergland_et_al_2014/bergland_pvalue_60bin_hist_R.pdf', height=3.42, width=3.42)
nbins <- 60
permuted_hist <- hist(dres$permuted_pvalue, nbins, plot=FALSE)
seasonal_hist <- hist(dres$seasonal_pvalue, nbins, plot=FALSE)

nb <- length(permuted_hist$breaks)-1

histdat <- rbind(permuted_hist$counts, seasonal_hist$counts)
labels <- apply(cbind(permuted_hist$breaks[1:nb], permuted_hist$breaks[2:(nb+1)]), 1, function(x) sprintf("(%.1f, %.1f]", x[1], x[2]))
colnames(histdat) <- labels
rownames(histdat) <- c('permuted', 'seasonal')
cols <- c('gray22', 'gray60')
barplot(histdat, beside=TRUE, border=0, angle=90, axes=FALSE, col=cols)
legend(x='topright', legend=rownames(histdat), fill=cols, bty='n', border=FALSE)
axis(2)
dev.off()
