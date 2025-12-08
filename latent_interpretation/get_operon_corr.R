

library(stringr)
library(reshape2)



corr_files_all <- list.files("/work/gd134/corr_comp_for_op/", pattern = "*.csv")
corr_names_all <- str_extract(corr_files_all, "([a-zA-Z0-9_.]*)(_pcorr.csv)", group = 1)

op_cor_files <- list.files('/work/gd134/operon_corr_comp/', pattern = "*_opcorr.csv")
op_cor_names <- str_extract(op_cor_files, "([a-zA-z0-9_:.]*)(_opcorr.csv)", group = 1)


corr_files <- corr_files_all[!(corr_names_all %in% op_cor_names)]
corr_names <- str_extract(corr_files, "([a-zA-z0-9_:.]*)(_pcorr.csv)", group = 1)

comp <- read.csv('../data_files/sepi_pan_genome_log_counts_norm_01.csv', 
                 stringsAsFactors = F)
print(dim(comp))

se_ops <- read.csv('sepi_pan_operon_format.txt', sep = '\t', 
                   stringsAsFactors = F)

corr_list <- lapply(corr_names,  function(x){
  print(x)
  t <- read.csv(paste0(paste0('/work/gd134/corr_comp_for_op/',x), '_pcorr.csv'),
                stringsAsFactors = F,
                row.names=1)
  print(dim(t))
  t[is.na(t)] <- 0
  rownames(t) <- comp$X
  colnames(t) <- comp$X
  rnames <- rownames(t)
  by_op <- lapply(se_ops$Genes, function(y){
    op_names <- rnames[rnames %in% unlist(str_split(y, ';'))]
    cd <- data.matrix(t[op_names, op_names])
    cd[lower.tri(cd)] <- NA
    #print(dim(cd))
    rownames(cd) <- colnames(cd)
    tm <- melt(as.matrix(cd))
    #tm$set <- "operon"
    tm[(!is.na(tm$value)) & (! tm$Var1 == tm$Var2),]
  })
  names(by_op) <- se_ops$OperonID
  by_nop <- lapply(se_ops$Genes, function(y){
    op_names <- rnames[rnames %in% unlist(str_split(y, ';'))]
    nop_names <- rnames[!(rnames %in% unlist(str_split(y, ';')))]
    cd <- data.matrix(t[op_names, sample(nop_names,1000)])
    #cd[lower.tri(cd)] <- NA
    #print(dim(cd))
    #rownames(cd) <- colnames(cd)
    tm <- melt(as.matrix(cd))
    #tm$set <- "noperon"
    tm[(!is.na(tm$value)) & !(tm$value == 0) ,]
  })
  names(by_nop) <- se_ops$OperonID
  
  se_operon_corrs_df <- data.frame( do.call(rbind, by_op))
  se_operon_corrs_df_m <- melt(se_operon_corrs_df)
  se_operon_corrs_df_m$set <- "operon"
  se_noperon_corrs_df <- data.frame( do.call(rbind, by_nop))
  se_noperon_corrs_df_m <- melt(se_noperon_corrs_df)
  se_noperon_corrs_df_m$set <- "noperon"
  se_operon_corrs_df_m_a <- rbind(se_operon_corrs_df_m,se_noperon_corrs_df_m)
  write.csv(se_operon_corrs_df_m_a, paste0("/work/gd134/operon_corr_comp/",paste0(x,"_opcorr.csv")))
  se_operon_corrs_df_m_a
}

)


