

library(stringr)
library(reshape2)



corr_files_all <- list.files("/work/gd134/corr_comp_for_op/", pattern = "*csv")
corr_names_all <- str_extract(corr_files_all, "([a-zA-Z0-9_.]*)(_pcorr.csv)", group = 1)

op_cor_files <- list.files('/work/gd134/kegg_corr_comp/', pattern = "*_kgcorr.csv")
op_cor_names <- str_extract(op_cor_files, "([a-zA-z0-9_:.]*)(_kgcorr.csv)", group = 1)


corr_files <- corr_files_all[!(corr_names_all %in% op_cor_names)]
corr_names <- str_extract(corr_files, "([a-zA-z0-9_:.]*)(_pcorr.csv)", group = 1)

comp <- read.csv('../data_files/sepi_pan_genome_log_counts_norm_01.csv', 
                 stringsAsFactors = F)
print(dim(comp))

se_ops <- read.csv('sepi_pan_KEGG_format.txt', sep = '\t', 
                   stringsAsFactors = F)



name_map1 <- read.csv('sepi_pan_genome_gene_name_corrector_map.csv', stringsAsFactors = F, row.names = 1)
names_m1 <- name_map1[comp$X,"X2"]
names_m1[is.na(names_m1)] <- "noname"
names_mu1 <- make.unique(names_m1)

name_map <- read.csv('sepi_pan_genome_gene_names.csv', stringsAsFactors = F, row.names = 1)
names_m <- name_map[names_mu1,"X2"]
names_m[is.na(names_m)] <- "noname"
names_mu <- make.unique(names_m)


corr_list <- lapply(corr_names,  function(x){
  print(x)
  t <- read.csv(paste0(paste0('/work/gd134/corr_comp_for_op/',x), '_pcorr.csv'),
                stringsAsFactors = F,
                row.names=1)
  print(dim(t))
  t[is.na(t)] <- 0
  rownames(t) <- names_mu
  colnames(t) <- names_mu
  rnames <- rownames(t)
  by_op <- lapply(c(1:nrow(se_ops)), function(z){
    y <- se_ops$Genes[z]
    op_names <- rnames[rnames %in% unlist(str_split(y, ';'))]
    cd <- data.matrix(t[op_names, op_names])
    cd[lower.tri(cd)] <- NA
    #print(dim(cd))
    rownames(cd) <- colnames(cd)
    tm <- melt(as.matrix(cd))
    if(nrow(tm) > 0){
      tm$path <- se_ops$KEGGPathID[z]
    }
    #tm$set <- "operon"
    tm2 <- tm[(!is.na(tm$value)) & (! tm$Var1 == tm$Var2),]
    #tm2$path <- se_ops$KEGGPathID[z]
    tm2
  })
  names(by_op) <- se_ops$KEGGPathID
  by_nop <- lapply(c(1:nrow(se_ops)), function(z){
    y <- se_ops$Genes[z]
    op_names <- rnames[rnames %in% unlist(str_split(y, ';'))]
    nop_names <- rnames[!(rnames %in% unlist(str_split(y, ';')))]
    cd <- data.matrix(t[op_names, sample(nop_names,1000)])
    #cd[lower.tri(cd)] <- NA
    #print(dim(cd))
    #rownames(cd) <- colnames(cd)
    tm <- melt(as.matrix(cd))
    if(nrow(tm) > 0){
      tm$path <- se_ops$KEGGPathID[z]
    }
    #tm$set <- "noperon"
    tm2 <- tm[(!is.na(tm$value)) & !(tm$value == 0) ,]
    #tm2$path <- se_ops$KEGGPathID[z]
    tm2
  })
  names(by_nop) <- se_ops$KEGGPathID
  
  se_operon_corrs_df <- data.frame( do.call(rbind, by_op))
  se_operon_corrs_df_m <- melt(se_operon_corrs_df)
  se_operon_corrs_df_m$set <- "kegg"
  se_noperon_corrs_df <- data.frame( do.call(rbind, by_nop))
  se_noperon_corrs_df_m <- melt(se_noperon_corrs_df)
  se_noperon_corrs_df_m$set <- "nkegg"
  se_operon_corrs_df_m_a <- rbind(se_operon_corrs_df_m,se_noperon_corrs_df_m)
  write.csv(se_operon_corrs_df_m_a, paste0("/work/gd134/kegg_corr_comp/",paste0(x,"_kgcorr.csv")))
  se_operon_corrs_df_m_a
}

)


