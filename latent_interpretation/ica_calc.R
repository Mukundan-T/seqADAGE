library(stringr)
library(ica)

comp_files_all <- list.files('./compendia/', pattern = "*_log_counts_norm_01.csv")
comp_names_all <- str_extract(comp_files_all, "([a-zA-z0-9_]*)(.csv)", group = 1)

cor_files <- list.files('/work/gd134/ica_comp/', pattern = "*_log_counts_norm_01_ica.csv")
cor_names <- str_extract(cor_files, "([a-zA-z0-9_]*)(_ica.csv)", group = 1)

comp_files <- comp_files_all#[!(comp_names_all %in% cor_names)]
comp_names <- str_extract(comp_files, "([a-zA-z0-9_]*)(.csv)", group = 1)

l <- lapply(comp_names, function(x){
  print(x)
  c <- read.csv(paste0(paste0("./compendia/",x),".csv"), stringsAsFactors = F)
  c[is.na(c)] <- 0
  r <- c[,-1]
  rownames(r) <- paste0("Feature:",c$X)
  q <- qr(r)$rank - 1
  print(q)
  print(dim(t(r)))
  i <- ica(t(r),nc= (q-1)) #(qr(r)$rank - 1))
  write.csv(i$M, paste0("/work/gd134/ica_comp/",paste0(x,"_ica.csv")))
  x
})