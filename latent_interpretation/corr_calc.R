library(stringr)

comp_files_all <- list.files('../models_for_corr/', pattern = "*_en_weights_da.csv")
comp_names_all <- str_extract(comp_files_all, "([a-zA-z0-9_]*)(.csv)", group = 1)

cor_files <- list.files('/work/gd134/corr_comp/', pattern = "*_en_weights_da_pcorr.csv")
cor_names <- str_extract(cor_files, "([a-zA-z0-9_]*)(_pcorr.csv)", group = 1)

comp_files <- comp_files_all#[!(comp_names_all %in% cor_names)]
comp_names <- str_extract(comp_files, "([a-zA-z0-9_:.]*)(.csv)", group = 1)



l <- lapply(comp_names, function(x){
  print(x)
  xclean <- gsub(":","",x)
  #print(xclean)
  c <- read.csv(paste0(paste0("../models_for_corr/",x),".csv"), stringsAsFactors = F, header = F)
  c[is.na(c)] <- 0
  #r <- c[,-1]
  #rownames(r) <- paste0("Feature:",c$X)
  #rownames(c) <- comp$X
  i <- cor(t(c))
  write.csv(i, paste0("/work/gd134/corr_comp/",paste0(xclean,"_pcorr.csv")))
  x
})
