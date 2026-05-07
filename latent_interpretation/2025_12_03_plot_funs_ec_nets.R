
##########################################
corr_plot_scaled <- function(t1, tlpos){
  corrplot(t1, 
           is.corr = F,
           #title = neb_hits_cat[x], #paste(names(neb_hits)[x], neb_hits[x]),
           col.lim = c(min(t1), max(t1)),
           method = 'color', 
           order = 'hclust', #'original', 
           tl.pos = tlpos, #'n', # ld
           type = "lower",
           tl.cex = .5,
           tl.col = "black",
           cl.cex = .3,
           title = "gene expression",
           mar = c(0,0,4,0),
           diag = FALSE)

}

#########################################
corr_compare <- function(genes, ec_anns, tlpos = "n"){
  
  par(mfrow = c(1,2))
  
  # correlations in the gene expression compendium
  t1 <- ec_comp_corr[genes,genes]
  t1[t1 == 1] <- 0

  gnames <- ec_anns[genes,'Common.Name']
  rownames(t1) <- gnames
  colnames(t1) <- gnames
 
  
  corr_plot_scaled(t1, tlpos)
  
  # correlations in the model
  t1 <- ec_model_corrs[[1]][genes,genes]
  t1[t1 == 1] <- 0

  gnames <- ec_anns[genes,'Common.Name']
  rownames(t1) <- gnames
  colnames(t1) <- gnames

  corr_plot_scaled(t1, tlpos)
}

###########################################
for_cytoscape <- function(ec_model_corrs, fname, t_low=0){
  ec_model_corrs_m <- melt(ec_model_corrs)
  m <- as.matrix(ec_model_corrs[[1]])
  m[lower.tri(m)] <- 0
  t <- triu(m)
  
  ec_model_corrs_m <- melt(m)
  ec_model_corrs_m_u <- ec_model_corrs_m[abs(ec_model_corrs_m$value) > t_low & 
                                           abs(ec_model_corrs_m$value) < 1,]
  colnames(ec_model_corrs_m_u) <- c("gene1","gene2","corr")
  
  write.csv(ec_model_corrs_m_u, fname)
  ec_model_corrs_m_u
}


###########################################

plot_net <- function(de_genes, edges, ec_anns, thresh = 0.15, save = F){ #
  de_edges <- edges[edges$weight > thresh & (edges$from %in% de_genes | edges$to %in% de_genes),]
  net_genes <- union(union(de_genes, de_edges$from), de_edges$to)
  net_edges <- de_edges
  net_edges <- net_edges[!is.na(net_edges$from),]
  net_edges <- net_edges[!net_edges$from == net_edges$to,]
  #rownames(ec_anns) <- ec_anns$Accession.1
  
  my_palette <- colorRampPalette(c("blue", "white", "red", "red3"))(n = 5)
  names(my_palette) <- unique(ec_anns$Characterization)[c(2,4,3,1,5)]
  
  g <- graph.data.frame(net_edges[,c(1,2,3)], directed=F)
  E(g)$width <-  ((net_edges$weight - min(net_edges$weight) ) / 
                    (max(net_edges$weight) - min(net_edges$weight))) * 10  #(net_edges$weight * 20)-15
  
  
  
  bd  <- sapply(vertex_attr(g, 'name'), function(x){
    if(x %in% de_genes){
      'black'}else{NA}
    })
  
  
  
  node.colors <- ec_anns[vertex_attr(g, 'name'), "Characterization"]
  V(g)$color <- my_palette[node.colors]
  
  gnames <- ec_anns[vertex_attr(g, 'name'),'Common.Name']
  gnames <- str_sub(gnames, "_", "\n")
  
  ceb <- cluster_edge_betweenness(g) 
  name_temp <- vertex_attr(g, 'name')
  vertex_attr(g, 'name') <- str_replace_all(name_temp, "-","\n")
  vertex_attr(g, 'outline') <- bd
  
  g_simple <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE,
                       edge.attr.comb = "max")
  
  if( save ){
    png(paste0(paste(de_genes, collapse = "_"),".png"),
        res= 300,
        height = 8,
        width = 8,
        units = "in")
    plot(g, 
         #vertex.label = gnames, 
         col = vertex_attr(g, 'color'), 
         vertex.label.color = "black",
         vertex.frame.color = vertex_attr(g, 'outline'), 
         vertex.frame.width = 5, 
         vertex.size = 20,
         mark.coly = brewer.pal(n = 8, name = "Set3"))
    dev.off()
    
    
  }
    plot(g_simple, 
         #vertex.label = gnames, 
         col = vertex_attr(g, 'color'), 
         vertex.label.color = "black",
         vertex.frame.color = vertex_attr(g, 'outline'), 
         vertex.frame.width = 5,
         vertex.size = 20,
         mark.coly = brewer.pal(n = 8, name = "Set3"))
  
}

############################################################
plot_net_mat <- function(de_genes, edges, ec_anns, thresh = 0.15, save=F){
  
  de_edges <- edges[edges$weight > thresh & (edges$from %in% de_genes | edges$to %in% de_genes),]
  net_genes <- union(union(de_genes, de_edges$from), de_edges$to)
  net_edges <- de_edges
  net_edges <- net_edges[!is.na(net_edges$from),]
  net_edges <- net_edges[!net_edges$from == net_edges$to,]
  #rownames(ec_anns) <- ec_anns$Accession.1
    

    
    g <- graph.data.frame(net_edges[,c(1,2,3)], directed=F)
    
    
    gnames <- ec_anns[vertex_attr(g, 'name'),'Common.Name']
    
    
    g_temp <- vertex_attr(g, 'name')
    
    t1 <- ec_comp_corr[g_temp,g_temp]
    t1[t1 == 1] <- 0
    rownames(t1) <- g_temp
    colnames(t1) <- g_temp
    
    t2 <- ec_model_corrs[[1]][g_temp,g_temp]
    t2[t2 == 1] <- 0
    rownames(t2) <- g_temp
    colnames(t2) <- g_temp
    if (save){
       png(paste0(paste(de_genes, collapse = "_"),"_corrplot.png"),
           res= 300,
           height = 4,
           width = 8,
           units = "in")
      
      par(mfrow = c(1,2))
      
      corr_plot_scaled(t1, 'ld')
      corr_plot_scaled(t2, 'ld')
      dev.off()
      
    }

    par(mfrow = c(1,2))
    corr_plot_scaled(t1, 'ld')
    corr_plot_scaled(t2, 'ld')
    par(mfrow = c(1,1))

}
