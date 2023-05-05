library(DESeq2)

##===============================================================
##                              RNA                             =
##===============================================================
list_count = list()
list_dir = list.dirs("input/RNA",recursive = F)
for(sample in list_dir){
  file = list.files(sample,pattern = "rna_seq.augmented_star_gene_counts")
  count = read.table(paste0(sample,"/",file),sep="\t",header = T)
 # count = count[5:100,]
  count = count[5:nrow(count),]
  gene_name = count["gene_name"]
  count = count["unstranded"]
  sample_name = gsub("^.*/","",sample)
  list_count[sample_name] = count 
  print(sample_name)
 
}

rna_count = Reduce(cbind,list_count)
rownames(rna_count) = gene_name$gene_name 
colnames(rna_count) = names(list_count)

metadata = read.csv("/gpfs/home/chenz05/DL2023/input/metadata/clinical.csv",header = T)
rownames(metadata) =  metadata$case_submitter_id
metadata = metadata[colnames(rna_count) , ]
metadata$label =  metadata$years_to_dead
metadata$label[metadata$label != "alive"] = "death" 
dds <- DESeqDataSetFromMatrix(countData = rna_count, colData = metadata, design = ~ 1)
dds = DESeqDataSet(dds,design= ~ label )
deseq=DESeq(dds)
normalized_counts=counts(deseq,normalized = T)
res = results(deseq)
write.csv(res,"/gpfs/home/chenz05/DL2023/input/metadata/RNA_sigtable.csv",quote = F)

sig_res = subset(res, abs(log2FoldChange) > 1 & padj < 0.05)
nc = normalized_counts[rownames(sig_res),]
nc = (nc - apply(nc,1,mean)) / apply(nc,1,sd)
write.csv(nc,"/gpfs/home/chenz05/DL2023/input/metadata/RNA_TOP_statistics.csv",quote = F)

