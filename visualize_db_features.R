library(scatterplot3d)

png('naive.png')
scatterplot3d(read.table('naive.table'),highlight.3d=TRUE)
dev.off()

png('PCA.png')
scatterplot3d(read.table('PCA.table'),highlight.3d=TRUE)
dev.off()

