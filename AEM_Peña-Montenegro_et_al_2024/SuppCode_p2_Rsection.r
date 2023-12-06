#Following method at http://kembellab.ca/r-workshop/biodivR/SK_Biodiversity_R.html
library(picante)
library(reshape2)

setwd("/Users/tito-admin/Tito/JOYELABACKUP/SK_BACKUP/p22_Jupyter/Data/b-diversity/")

comm_longfmt <- read.csv("/Users/tito-admin/Tito/JOYELABACKUP/SK_BACKUP/p22_Jupyter/Data/Fig1_data_simple_absolute_melt.csv", header = TRUE, row.names = 1)
#Transforming into compact(wide) format
comm = dcast(comm_longfmt, sample_names~variable)
comm$sample_names <- c('BC_0', 'BC_1', 'BC_2', 'BC_3', 'BC_4', 'D_0', 'D_1', 'D_2', 'D_3', 'D_4', 'WAF_0', 'WAF_1A', 'WAF_1B', 'WAF_2', 'WAF_3', 'WAF_4A', 'WAF_4B', 'CEWAF_0', 'CEWAF_1A', 'CEWAF_1B', 'CEWAF_2', 'CEWAF_3' ,'CEWAF_4A' ,'CEWAF_4B' , 'CEWAF+N_0' ,'CEWAF+N_1' , 'CEWAF+N_4')

#setting row names
comm2 <- comm[,-1]
rownames(comm2) <- comm[,1]
comm_v0 = comm
comm = comm2


comm['A'] <- comm$Bacteroidetes + comm$Flavobacteriaceae
comm$Bacteroidetes <- NULL
comm$Flavobacteriaceae <- NULL
colnames(comm)[colnames(comm)=="A"] <- "Bacteroidetes"
comm['A'] <- comm$Pseudomonas + comm$Gammaproteobacteria
comm$Pseudomonas <- NULL
comm$Gammaproteobacteria <- NULL
colnames(comm)[colnames(comm)=="A"] <- "Gammaproteobacteria"

head(rownames(comm))

head(colnames(comm))

comm[1:5,1:5]

#check total abundance in each sample
head(apply(comm, 1, sum))

#Turn percent cover to relative abundace by diving each value by sample total abundance
comm <-decostand(comm, method="total")
#check total abundacne in each sample
head(apply(comm, 1, sum))

#look at the transformed data
comm[1:5,1:5]

#replace filename with file.choose() to open interactive window 
metadata <- read.csv('metadata_picante.csv', header=TRUE, row.names = 1)
#take a peek at the data
head(metadata)

phy <- read.tree('v3_1000_iterations.newick')
class(phy)

phy

plot(phy, cex=0.5)

#check for mismatches/missing species
combined <- match.phylo.comm(phy, comm)
#the resulting object is a list with $phy and $data elements. 
#Replace our original data with the sorted/matched data
phy <- combined$phy
comm <- combined$comm

#we should check whether our community data and metadata are in the same order
all.equal(rownames(comm), rownames(metadata))

#if sorting is needed
#metadata <- metadata[rownames(comm),]

## check later this if there is time 
#chisq.test(specnumber(comm)~metadata$Treatment)

# calculate Bray-Curtis distance among samples
comm.bc.dist <- vegdist(comm, method = "bray")
# cluster communities using average-linkage algorithm
comm.bc.clust <- hclust(comm.bc.dist, method = "average")
# plot cluster diagram
plot(comm.bc.clust, ylab = "Bray-Curtis dissimilarity")

svg(filename="FigS4G_Bray-Curtis_dissimilarity.svg")
plot(comm.bc.clust, ylab = "Bray-Curtis dissimilarity")
dev.off()

# The metaMDS function automatically transforms data and checks solution
# robustness
comm.bc.mds <- metaMDS(comm, dist = "bray")

# Assess goodness of ordination fit (stress plot)
stressplot(comm.bc.mds)

# plot site scores as text
ordiplot(comm.bc.mds, display = "sites", type = "text")

p1 = comm.bc.mds$points
p1 = as.data.frame(p1)
p1['Feature'] <- c('sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample','sample')
p2 = comm.bc.mds$species
p2 = as.data.frame(p2)
p2['Feature'] <- c('species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species','species')
nmds_data <- rbind(p1,p2) 
write.csv(nmds_data, file = "Fig3_nmds_data_for_plot.csv") 

# automated plotting of results - tries to eliminate overlapping labels
ordipointlabel(comm.bc.mds)

# plot Colwellia abundance. cex increases the size of bubbles.
ordisurf(comm.bc.mds, comm[, "Colwellia"], bubble = TRUE, main = "Colwellia abundance", 
    cex = 3)

# plot Colwellia abundance. cex increases the size of bubbles.
ordisurf(comm.bc.mds, comm[, "Marinobacter"], bubble = TRUE, main = "Marinobacter abundance", 
    cex = 3)



ordiplot(comm.bc.mds)
# calculate and plot environmental variable correlations with the axes use
# the subset of metadata that are environmental data
plot(envfit(comm.bc.mds, metadata[, 2:4]))

# The metaMDS function automatically transforms data and checks solution
# robustness
#comm.bc.pca <- metaMDS(comm, dist = "bray")

# First step is to calculate a distance matrix. 
# Here we use Bray-Curtis distance metric
comm.bc.vegdist <- vegdist(comm,  method = "bray")


# PCoA is not included in vegan. 
# We will use the ape package instead
library(ape)
PCOA <- pcoa(comm.bc.vegdist)

# plot the eigenvalues and interpret
#barplot(PCOA$values$Relative_eig[1:10])
PCOA$values$Relative_eig[1:10]

Y = comm
x = PCOA
plot.axes = c(1, 2)
pr.coo <- x$vectors
n <- nrow(Y)
points.stand <- scale(pr.coo[, plot.axes])
S <- cov(Y, points.stand)
U <- S %*% diag((x$values$Eigenvalues[plot.axes]/(n - 1))^(-0.5))
#We need is points.stand and U to export to be plotted in altair

# Write CSV in R
write.csv(points.stand, file = "Fig3_pca_samples.csv")
write.csv(U, file = "Fig3_pca_species.csv")

# convert phylogenety to a distance matrix
phy.dist <- cophenetic(phy)

# calculate ses.mpd
comm.sesmpd <- ses.mpd(comm, phy.dist, null.model = "taxa.labels", abundance.weighted = TRUE, runs = 999)
head(comm.sesmpd)
#If there is an error run 
#vec1 = colnames(comm)
#vec2 = colnames(phy.dist)
#setdiff(vec2,vec1)
#setdiff(vec1,vec2)

# calculate ses.mntd
comm.sesmntd <- ses.mntd(comm, phy.dist, null.model = "taxa.labels", abundance.weighted = TRUE, 
    runs = 999)
head(comm.sesmntd)


# compare ses.mpd between habitats
plot(comm.sesmpd$mpd.obs.z ~ metadata$Treatment, xlab = "Treatment", ylab = "SES(MPD)")
abline(h = 0, col = "gray")

# compare ses.mpd between habitats
plot(comm.sesmpd$mpd.obs.z ~ metadata$Time, xlab = "Time", ylab = "SES(MPD)")
abline(h = 0, col = "gray")

#t.test(comm.sesmpd$mpd.obs.z ~ metadata$habitat)
# Compute the analysis of variance
res.aov <- aov(comm.sesmpd$mpd.obs.z ~ metadata$Treatment + metadata$Time, data = comm)
# Summary of the analysis
summary(res.aov)

# compare ses.mntd between habitats
plot(comm.sesmntd$mntd.obs.z ~ metadata$Treatment, xlab = "Treatment", ylab = "SES(MNTD)")
abline(h = 0, col = "gray")

# compare ses.mntd between habitats
plot(comm.sesmntd$mntd.obs.z ~ metadata$Time, xlab = "Time", ylab = "SES(MNTD)")
abline(h = 0, col = "gray")

#t.test(comm.sesmntd$mntd.obs.z ~ metadata$habitat)
# Compute the analysis of variance
res.aov <- aov(comm.sesmntd$mntd.obs.z ~ metadata$Treatment + metadata$Time, data = comm)
# Summary of the analysis
summary(res.aov)

ordiplot(comm.bc.mds)

#TEST IF IT IS ROOTED
is.rooted(phy)

#rootedphy = root(phy, outgroup = 'Cenarchaeum',resolve.root = TRUE)

# Calculate Faith's PD
comm.pd <- pd(comm, phy)
head(comm.pd)

# Plot Faith's PD by habitat
boxplot(comm.pd$PD ~ metadata$Treatment, xlab = "Treatment", ylab = "Faith's PD")

# Test for PD differences among habitats
#t.test(comm.pd$PD ~ metadata$habitat)
# Compute the analysis of variance
res.aov <- aov(comm.pd$PD ~ metadata$Treatment + metadata$Time, data = comm)
# Summary of the analysis
summary(res.aov)

# Compare PD and species richness
plot(comm.pd$PD ~ comm.pd$SR, xlab = "Species richness", ylab = "Faith's PD")

# convert phylogenety to a distance matrix
phy.dist <- cophenetic(phy)

# calculate phylogenetic MNTD beta diversity
comm.mntd.dist <- comdistnt(comm, phy.dist, abundance.weighted = TRUE)
# calculate Mantel correlation for taxonomic Bray-Curtis vs. phylogenetic
# MNTD diversity
mantel(comm.bc.dist, comm.mntd.dist)

# calculate phylogenetic MPD beta diversity
comm.mpd.dist <- comdist(comm, phy.dist, abundance.weighted = TRUE)
# calculate Mantel correlation for taxonomic Bray-Curtis vs. phylogenetic
# MNTD diversity
#mantel(comm.bc.dist, comm.mntd.dist)

# NMDS ordination of phylogenetic distances - use monoMDS since we only have
# among-sample distances
comm.mntd.mds <- monoMDS(comm.mntd.dist)

# Assess goodness of ordination fit (stress plot)
stressplot(comm.mntd.mds)

# plot site scores as text
ordiplot(comm.mntd.mds, display = "sites", type = "text")

# automated plotting of results - tries to eliminate overlapping labels
ordipointlabel(comm.mntd.mds)

# Taxonomic (Bray-Curtis) dissimilarity explained
adonis(comm.bc.dist ~ Dispersant + Oil + Dispersant*Oil + Dispersant*Oil*Nutrients + Time + Dispersant*Time + Oil*Time + Dispersant*Oil*Time + Dispersant*Oil*Nutrients*Time, data = metadata)



# Phylogenetic MNTD dissimilarity explained
adonis(comm.mntd.dist ~ Dispersant + Oil + Dispersant*Oil + Dispersant*Oil*Nutrients + Time + Dispersant*Time + Oil*Time + Nutrients*Time + + Dispersant*Oil*Nutrients*Time, data = metadata)


# Phylogenetic MPD dissimilarity explained
adonis(comm.mpd.dist ~ Dispersant + Oil + Dispersant*Oil + Dispersant*Oil*Nutrients + Time + Dispersant*Time + Oil*Time + Nutrients*Time + + Dispersant*Oil*Nutrients*Time, data = metadata)




