library(dplyr)
library(devtools)
library(CMSITools)
library(batchCorr)  
library(ranger)
library(devtools)
library(doParallel)
library(BiocParallel)
library(MSnbase)
library(xcms)
library(tidyverse)
library(stringi)
register(bpstart(SnowParam(18))) 


rawPath <- "/scratch/users/gkreder/vitD/combined/"
inDir <- "/scratch/users/gkreder/vitD/XCMS_RT/RP_NEG/"
outDir <- "/scratch/users/gkreder/vitD/Batch/RP_NEG/"


xdata <- readRDS(paste0(inDir, "xdata_filled2.rds"))
LCMS_data <- CMSITools::getFiles(rawPath) %>% CMSITools::cleanFiles(c("blank", "cond", "SQCcond", "WASH", "iterative")) 
LCMS_data <- LCMS_data %>% CMSITools::getRP() %>% CMSITools::getNEG()
LCMS_data$group <- rep(NA,nrow(LCMS_data))
LCMS_data$group[LCMS_data$sample %>% grep("sQC",.)] <- "sQC"
LCMS_data$group[LCMS_data$sample %>% grep("ltQC",.)] <- "ltQC"
LCMS_data$group[-c(LCMS_data$sample %>% grep("sQC",.), LCMS_data$sample %>% grep("ltQC",.))] <- "sample"
LCMS_data_names <- LCMS_data %>% extractNames()
LCMS_data$sample_name <- str_remove(LCMS_data$filename, paste0(".", LCMS_data$fileformat))
merged <- merge(pData(xdata), LCMS_data, by = "sample_name", all.x = TRUE, sort = FALSE)





# FT_fill_imputation <- readRDS(paste0(inDir, "project_mode_FT_fill_imputation.rds"))
# xdata_filled <- readRDS(paste0(outDir, "xdata_filled2.rds"))
# FT_fill <- getTable(xdata_filled)
FT_NOFill <- readRDS(paste0(inDir, "XCMS_CorrectedGroupedNoFill.rds"))
FT_fill_imputation <- readRDS(paste0(inDir, "project_mode_FT_fill_imputation.rds"))


        
# peakIn <- peakInfo(PT = FT_fill_imputation, sep = '@', start = 1)
peakIn <- peakInfo(PT = FT_NOFill, sep = '@', start = 1)
alignBat <- alignBatches(peakInfo = peakIn, PeakTabNoFill = FT_NOFill, PeakTabFilled = FT_fill_imputation, batches = merged$batch, sampleGroups = merged$group, 
selectGroup = 'sQC')  


# alignBat <- alignBatches(peakInfo = peakIn, PeakTabNoFill = PTnofill, PeakTabFilled = PTfill, batches = meta$batch, sampleGroups = meta$grp, selectGroup = 'QC')

# --- correct drifts --- 
batch_num <- 0
correctedBatches <- list()
# batch code as in "B4W5"
for (batch_code in as.character(unique(LCMS_data$batch))){
    batch_num <-  batch_num + 1
    print(paste("Drift correction for batch", batch_num, ": ", batch_code))
    selected_batch <- getBatch(peakTable = FT_fill_imputation, meta = LCMS_data, batch = LCMS_data$batch, select = batch_code)
    # TODO: modelNames and G selection is the core optimization, see tutorial
    drift_correction <- correctDrift(peakTable = selected_batch$peakTable, injections = selected_batch$meta$injection, sampleGroups = 
                             selected_batch$meta$group, QCID = "sQC", G = seq(5,35, by = 3) )
    correctedBatches[[batch_num]] <- drift_correction
}

saveRDS(correctedBatches, file=paste0(outDir, "corrected.rds"))
peakTdata <- mergeBatches(correctedBatches)


# --- normalize samples --- 
PT_data_norm <- normalizeBatches(peakTableCorr = peakTdata$peakTableCorr, peakTableOrg = peakTdata$peakTableOrg, batches=LCMS_data$batch, sampleGroup = LCMS_data$group, refGroup = 'sQC', population='sample')

sum(is.na(PT_data_norm$peakTable))
sum(is.nan(PT_data_norm$peakTable))
print(paste("is.na: ", sum(is.na(PT_data_norm$peakTable))))
print(paste("is.nan: ", sum(is.nan(PT_data_norm$peakTable))))

# save(LCMS_data, PT_data_norm, file=paste(filepath, "_batchcorr_final.rda", sep = ""))
saveRDS(PT_data_norm, file = paste0(outDir, "batchCorrFinal.rds"))
saveRDS(LCMS_data, file = paste0(outDir, "LCMS_data.rds"))

# saveRDS()

