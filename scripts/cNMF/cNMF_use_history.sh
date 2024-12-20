#--------------------------------------------------------------------
## Example taken from github: https://github.com/codyheiser/cnmf.git 
#--------------------------------------------------------------------
cnmf prepare ./example_data/counts_prefiltered.txt \
--output-dir ./example_data \
--name example_cNMF \
-k 5 6 7 8 9 10 11 12 13 \
-n 30 \
-j 1

cnmf factorize \
--output-dir ./example_data \
--name example_cNMF \
--worker-index 0 

#----------------
## First Attmept
#----------------
cnmf prepare /mnt/md0/cody/scRNA_anndatas/VUMC_COMBINED.h5ad \
--output-dir /home/james/data/cNMF_out \
--name test_2_cNMF \
-k 5 6 7 8 9 10 11 12 13 \
-n 30 \
-j 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name test_2_cNMF \
--worker-index 0 

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name test_2_cNMF \
-k 5 6 7 8 9 10 11 12 13

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name test_2_cNMF

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name test_2_cNMF \
--local-density-threshold 0.05 \
--auto-k 
#--cleanup ## DO NOT USE ## Ignore the exmaple on github

#-------------------------------------------------------
## Second attmept to reproduce the actuall useful usage
#-------------------------------------------------------
cnmf prepare /mnt/md0/cody/scRNA_anndatas/VUMC_COMBINED.h5ad \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF \
-k 24 26 28 30 32 34 36  \
-n 30 \
-j 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name VUMC_test_cNMF \
--worker-index 0 

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name VUMC_test_cNMF \
-k 24 26 28 30 32 34 36 

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF 
# The png file confirms that -k 30 is indeed a local maximum for stability

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF \
--local-density-threshold 0.05 -k 30

# -------------------------------------------------------------------------------------
# # Most similar to Cody's results and pulled form his graphs to replicate his results
# -------------------------------------------------------------------------------------
# code constructed based on the plot locacted in the below directory
# directory: /mnt/md0/cody/scRNA_NMF/VUMC_NMF
# file: VUMC_NMF.k_selection.png
# full path: /mnt/md0/cody/scRNA_NMF/VUMC_NMF/ VUMC_NMF.k_selection.png
# this was done in colaboration with the directory below:
# /mnt/md0/cody/scRNA_NMF/VUMC_NMF/cnmf_tmp 
# this directory holds allow of the merged spectra files for each K ran

cnmf prepare /mnt/md0/cody/scRNA_anndatas/VUMC_COMBINED.h5ad \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF \
-k 8 9 10 11 12 13 14 15 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 \
-n 30 \
-j 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name VUMC_test_cNMF \
--worker-index 0

nmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF 
# The png file confirms that -k 30 is indeed a local maximum for stability

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF \
--local-density-threshold 0.05 \
-k 30

#------------------------------------------
## Using this code on new data (wrong data)
#-------------------------------------------
cnmf prepare /home/james/data/immune_exclusion_data/combined_all4_tcell_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell \
-k 8 9 10 11 12 13 14 15 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 \
-n 30 \
-j 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name all4_tcell \
--worker-index 0

nmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name VUMC_test_cNMF 
# The png file confirms that -k 30 is indeed a local maximum for stability

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell \
--auto-k \
--local-density-threshold 0.05

#-----------------------------------------------------------------------
## File generation is automatic when the --cleanup parameter is removed
#-----------------------------------------------------------------------
## Files i need after NMF that are not automatically created
## Each of these files uses the 30 factorization replicates for k = 30
VUMC_NMF.gene_spectra_score.k_30.dt_0_1.txt
    # neg and postive spectra scores for all genes
VUMC_NMF.gene_spectra_tpm.k_30.dt_0_1.txt
    # pos tpm spectra scores for all genes
VUMC_NMF.spectra.k_30.dt_0_1.consensus.txt
    # spectra score for overdispersed genes identified through NMF but no the same values as from the other spectra file
VUMC_NMF.usages.k_30.dt_0_1.consensus.txt
    # usage scores by cell id (barcode) 

#-------------------------------------------
## All wrong data 
#-------------------------------------------
"""
cnmf prepare /home/james/data/immune_exclusion_data/combined_all4_tcell_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell \
-k 8 9 10 11 12 13 14 15 16 17 \
-n 30 \
-j 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name all4_tcell \
--worker-index 0

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell 
# The png file confirms that -k 30 is indeed a local maximum for stability

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell \
--local-density-threshold 0.05 -k 25 30 35
"""
#-------------------------------------------
## WRONG DATA AGAIN
#-------------------------------------------
"""
cnmf prepare /home/james/data/immune_exclusion_data/combined_all4_tcell_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell_2 \
-k 11 14 16 20 22 25 27 30 33 35 \
-n 30 \
-j 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name all4_tcell_2 \
--worker-index 0

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name all4_tcell_2 \
-k 11 14 16 20 22 25 27 30 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell_2 

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name all4_tcell \
--auto-k \
--local-density-threshold 0.05
"""
#-------------------------------------------
## All new snRNA data (NOT DOWNSAMPLED) 
#-------------------------------------------
#
cnmf prepare /home/james/data/immune_exclusion_data/outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name outer_combined_all4_nmf \
-k 11 14 16 20 22 25 27 30 33 35 \
-n 30 \
-j 1

# Factorization required the introduction of parrell process due to
# the new data being much larger (turns out to need to be down sampled)
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name outer_combined_all4_nmf \
--n-jobs 4 \
--worker-index 0

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name outer_combined_all4_nmf \
--n-jobs 4 \
--worker-index 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name outer_combined_all4_nmf \
--n-jobs 4 \
--worker-index 2

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name outer_combined_all4_nmf \
--n-jobs 4 \
--worker-index 3

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name outer_combined_all4_nmf \
-k 11 14 16 20 22 25 27 30 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name outer_combined_all4_nmf 

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name outer_combined_all4_nmf \
-k 16 25 30 \
--local-density-threshold 0.05

# See what the systems thinks the best k is 
# best K was determined to be 16 though this
cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name outer_combined_all4_nmf \
--auto-k \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the down sampled immune_exclusion data
#---------------------------------------------------------
# method = proportional downsample
cnmf prepare /home/james/data/immune_exclusion_data/ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name ds_immun_excl_nmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization is back down to one processor now that we have down sampled
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name  ds_immun_excl_nmf \
--n-jobs 1 \
--worker-index 0

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name ds_immun_excl_nmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name ds_immun_excl_nmf 

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name ds_immun_excl_nmf \
-k 25 30 \
--local-density-threshold 0.05

# See what the systems thinks the best k is
cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name ds_immun_excl_nmf \
--auto-k \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the down sampled immune_exclusion data
#---------------------------------------------------------
# method = proportional downsample with a 100 count filter
# any cell type that had less than 100 obs was removed
cnmf prepare /home/james/data/immune_exclusion_data/above100_ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name above100_ds_immmun_excl \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization is back down to one processor now that we have down sampled
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name  above100_ds_immmun_excl \
--n-jobs 1 \
--worker-index 0

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name above100_ds_immmun_excl \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name above100_ds_immmun_excl

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name above100_ds_immmun_excl \
-k 22 \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the down sampled immune_exclusion data
#---------------------------------------------------------
# method = downsample with a 2000 target of observations
# all cell types greater than 2000 obs were reduced to 2000
# all cell types less than 2000 obs were kept the same
cnmf prepare /home/james/data/immune_exclusion_data/ds_2000_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name count_2000_ds_immune_excl \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization is back down to one processor now that we have down sampled
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name  count_2000_ds_immune_excl \
--n-jobs 1 \
--worker-index 0

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name count_2000_ds_immune_excl \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name count_2000_ds_immune_excl

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name count_2000_ds_immune_excl \
-k 22 30 \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the WRONG density down sampled immune_exclusion data
#---------------------------------------------------------
# method = density dependent down sampling designed by james
cnmf prepare /home/james/data/immune_exclusion_data/density_ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name density_ds_immune_excl \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization is back down to one processor now that we have down sampled
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name  density_ds_immune_excl \
--n-jobs 1 \
--worker-index 0

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name density_ds_immune_excl \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name density_ds_immune_excl

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name density_ds_immune_excl \
-k 33 \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the density dependent down sampled immune_exclusion data
#---------------------------------------------------------
# method = density dependent down sample using the pca info
cnmf prepare /home/james/data/immune_exclusion_data/pca_density_ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name pca_density_ds_cnmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name pca_density_ds_cnmf \
--n-jobs 4 \
--worker-index 0

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name pca_density_ds_cnmf \
--n-jobs 4 \
--worker-index 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name pca_density_ds_cnmf \
--n-jobs 4 \
--worker-index 2

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name pca_density_ds_cnmf \
--n-jobs 4 \
--worker-index 3

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name pca_density_ds_cnmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name pca_density_ds_cnmf

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name pca_density_ds_cnmf \
-k 27 \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the density dependent down sampled immune_exclusion data
#---------------------------------------------------------
# method = density dependent down sample using the pca info
# in addition the rare cell counts were kept the same meaning
# if the cell type count was below 1% of the total cell count
# it was kept
cnmf prepare /home/james/data/immune_exclusion_data/rare_pca_density_ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name rare_pca_density_ds_cnmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

cnmf prepare /home/james/data/immune_exclusion_data/rare_pca_density_ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name rare_pca_density_ds_cnmf \
-k 34 36 37 38 39 40 \
-n 30 \
-j 1

# Factorization
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name rare_pca_density_ds_cnmf \
--n-jobs 1 \
--worker-index 0

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name rare_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name rare_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 2

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name rare_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 3

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name rare_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 4

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name rare_pca_density_ds_cnmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name rare_pca_density_ds_cnmf

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name rare_pca_density_ds_cnmf \
-k 25 \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the density dependent down sampled immune_exclusion data
#---------------------------------------------------------
# method = density dependent down sample using the pca info
# in addition the rare cell counts were removed by the same rule of 1%
cnmf prepare /home/james/data/immune_exclusion_data/filter_pca_density_ds_outer_combined_all4_dat.h5ad \
--output-dir /home/james/data/cNMF_out \
--name filter_pca_density_ds_cnmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization 
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name filter_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 0

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name filter_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 1

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name filter_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 2

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name filter_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 3

cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name filter_pca_density_ds_cnmf \
--n-jobs 5 \
--worker-index 4

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name filter_pca_density_ds_cnmf \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name filter_pca_density_ds_cnmf

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name filter_pca_density_ds_cnmf \
-k 22 \
--local-density-threshold 0.05

#---------------------------------------------------------
## Applying cNMF to the one individual sample
#---------------------------------------------------------
# method = density dependent down sample using the pca info
cnmf prepare /home/james/data/immune_exclusion_data/10096_s1.h5ad \
--output-dir /home/james/data/cNMF_out \
--name sample_10096_s1_cNMF \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35 \
-n 30 \
-j 1

# Factorization is back down to one processor now that we have down sampled
cnmf factorize \
--output-dir /home/james/data/cNMF_out/ \
--name  sample_10096_s1_cNMF \
--n-jobs 1 \
--worker-index 0

cnmf combine \
--output-dir /home/james/data/cNMF_out/ \
--name sample_10096_s1_cNMF \
-k 8 9 10 11 14 16 18 19 20 22 25 26 27 28 30 31 33 35

cnmf k_selection_plot \
--output-dir /home/james/data/cNMF_out \
--name sample_10096_s1_cNMF

cnmf consensus \
--output-dir /home/james/data/cNMF_out \
--name sample_10096_s1_cNMF \
-k 16 \
--local-density-threshold 0.05

#########################################
## useing for each indivdual sample after the automation
#########################################
cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_01_s1_sample_cNMF \
-k 17 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_01_s2_sample_cNMF \
-k 5 15 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_01_s3_sample_cNMF \
-k 15 19 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_01_s4_sample_cNMF \
-k 23 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_02_s1_sample_cNMF \
-k 19 23 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_02_s2_sample_cNMF \
-k 17 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_02_s3_sample_cNMF \
-k 17 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10180_02_s4_sample_cNMF \
-k 13 27 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_9142_s1_sample_cNMF \
-k 11 19 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_9142_s2_sample_cNMF \
-k 17 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10096_s1_sample_cNMF \
-k 15 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10096_s2_sample_cNMF \
-k 15 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10096_s3_sample_cNMF \
-k 11 15 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10096_s4_sample_cNMF \
-k 13 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10284_s1_sample_cNMF \
-k 15 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10284_s2_sample_cNMF \
-k 9 21 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10284_s3_sample_cNMF \
-k 19 \
--local-density-threshold 0.05

cnmf consensus \
--output-dir /home/james/data/cNMF_out/per_sample_cnmf \
--name sep_10284_s4_sample_cNMF \
-k 13 \
--local-density-threshold 0.05
