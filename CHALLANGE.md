# BraTS 2026

## Data

### Rules
Participants are allowed to use only the provided BraTS 2026 dataset for training the model they are about to submit. Utilizing pre-trained model weights from other segmentation tasks is not allowed.

### Description
The dataset comprises multiparametric MRI (mpMRI) scans, which include the following series:

- pre-contrast T1-weighted (T1W)
- post-contrast T1-weighted (T1C)
- T2-weighted (T2W)
- T2-weighted Fluid Attenuated Inversion Recovery (FLAIR)

In 2025, T2W became non-mandatory in BraTS-METS. Some cases have native T2, some have synthetic T2, some don't have T2. All imaging volumes were segmented using the STAPLE fusion of different brain metastases segmentation algorithms

### Labels
For BraTS 2026 Brain Metastases, the following 4-label system is used:

Nonenhancing tumor core (NETC; Label 1): All portions of tumor core without contrast enhancement that are enclosed by enhancing tumor (ET). It represents the bulk of the tumor, which is what is typically considered for surgical excision.

Surrounding non-enhancing FLAIR hyperintensity (SNFH; Label 2): Peritumoral edematous and infiltrated tissue, defined by the abnormal hyperintense signal envelope on the T2 FLAIR volumes, which includes the infiltrative non enhancing tumor, as well as vasogenic edema in the peritumoral region. Non tumor related FLAIR signal abnormality such as prior infarcts or microvascular ischemic white matter changes are NOT included.

Enhancing Tumor (ET; Label 3): All tumor portions with noticeable contrast enhancement on postcontrast T1-weighted images. Adjacent blood vessels, bleeding or intrinsic T1 hyperintensity are NOT included in this label.

Resection Cavity (RC; Label 4): Delineates the resection of region within the brain in post-treatment cases.

For 2026, we will add a detection leaderboard, with the intention of promoting algorithms sensitive in detecting lesions. This is clinically relevant to cases of small (<27mm^3) lesions that either need to be counted as separate entities, or that they need to be quantified separately.

### Image registration:

The BraTS 2025 Metastases dataset consists of a mix of cases in native space, co-registered to T1C 1mm^3 and cases registered in SRI24 space. All cases provided by Ulm University, UCSF and UCSD are in native space, totaling 1268 cases. In contrast, the remaining cases are registered in SRI24 space, amounting to 328.

Registering neuroimaging cases in a common space, such as SRI24, allows for a consistent anatomical reference that facilitates comparisons across different subjects, studies, and datasets. However, it is more natural for radiologists to review cases in their native space, as interpolation can distort images and obscure small lesions.


## Evaluation

In terms of segmentation evaluation metrics, we use the following subject-wise metrics:
i) Dice Similarity Coefficient (DSC), which is commonly used in the assessment of segmentation performance
ii) Normalized Surface Distance (NSD), which introduces a tolerance parameter and is complementary to traditional metrics such as DSC.

In terms of lesion detection evaluation, we use the following lesion-wise metrics:
i) F1 score – the harmonic mean of precision and recall, to determine whether an algorithm has the tendency to over- or undersegment

We will apply the segmentation evaluation metrics only for lesions larger than 27 mm3.

Ranking details:
We will follow the DELPHI-based recommendations for image analysis validation [1,2], incorporating i) algorithmic ranking, and ii) statistical significance testing. For ranking of multidimensional outcomes (or metrics), for each team, we will compute the summation of their ranks across the average of the metrics described above as a univariate overall summary measure. This measure will decide the overall ranking for each specific team. All teams will then be placed in a ranked order and their average rankings will be randomly permuted (i.e., 500,000 permutations), in a pair-wise manner. Corresponding pairwise p-values will be computed to determine the pair-wise statistical significance and report actual differences between the ordered ranked approaches. These p-values will be reported in an upper triangular matrix revealing the statistical insignificance of potential teams that will be grouped together in tiers and the significant superiority among others that we will clearly indicate. This is an evolved version of the systematic ranking that has been used on previous years for BraTS and other challenges, and will be packaged & distributed as an independent tool allowing reproducibility and use in other challenges.

For the cases in which the algorithm fails to produce a result metric for a specific test case, there will be no penalties, i.e. the metric won't be set to its worst possible value (e.g., 0 for the DSC and the NSD).
