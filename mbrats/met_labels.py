# labels

#Nonenhancing tumor core (NETC; Label 1): All portions of tumor core without contrast enhancement that are enclosed by enhancing tumor (ET). It represents the bulk of the tumor, which is what is typically considered for surgical excision.
#Surrounding non-enhancing FLAIR hyperintensity (SNFH; Label 2): Peritumoral edematous and infiltrated tissue, defined by the abnormal hyperintense signal envelope on the T2 FLAIR volumes, which includes the infiltrative non enhancing tumor, as well as vasogenic edema in the peritumoral region. Non tumor related FLAIR signal abnormality such as prior infarcts or microvascular ischemic white matter changes are NOT included.
#Enhancing Tumor (ET; Label 3): All tumor portions with noticeable contrast enhancement on postcontrast T1-weighted images. Adjacent blood vessels, bleeding or intrinsic T1 hyperintensity are NOT included in this label.
#Resection Cavity (RC; Label 4): Delineates the resection of region within the brain in post-treatment cases.

BACKGROUND = 0 
NONENHANCING_TUMOR_CORE = 1
FLAIR_HYPERINTENSITY = 2
ENHANCING_TUMOR = 3
RESECTION_CAVITY = 4

# https://github.com/BraTS/BraTS_evaluation/blob/main/brats_evaluation/configs/config_mets.yaml
TUMOR_CORE_GROUP = [1, 3]
WHOLE_TUMOR_GROUP = [1, 2, 3] # kindof weird that resection cavity does not count (if recurrence)