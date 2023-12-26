import numpy as np
import nibabel as nib

#func_img = nib.load('/Volumes/AUERBACHLAB/Columbia/HCP_BANDA_1.0_Data/NDA_FINAL/derivatives/BANDA003_MR/func/faces/BANDA003_MR_task-face_level2.gfeat/cope7.feat/stats/tstat1.nii.gz')
#labels = nib.load('/Volumes/AUERBACHLAB/Columbia/General/Tapasi_Brahma/Atlas/HCPex/HCPex_FSLspace.nii.gz')

def pull_roi(func_path, label_path, thresh_value):
	func_img = nib.load(func_path)
	labels = nib.load(label_path)
	func_data = func_img.get_fdata()
	labels_data =  labels.get_fdata()

	#  take the data just from the ROI with a certain number label
	func_roi = func_data[labels_data == thresh_value]

	return(np.mean(func_roi))



'''
Lines of code we used to align the HCPex data to FSL MNI 2mm space

flirt -in mni_icbm152_t1_tal_nlin_asym_09c_brain.nii.gz -ref /Users/eastvillage/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz -omat icbm2fsl.mat -dof 6
flirt -ref /Users/eastvillage/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz  -in HCPex.nii.gz -applyxfm -init icbm2fsl.mat -out HCPex_FSLspace.nii.gz -interp nearestneighbour
'''