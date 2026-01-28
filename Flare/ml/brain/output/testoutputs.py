from outputs import ModalityResult, fuse_results

#Fake CT Output 
ct= ModalityResult(
    modality="CT", 
    prediction=0.76,
    label="tumor", 
    model_version="ct_v1"
)

#Fake MRI Output
mri= ModalityResult(
    modality="MRI",
    prediction=0.91,
    label="tumor",
    model_version="ct_v1"
)

# Fuse both using unequal weights 
fusion = fuse_results(ct=ct, mri=mri, w_ct=0.3, w_mri=0.7)

print("Fused result: ")
print(fusion)