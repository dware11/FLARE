export type CancerType = 'brain' | 'breast'

export type CreateCaseRequest = {
  location: string
  cancerType: CancerType
  patient: {
    firstName: string
    lastName: string
    medicalId: string
    dob: string
  }
}

export type CaseResult = {
  caseId: string
  classification: 'Normal' | 'Benign' | 'Malignant'
  confidence: number // 0..1
  localizationImageUrl?: string // Grad-CAM/seg overlay image URL
  createdAt: string
}
