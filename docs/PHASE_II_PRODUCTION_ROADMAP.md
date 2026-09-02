# FLARE — Phase II Production-Readiness Roadmap

**Status:** Proposed team direction after Phase I  
**Goal:** Move from a working research prototype toward an externally validated, pilot-ready clinical decision-support system without overstating clinical or regulatory readiness.

## Recommended Direction: Staged Hybrid

FLARE should pursue a **staged hybrid research + product path**.

### Track A — Research validation

Strengthen the evidence behind the current CT, MRI, segmentation, explainability, and fusion pipelines so the team can support publication-quality claims.

### Track B — Pilot readiness

Harden the application, data governance, security, deployment, clinical workflow, and documentation required for a controlled pilot or external research collaboration.

The two tracks support each other. Research validation establishes whether the models deserve trust; product engineering determines whether the system can operate safely and reproducibly outside the development environment.

---

# 1. Production Readiness Is More Than Model Accuracy

A model with strong test metrics is not automatically a deployable medical product. FLARE must mature across six layers.

## Layer 1 — Scientific validity

Required work:

- enforce patient-level train/validation/test separation,
- eliminate or quantify data leakage risks,
- harmonize MRI acquisition protocols,
- evaluate against external institutions/datasets,
- acquire paired healthy CT/MRI controls,
- report sensitivity, specificity, ROC-AUC, precision, recall, calibration and confidence intervals,
- benchmark fusion against CT-only and MRI-only baselines,
- compare the current weighted late-fusion method against at least one alternative,
- perform subgroup/error analysis,
- document every model checkpoint, preprocessing version, dataset split and threshold.

**Exit criterion:** Another technical team can reproduce the experiment and understand exactly what evidence supports every published claim.

## Layer 2 — Clinical validity and usefulness

Required work:

- define the exact intended user,
- define the exact intended use,
- identify where FLARE enters the radiologist workflow,
- determine what result is actionable versus informational,
- validate what clinicians actually need on the result screen,
- test how uncertainty/disagreement should be displayed,
- identify clinically unacceptable failure modes,
- define when FLARE must abstain or escalate to manual review,
- conduct structured clinician usability/human-factors feedback.

**Exit criterion:** The team can explain not only whether the AI predicts correctly, but how the output would safely assist a real clinician.

## Layer 3 — Data governance and research approvals

Before accepting real clinical data, determine:

- whether the activity requires IRB review or an exemption determination,
- whether a Data Use Agreement is required,
- who is legally permitted to receive and retain the data,
- whether data contains PHI/ePHI,
- what consent or waiver terms apply,
- how DICOM metadata is de-identified,
- how burned-in pixel identifiers are detected and removed,
- how identifiers, dates, UIDs and private DICOM attributes are handled,
- data-retention and deletion rules,
- access-control requirements,
- whether re-identification keys exist and who controls them.

Useful current references:

- HHS HIPAA de-identification guidance: Expert Determination and Safe Harbor methods.
- DICOM PS3.15 Attribute Confidentiality Profiles for image/dataset de-identification.
- The university IRB should make the institutional determination for actual planned research activity.

**Exit criterion:** The team has written institutional approval and data-handling rules before private clinical data is transferred.

## Layer 4 — Secure product architecture

Current Phase I architecture is a strong prototype foundation, but a pilot requires explicit enforcement and operational validation.

Required work:

- validate JWTs on protected backend endpoints,
- enforce role-based authorization server-side,
- implement least-privilege access,
- require HTTPS/TLS everywhere outside local development,
- encrypt sensitive data at rest,
- move secrets to managed secret storage,
- protect model checkpoints from unauthorized modification,
- maintain durable audit events for login, scan access, inference, review and export,
- implement retention/deletion workflows,
- implement backup and restore testing,
- add rate limiting and abuse protection,
- validate all upload formats and sizes,
- test malicious/corrupted DICOM and image inputs,
- run dependency and vulnerability scanning,
- establish incident-response ownership.

**Exit criterion:** Security controls are enforced by the deployed system, not merely described in the report.

## Layer 5 — Reliable engineering and MLOps

Required work:

- pin software/package versions,
- eliminate duplicate or obsolete frontend/backend paths,
- define one supported deployment path,
- containerize reproducibly,
- add CI tests for API contracts,
- add model smoke tests,
- add end-to-end UI/API tests,
- maintain model registry/version identifiers,
- log which model version produced each clinical result,
- log preprocessing and threshold versions,
- monitor latency, failures and model drift,
- define rollback procedures,
- establish staging and production environments,
- move long-lived inference from demo tunneling toward controlled infrastructure.

**Exit criterion:** A new authorized engineer can deploy the documented version without relying on undocumented local knowledge.

## Layer 6 — Regulatory and quality strategy

FLARE should obtain professional regulatory guidance before claiming clinical deployment readiness.

The team must define:

- intended use statement,
- intended user,
- clinical claims,
- whether the software function is a regulated medical device function,
- likely regulatory pathway if commercialization is pursued,
- required validation evidence,
- software lifecycle documentation,
- risk-management process,
- cybersecurity documentation,
- design/change-control process,
- post-deployment model monitoring strategy.

Current FDA resources to review include:

- 2026 Clinical Decision Support Software final guidance,
- FDA digital-health guidance catalog,
- 2025 AI-enabled device software Predetermined Change Control Plan guidance,
- current medical-device cybersecurity guidance.

**Exit criterion:** Regulatory counsel/qualified advisor has reviewed the intended use and the team understands the evidence and quality-system obligations for the selected path.

---

# 2. Proposed Phase II Workstreams

| Workstream | Core question | Suggested owner role | External help needed |
| --- | --- | --- | --- |
| Model validation | Do the models generalize beyond current datasets? | Research/ML lead | ML researcher, radiologist |
| Fusion research | Does multimodal fusion add value over strong single-modality baselines? | Research lead | Imaging researcher, statistician |
| Clinical workflow | Is the interface useful and safe for actual radiology workflow? | Product/UI lead | Radiologist, imaging technologist |
| Data governance | Can we legally and ethically receive the data we want? | Research/data governance lead | IRB, privacy/compliance expert |
| Security | Are controls actually enforced end-to-end? | Security/technical lead | Healthcare security engineer |
| Infrastructure | Can the system be deployed reproducibly and monitored? | Deployment lead | Cloud/MLOps engineer |
| Regulatory/product | What path would turn this into a pilot/product? | Project/product lead | FDA/regulatory/product mentor |

---

# 3. Priority Order

## Priority 0 — Freeze Phase I

- Tag/branch the demonstrated Phase I baseline.
- Ensure README, report and code describe the same implemented system.
- Record current model checkpoints and validated metrics.
- Mark unsupported/future features explicitly.

## Priority 1 — Fix evidence before adding features

- patient-level split verification,
- MRI protocol/domain harmonization,
- paired healthy controls,
- reproducible evaluation scripts,
- fusion baseline comparison.

## Priority 2 — Get clinical feedback

- result screen,
- Grad-CAM usefulness,
- segmentation usefulness,
- uncertainty presentation,
- review/approval workflow,
- geo/outbreak feature validity.

## Priority 3 — Establish data/compliance path

Do not accept private clinical data before the university/partner establishes the correct IRB, DUA, privacy, security and de-identification process.

## Priority 4 — Pilot architecture

- controlled cloud/HPC backend,
- secure database/storage,
- server-side authorization,
- audit trails,
- CI/CD,
- monitoring,
- backups,
- documented deployment.

## Priority 5 — External pilot proposal

Only after the preceding milestones should the team propose a limited research/clinical pilot with defined success criteria.

---

# 4. Suggested 90-Day Outcome

By the end of the next 90-day sprint, FLARE should be able to show:

1. a frozen, reproducible Phase I baseline,
2. a clean technical architecture and model card for each implemented pipeline,
3. patient-stratified evaluation where datasets permit,
4. a documented MRI domain-shift analysis,
5. a fusion comparison using clearly defined baselines,
6. a written plan for acquiring paired healthy controls,
7. at least one structured clinician feedback session,
8. a written IRB/data-governance decision path,
9. a deployment/security gap assessment,
10. an external advisory roadmap with named contacts and next actions.

---

# 5. What FLARE Should Not Claim Yet

Until validated, avoid claims that FLARE is:

- clinically approved,
- FDA cleared/authorized,
- HIPAA compliant as a complete deployed service,
- production ready for hospital use,
- proven to outperform radiologists,
- proven to improve patient outcomes,
- validated across broad demographic populations,
- validated for breast cancer,
- validated for full CT/MRI fusion specificity using healthy paired controls.

Instead use language such as:

- research prototype,
- clinical decision-support research system,
- proof of concept,
- externally validating,
- pilot-readiness roadmap,
- clinician-in-the-loop workflow.

---

# 6. Definition of Pilot-Ready

A controlled external pilot should not begin until the team can answer **yes** to all of the following:

- Is intended use clearly defined?
- Are model inputs/outputs/versioning documented?
- Are evaluation splits defensible and patient-safe from leakage?
- Has external validation been performed or formally planned?
- Is clinician review mandatory for every AI output?
- Are PHI/data rules approved by the appropriate institution?
- Is DICOM de-identification validated?
- Are authentication and authorization enforced server-side?
- Are audit logs durable?
- Are backups and incident response tested?
- Can the exact model/deployment version be reproduced?
- Are known limitations shown to every external reviewer?
- Has a qualified regulatory/compliance advisor reviewed the planned pilot?

If any answer is no, that item remains a Phase II blocker rather than something to hide in a pitch.
