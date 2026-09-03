# Advanced Bipolar Rhythm Classifier

## Status and intended use

This is an academic engineering experiment for the ECG V2 PFE. It supports
cardiologist review and is not a diagnosis, medical-device claim, or substitute
for specialist judgment. The final PCB, acquisition settings, clinical taxonomy,
and specialist validation protocol remain pending project decisions.

## Confirmed physiological input scope

The V2 prototype uses four electrodes: RA, LA, LL, and RL. RL is the
reference/return electrode. The model receives only the three bipolar limb
leads:

```text
I   = LA - RA
II  = LL - RA
III = LL - LA = II - I
```

Only I and II are independent. The inference program accepts either I/II/III or
I/II NumPy data in mV. For an I/II input, it reconstructs III as `II - I` and
records the reconstruction in the output. Augmented and precordial leads are not
model inputs. The 12-lead network used for knowledge distillation is a training
teacher only and does not change the deployed input scope.

## Engineering label scope

The model is multi-label and covers the approved initial benchmark:

| Class | Chapman SNOMED CT | PTB-XL SCP-ECG |
| --- | --- | --- |
| Sinus rhythm | 426783006 | SR |
| Sinus bradycardia | 426177001 | SBRAD |
| Sinus tachycardia | 427084000 | STACH |
| Sinus arrhythmia | 427393009 | SARRH |
| Atrial fibrillation | 164889003 | AFIB |
| Atrial flutter | 164890007 | AFLT |

This taxonomy is a reproducible PFE benchmark, not the final clinical taxonomy.

## Data separation

- Chapman training and validation records use the deterministic 80/10/10 split
  already established by the project. Its final 10% is held out for testing.
- PTB-XL folds 1-8 join training.
- PTB-XL fold 9 joins validation.
- PTB-XL fold 10 remains a separate final test.
- Normalization, temperature scaling, and decision thresholds do not use either
  test set.
- Experiment ranking uses validation macro average precision, never test
  performance.

The full configuration contains 53,538 training records, 6,698 validation
records, 4,515 Chapman held-out records, and 2,198 PTB-XL fold-10 records.

## Cross-dataset label audit

The reproducible audit covers 45,150 Chapman records and 21,799 PTB-XL records.
It found 958 Chapman and 32 PTB-XL records with more than one approved target,
including 17 PTB-XL records annotated with both atrial fibrillation and atrial
flutter. It also reports records outside the six-label target scope and the
distribution of PTB-XL statement likelihood values. Those likelihood values are
descriptive metadata, not confidence thresholds.

Artifacts are under `artifacts/rhythm_label_alignment/`.

## Model and training changes

The advanced model combines a one-dimensional residual convolutional encoder
with a branch of deterministic rhythm features:

- estimated heart rate;
- RR mean, standard deviation, RMSSD, and pNN50;
- detected QRS count;
- robust signal range and derivative median absolute difference;
- baseline- and high-frequency-power ratios;
- spectral entropy;
- Einthoven-identity residual when I/II/III are present.

The training pipeline adds:

- masked-signal autoencoder pretraining;
- conservative waveform augmentation;
- balanced sampling across rare targets;
- asymmetric multi-label loss;
- an extra penalty for the hardest negative predictions;
- an atrial-family auxiliary head;
- an AF-versus-AFL auxiliary head;
- soft-target distillation from the frozen historical 12-lead teacher;
- per-class temperature scaling fitted on combined validation data;
- per-class thresholds selected for maximum validation F1 while requiring 0.95
  specificity, increased to 0.98 for AF and AFL;
- atomic resume checkpoints after every supervised epoch.

The model returns probabilities for all six labels. It does not force the labels
to be mutually exclusive.

## Final benchmark results

The selected checkpoint reached 0.8143 validation macro average precision. The
following test results were computed once after model selection and calibration.
Macro average precision is the primary ranking metric; exact-match accuracy
requires the complete six-label prediction vector to match the reference vector.

| Evaluation set | Macro AP | Macro AUROC | Macro F1 | Micro F1 | Exact match |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined validation | 0.8143 | 0.9690 | 0.7505 | 0.8524 | 0.7677 |
| Chapman held-out test | 0.7900 | 0.9720 | 0.7278 | 0.8628 | 0.7898 |
| PTB-XL fold-10 test | 0.6108 | 0.9060 | 0.5032 | 0.7816 | 0.6720 |

PTB-XL fold-10 class results expose the cross-dataset limitations hidden by a
single aggregate score:

| Class | Positives | AP | Precision | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sinus rhythm | 1,674 | 0.9403 | 0.9180 | 0.7622 | 0.7824 |
| Sinus bradycardia | 64 | 0.1954 | 0.2400 | 0.0938 | 0.9911 |
| Sinus tachycardia | 82 | 0.8875 | 0.7526 | 0.8902 | 0.9887 |
| Sinus arrhythmia | 77 | 0.3074 | 0.4848 | 0.2078 | 0.9920 |
| Atrial fibrillation | 152 | 0.9109 | 0.9839 | 0.4013 | 0.9995 |
| Atrial flutter | 7 | 0.4235 | 0.3333 | 0.4286 | 0.9973 |

The high AF average precision but lower thresholded sensitivity reflects the
validation policy requiring at least 0.98 specificity for atrial labels. Atrial
flutter has only seven PTB-XL fold-10 positives, so its estimate is particularly
uncertain. These are engineering benchmark measurements, not clinical accuracy
claims.

Compared with the earlier Chapman-only bipolar baseline, macro AP increased from
0.7185 to 0.7900 on the Chapman held-out split and from 0.4702 to 0.6108 on
PTB-XL fold 10. The advanced model used PTB-XL folds 1-8 for training, whereas
the earlier model treated all PTB-XL data as external. The difference therefore
cannot be attributed to any one architecture or training method.

## Error analysis

Among records with none of the six target labels, the rate of at least one false
positive was 7.26% for 179 Chapman held-out records and 41.50% for 147 PTB-XL
fold-10 records. This confirms substantial residual domain shift and target-scope
ambiguity.

The largest identified cross-confusion was on Chapman: 69.10% of the 178
AF-only records also crossed the atrial-flutter threshold. On PTB-XL, that rate
was 4.00% across 150 AF-only records, while fold 10 contained only five
flutter-only records. AF-versus-AFL output must not be treated as clinically
resolved.

## Experimental signal-quality and uncertainty gate

Inference reports an inconclusive status instead of positive labels when the
experimental quality gate fails or a probability is too close to its decision
threshold. The gate checks non-finite samples, flat or disconnected leads,
implausible QRS counts, excessive amplitude, derivative noise, baseline drift,
high-frequency noise, and I/II/III inconsistency.

These limits are software-research settings only. They are explicitly not the
pending PCB requirements, sampling specification, lead-off criteria, or clinical
abstention policy.

## Controlled ablations

The experiment runner compares:

- independent I/II versus I/II/III at 100 Hz;
- I/II/III at 100, 250, and 500 Hz.

Each exploratory run uses the same deterministic seed, data policy, record
limits, training length, and loss settings. The 100/250/500 Hz comparison is an
offline algorithm ablation and does not select the hardware sampling rate.

The completed exploratory runs used 12,000 training records, 3,000 validation
records, four supervised epochs, no masked pretraining, and no teacher
distillation. They are ranked only by validation macro AP:

| Rank | Input | Rate | Validation macro AP | Chapman test macro AP | PTB-XL test macro AP |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | I/II | 100 Hz | 0.6806 | 0.7187 | 0.4968 |
| 2 | I/II/III | 100 Hz | 0.6658 | 0.7049 | 0.4792 |
| 3 | I/II/III | 250 Hz | 0.6448 | 0.6692 | 0.4458 |
| 4 | I/II/III | 500 Hz | 0.5524 | 0.5619 | 0.3363 |

Under this fixed short-training budget, the two independent 100 Hz leads ranked
first. Explicitly supplying derived lead III did not improve validation, and
higher sample rates required substantially more computation. The 500 Hz model
may be disadvantaged by the same four-epoch budget, so this experiment does not
show that higher-frequency information is generally useless. It also does not
approve a hardware sample rate; that decision still requires analog-front-end,
timing, storage, transmission, and reference-signal validation.

## Reproducibility

Train the full configuration:

```powershell
.\.venv\Scripts\python.exe host/ai/train_advanced_rhythm_classifier.py
```

Run the controlled ablations:

```powershell
.\.venv\Scripts\python.exe host/ai/run_advanced_rhythm_ablations.py
```

Audit labels and analyze final errors:

```powershell
.\.venv\Scripts\python.exe host/ai/audit_rhythm_label_alignment.py
.\.venv\Scripts\python.exe host/ai/analyze_advanced_rhythm_errors.py
```

Predict and explain one compatible WFDB record:

```powershell
.\.venv\Scripts\python.exe host/ai/predict_advanced_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr
.\.venv\Scripts\python.exe host/ai/explain_advanced_rhythm.py database/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr --class-name sinus_rhythm
```

## Limitations and next validation step

- There is no ECG acquired through the final PCB, so hardware-domain performance
  is unknown.
- Chapman and PTB-XL differ in population, equipment, annotation, and target
  prevalence.
- Atrial flutter remains rare, especially in PTB-XL fold 10.
- The deterministic QRS detector, quality limits, augmentations, uncertainty
  margin, and thresholds require validation rather than assumption.
- Distillation can transfer teacher errors.
- Integrated gradients measures model sensitivity, not medical causality.
- Test results estimate benchmark performance only and do not establish clinical
  safety or efficacy.

The next external step is a prospective, versioned collection of synchronized
I/II data from the final PCB, with III reconstructed or recorded as metadata,
electrode/hardware status, a traceable reference ECG, and specialist-reviewed
labels. Sampling rate, filtering, electrical safety, participant protocol, and
acceptance criteria must be approved before that collection.

## Research references

- Wagner et al., PTB-XL dataset and official patient-respecting folds:
  <https://doi.org/10.1038/s41597-020-0495-6>
- Hu et al., reduced-lead ECG knowledge distillation:
  <https://pubmed.ncbi.nlm.nih.gov/36638544/>
- Kiyasseh et al., lead-agnostic ECG representation learning:
  <https://arxiv.org/abs/2203.06889>
