# Target features per recording type

What should eventually be extracted from each DICOM, based on the user's annotated examples
(`Points of interest\` next to the 17 cardiac examples; blue dot = a velocity, blue line = a timing).
Compiled 2026-10-05. Implementation state and the comparison with the marks are in `HANDOFF.md`
("Target features").

The recordings fall into two clinical uses: **diastolic function and atrial conduction**
(TDI, mitral inflow, CW jet velocity) and the **electromechanical window** (aortic valve closure +
end of the T wave).

| Recording | Examples | Feature | Definition (what we extract) | Clinical context |
|---|---|---|---|---|
| **PW tissue Doppler (TDI) at the mitral annulus**, apical 4-chamber | 01_10292, 01_10364, 01_10638, 01_277, 01_376, 01_5012 (no marks, too dark), 03_10673 (AF) | **e′ peak velocity** (dot) | Deepest point of `v_min` in the first negative wave after the T wave; reported as a positive speed (cm/s). | Early diastolic relaxation of the myocardium. A low e′ is one of the four ASE/EACVI 2016 criteria for diastolic dysfunction: septal e′ < 7 cm/s or lateral e′ < 10 cm/s. |
| | | **a′ peak velocity** (dot) | Deepest point of `v_min` in the negative wave after the P wave and before the next R peak. Absent in AF. | Velocity of the annulus during atrial contraction. In AF there is no a′, so only e′ applies (03_10673). |
| | | **a′ onset** (line on the baseline) | Last point where `v_min` returns to 0 before the a′ peak. | Mechanical start of atrial contraction at that site. Atrial electromechanical delay ("PA") runs from P-wave onset to the start of the late-diastolic wave. |
| | | **P-wave onset** (ECG; not marked, but needed) | Start of the P wave before each R peak. | Electrical start of atrial activation. PA-TDI, the classic total atrial conduction time, runs from P-wave onset (lead II) to the a′ **peak** on the lateral left-atrial wall; Merckx et al. validated it against P-wave duration on the signal-averaged ECG (R = 0.91). |
| | | **iLAD** (derived) | (P onset → a′ onset, lateral annulus) − (P onset → a′ onset, septal annulus) | Assumed to mean intra-left-atrial electromechanical delay ("PA lateral − PA septal" in the atrial-delay literature), matching the user's lines at a′ onset. Needs **two TDI recordings per patient** (lateral and septal). To be confirmed. |
| **PW mitral inflow**, apical 4- or 5-chamber | E3Q0C6H6, 10529, L3VHDBZ5 | **E peak velocity** (dot) | Highest point of `v_max` in the first positive wave after the T wave. | Early filling of the left ventricle, driven by the left-atrial–ventricular pressure gradient. |
| | | **A peak velocity** (dot) | Highest point of `v_max` in the positive wave after the P wave. Absent in AF. | Filling by atrial contraction. E/A is the first step in the 2016 grading of diastolic dysfunction. |
| | | **E/e′** (derived) | E from the mitral-inflow recording divided by e′ from the TDI recording **of the same patient**. | Estimate of left-ventricular filling pressure; average E/e′ > 14 is one of the four 2016 criteria. Combines two DICOMs, so it cannot be computed from a single image. |
| **CW, negative systolic jet**, probably tricuspid regurgitation (to be confirmed) | E3Q0C6HS, L3VHDS0Z | **Peak jet velocity** (dot) | Deepest point of `v_min` in each systolic jet. The user marks the **dense edge**, not the faint outer tail, so the tracing convention must match. | The CW cursor runs through the right heart in an apical 4-chamber view, which suggests TR. TR velocity > 2.8 m/s is the fourth 2016 criterion, and 4v² gives the RV–RA pressure gradient. If it is aortic outflow instead, the peak velocity grades aortic stenosis. |
| **CW through the left-ventricular outflow tract and aortic valve**, apical 5-chamber | L3VHCNLD, 01_232, 04_11747, 04_12485, 04_14018 | **Aortic valve closure (AVC)** (line) | Centre of the closing click at the end of the negative outflow signal, 0.2–0.5 s after the R peak (`t_avc`; implemented). | End of left-ventricular ejection. In the EMW method, closure is timed at the midline of the click on CW. |
| | | **End of the T wave** (ECG line, 01_232) | Return of the T wave to the baseline; the rule (e.g. tangent method) is still to be chosen. | End of ventricular repolarisation (QT end). |
| | | **Electromechanical window (EMW)** (derived) | (Q onset → AVC) − (Q onset → T end), in the same beat. Q onset cancels, so this equals AVC time − T-end time. | End of mechanical systole minus the QT interval. ter Bekke et al. 2015 (long QT syndrome): positive in controls (+22 ± 19 ms), negative in LQTS (−43 ± 46 ms), more negative in patients with arrhythmic events; discriminated arrhythmic events better than QTc. |

## Implications for the implementation

- Every feature can be derived from `v_max` / `v_min` plus the ECG, but each needs the beat's ECG
  phases to label it (P wave, R peak, T end): E vs A and e′ vs a′ cannot be told apart from the
  Doppler signal alone.
- The derived ratios need two recordings each: E/e′ (mitral inflow + TDI) and iLAD (lateral + septal
  TDI). This needs pairing of DICOMs per patient and the site of each TDI recording (lateral or
  septal), which may only be in the DICOM header or the study structure.
- AF: detect it from irregular beat-to-beat intervals, then skip a′, A, the P wave and iLAD.

## Open questions to the user

1. Is "iLAD" intra-left-atrial electromechanical delay as above?
2. a′ onset (as drawn; the atrial-delay convention) or a′ peak (PA-TDI)?
3. Are the two CW peak-velocity recordings (E3Q0C6HS, L3VHDS0Z) tricuspid regurgitation?

## Sources

- PA-TDI definition and the Merckx et al. validation: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8563556>
- De Vos et al., PA-TDI and new-onset AF, Heart 2009;95:835: <https://heart.bmj.com/content/95/10/835.full>
- Atrial electromechanical delay, PA lateral − PA septal: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5337037>,
  <https://journals.viamedica.pl/cardiology_journal/article/download/CJ.2012.0089/18196>
- ter Bekke et al., Electromechanical window negativity in genotyped long-QT syndrome patients,
  Eur Heart J 2015: <https://cris.maastrichtuniversity.nl/en/publications/electromechanical-window-negativity-in-genotyped-long-qt-syndrome/>
- Nagueh et al., ASE/EACVI 2016 diastolic function recommendations (summary):
  <https://www.asecho.org/wp-content/uploads/2019/03/Left-Ventricular-Diastolic-Function-Summary-Doc.-to-Publish.pdf>
