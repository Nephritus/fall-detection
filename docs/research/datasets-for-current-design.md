# Datasets for testing the current design (stage-1 CNN + stillness check)

Researched 4 November 2026. The current design: accelerometer only, 25 Hz, ±4 g, one wrist; a 1D-CNN on 4 s windows, then an
alarm only after the wrist stays still (per-second std < 0.02 g) for 10–60 s. Open questions: (1) what the stillness check costs
on falls, which the lab data cannot show because recordings end 3–9 s after the impact; (2) whether the false-alarm rate measured
on Capture-24 (Oxford adults, Axivity AX3) holds for other people and devices; (3) stage-1 sensitivity on more unseen fall data.

## Bottom line

- **No public dataset has real falls, or falls by older adults, on the wrist.** The only public real-fall sets (FARSEEING, OHSU
  multiple-sclerosis falls) use a lower-back, thigh or waist sensor. No public wrist set records 60 s after every impact, so the
  cost of a 60 s check cannot be measured on public data; question (1) needs CMDFALL (on request) or our own recordings.
- Real-world evidence cuts against a long check: in FARSEEING falls (lower back), people who got up rested a median 10.5 s and
  those who could not rested 34.5 s while moving more as they tried (Schwickert 2018, cited in Part A).

| Rank | Dataset | Answers | Access | Main limits |
|---|---|---|---|---|
| 1 | **CMDFALL** (Hanoi) | (1) and (3): left wrist, 50 Hz, 50 people, 400 simulated falls inside one continuous 7.5-min recording per person, so each fall is followed by lying and getting up | e-mail request, "free for research purpose" | young adults; research-only use; 50 Hz from a secondary source |
| 2 | **PAAWS Free-Living** R1+R2 | (2): 40 people × ~7 days, ActiGraph on **both** wrists, 80 Hz ±8 g, second-by-second video labels; 7 of the first 20 aged 55+ | direct download (throttled), ~111 GB per release | data licence not stated; research device |
| 3 | **NHANES 2011–14 raw 80 Hz** | (2) for older adults: 3,062 people aged ≥ 60, ~8 days each, non-dominant wrist, ±6 g | public FTP (NCHS data-user agreement), ~176 MB per person | no activity labels; take a subset |
| 4 | **FallAllD** | (3): 15 people, 35 fall types, wrist at 238 Hz ±8 g | free IEEE account, CC BY 4.0, 401 MB | every record is 20 s, so only short stillness checks |

**BITS-2 (Zenodo, left wrist, CC BY 4.0) was checked and rejected.** Its files are labelled 20 Hz but the timestamps are
unusable (`1.10E+13`), and the signal says otherwise: assuming 20 Hz, walking and jogging peak at about 0.7 and 0.9 Hz instead of
about 1.7–2 and 2.5–3 Hz, only 12% of falls show any free fall before the impact and those last 1–2 samples (a 0.3–0.5 s free
fall is 6–10 samples at 20 Hz), and 2-minute walking trials hold a median of 447 samples. The effective rate looks like roughly
7–10 Hz or irregular, so 4 s windows and per-second stillness cannot be reproduced on it.

Part A covers fall datasets, Part B free-living wear; every claim there cites its source, and `[UNVERIFIED]` marks what was not
confirmed.

# Part A. Fall datasets


*Compiled 2026-11-04; every link was opened on that date. Tags: **[UNVERIFIED]** = not confirmed in a primary source; **[INFERENCE]** = my reasoning, not stated by a source; **[MEASURED]** = I downloaded the data and computed it myself (method in §C). "Post-impact" = continuous signal after the acceleration peak, which is the number that decides whether the 10–60 s stillness check can be measured.*

Excluded because already used: WEDA-FALL, UP-Fall, Erciyes (UCI 455), UMAFall, Capture-24, LAFAN1. **WEDA-FALL:** the only commit since 2023 is "Update contact email" (2026-08-24); the data last changed on 2022-10-28 ("Added Fall timestamps") ([commits](https://github.com/joaojtmarques/WEDA-FALL/commits/main)), so there is nothing new to use. I did not re-check UP-Fall, Erciyes or UMAFall for new versions.

### A. Summary table

| # | Dataset | Wrist? (side) | Rate / range | Subjects (age) | Falls | **Post-impact continuous** | Access / licence | Red flags |
|---|---|---|---|---|---|---|---|---|
| 1 | **CMDFALL** | **Yes, left wrist** + left hip | 50 Hz, ±16 g | 50 (21–40) | 400 simulated (8 types × 50); subject lies still, then gets up for the next action | Each subject is one **continuous 7.5 min stream**. Each fall action is allotted 30 s (20 s for falls from a chair) and ends when the subject "lies motionless on the floor". After that the subject gets up for the next action | E-mail request, "free for research purpose" | Request-only; research-only wording; mattress; young subjects |
| 2 | **BITS-2** (Nandi 2023) | **Yes, left wrist** | 20 Hz (stated); range not stated | 41 (20–47) | 328 files = 41 × 8 simulated types | Paper: fall trials last 40 s. **[MEASURED]** median 17 s after the peak, 75th percentile 29 s, 22 % ≥ 30 s, max 153 s | **Direct download** from Zenodo, CC-BY-4.0, 26 MB | 20 Hz is below our 25 Hz; timestamps were saved as `1.10E+13` (precision lost); sensor labels look swapped; no older adults |
| 3 | **FallAllD** | **Yes** (side [UNVERIFIED]) + waist + neck | 238 Hz, ±8 g | 15 (21–53) | 1,722 fall samples over 3 devices, 35 types (simulated) | Every sample is 20 s, so post-impact is under 20 s [INFERENCE] | Free IEEE account (no membership), CC-BY-4.0, 401 MB | Short records; Casilari's table lists chest where DataPort lists neck |
| 4 | **SmartFallMM** | **Yes, wrists** (watch + Meta sensor) + hips | watch 32 Hz; Meta sensor 50 Hz | 30 young + 21 old (ages [UNVERIFIED]) | 5 types, **young subjects only** | One cropped clip per trial. **[MEASURED]** watch clips about 6–37 s; Meta wrist clips about 3–10 s | GitHub, no login; "© … All rights reserved" | No open licence; older adults did ADLs only |
| 5 | SmartFall / Notch (2018) | Yes (Smartwatch set: left) | 31.25 Hz | 7 (21–55); 7 (20–35) | 91; 107 | Paper calls the recordings continuous; length not stated | URL now returns **HTTP 403** | Not obtainable |
| 6 | TST Fall v2 | Yes, **right wrist** + waist | 100 Hz, ±8 g | 11 (22–39) | 132 (4 types, 3 end lying, 1 ends sitting) | Whole sequences 3.8–18.3 s | DataPort **subscription required**; original site returns 404 | Paywalled; short |
| 7 | AybuFall (2026) | **Forearm**, right arm (distal to elbow) + forehead | 200 Hz | 17 (23–43) | 11 types × 3 | Trials average about 20 s; recording stops when the movement ends | Zenodo, CC-BY-4.0, 30 MB | Forearm, not wrist; short |
| 8 | FARSEEING (real) | **No** (lower back L5 / thigh) | 20 or 100 Hz | 15 public subjects (56–86); full set 94 fallers, mean age 76 | 22 public / 208 verified **real** falls | **1,200 s traces** with annotated impact | On request; **not shared with primarily commercial projects** | No wrist sensor; commercial use excluded |
| 9 | OHSU MS real falls (FFFStudy) | **No** (waist or trouser pocket) | 50 Hz, ±8 g | 25 people with MS (33–76) | 54 **real** falls (49 identifiable) | **Whole days** (.npz files), fall timestamps provided | Free IEEE login, CC-BY-4.0, 25.65 GB | No wrist sensor |
| 10 | Project Gravity | Yes (LG G Watch) + thigh phone | phone 50 Hz, watch 157 Hz | 2 (26–32) | 72 (12 types) | Samples 9–86 s | GitHub; data licence not stated | Only magnitude + vertical acceleration (no 3 axes); 2 subjects |
| 11 | MobiAct / MobiFall | No (phone in trouser pocket) | 87 Hz, ±2 g | 57 (20–47) | 647 | Some falls include a "long interval of lying" | [UNVERIFIED] | No wrist sensor; ±2 g |
| 12 | KFall (+ add-in) | No (lower back) | 100 Hz, ±16 g | 32 young men; add-in 10 older adults (ADLs only) | 2,346 | 2–41 s | Google Drive | No wrist sensor |
| 13 | SisFall | No (waist) | 200 Hz | 38 (19–75) | 1,798 | 10–180 s files | Website | No wrist sensor |
| 14 | DOFDA (taken to be Cotechini 2019 [INFERENCE]) | No (waist) | 33 Hz, ±16 g | 8 (22–29) | 13 types, including "ending up lying **with recovery**" | 2–17 s | Data in Brief | No wrist sensor |
| 15 | UniMiB SHAR | No (phone in pocket) | 50 Hz, ±2 g | 30 | 1,699 | **1 s windows** | Website | Pre-cut windows, no raw stream |
| 16 | Graz UT OL | No (phone in waist bag) | **5 Hz** | 5 | 220 | — | figshare | No wrist; 5 Hz |
| 17 | URFD | No (pelvis) | 256 Hz | 6 | 30 | 2–14 s | Website | No wrist sensor |
| 18 | UNIVRFall (2026) | No (lower back L1–L2) | 100 Hz, ±16 g | 39 (20–49) | 573 | — | Zenodo, CC-BY-4.0 | No wrist sensor |
| 19 | QUIDA (PeerJ 2025) | No (phone on torso, T5–T10) | 62.5 Hz | 10 | 10 types | — | OSF | No wrist sensor |
| 20 | ShimFall&ADL | No (chest) | 50 Hz | 35 (19–34) | — | — | Zenodo, **CC-BY-NC-ND** | No wrist sensor; non-commercial |

**What answers our open questions.**

- **Real falls or falls by older adults on a wrist:** no public dataset has these.
  - The only public real-fall accelerometer datasets are FARSEEING and FFFStudy, and both are waist sensors ([Casilari & Silva 2022](https://doi.org/10.1016/j.measurement.2022.111843), §3.1).
  - Every wrist dataset with older adults has ADLs only: WEDA-FALL, the SmartFallMM "old" group and the KFall add-in (lower back).
- **Wrist falls with a long post-impact segment:** CMDFALL (continuous stream with natural get-up) and BITS-2 (up to 153 s, scripted lying). No public wrist set systematically reaches **60 s** after the impact.

---

### B. Per-dataset notes

#### 1. CMDFALL (MICA Hanoi + PTIT)
- **Link:** <https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html>. Paper: Tran et al., ICPR 2018, [doi:10.1109/ICPR.2018.8546308](https://doi.org/10.1109/ICPR.2018.8546308).
- **Sensors:** two WAX3 wireless accelerometers (Newcastle Open Lab), worn **on the left wrist and left hip**, ±16 g ([project page](https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html)). Also 7 Kinect cameras.
- **Sampling rate:** 50 Hz according to [Casilari & Silva 2022, Table 2](https://doi.org/10.1016/j.measurement.2022.111843). The project page only says the sensor can sample "up to 2 ks/s" [50 Hz UNVERIFIED in the primary source].
- **Subjects:** 50 (30 men / 20 women), aged 21–40. No older adults.
- **Falls:** 20 actions per subject, 8 of them falls: forward, back, left and right while walking; left and right from a bed; left and right from a chair. All are simulated onto a mattress, giving 400 falls.
- **Continuous recording:** "Each person performed all 20 actions in 7.5 minutes", about 375 minutes in total.
  - Falls while walking or from a bed have a recording length of 30 s; falls from a chair have 20 s.
  - Annotation runs from "Subject stands straight before … fall" to "Subject lies motionless on the floor" ([project page](https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html)).
  - Because the stream is continuous, every fall is followed by the subject getting up to start the next action. This makes it the only wrist set where a **fall → lying → getting up** sequence exists without being cut off [INFERENCE from the protocol; how long subjects lie still was not measured].
- **Impact annotation:** only action start and end are annotated. The impact must be taken from the acceleration peak [INFERENCE].
- **ADL:** 12 non-fall actions: walk, run, stagger, crawl, move a chair, hand/knee movements, pick up an object with each hand, jump, sit down and stand up from a chair and from a bed, lie on a bed and sit up.
- **Access:** "available and free for research purpose … please contact thanh-hai.tran@mica.edu.vn". No formal licence is shown.
- **Format and size:** ASCII text with UNIX timestamps. The accelerometer data is about 2 MB per subject (Kinect data about 7 GB per subject).
- **Open questions it answers:**
  - stage-1 sensitivity on an unseen dataset, left wrist, so no mirroring is needed;
  - stillness-check pass rate when the subject lies still and then gets up;
  - the false-alarm rate inside the same stream.
- **Red flags:** access is on request; "research purpose" wording may exclude commercial use [UNVERIFIED]; young subjects falling onto a mattress.

#### 2. BITS-2 — "Inertial measurement and heart-rate sensor-based dataset for geriatric fall detection using custom built wrist-worn device"
- **Rejected after checking the signal** (see Bottom line): the effective sampling rate is not 20 Hz (walking and jogging peak 2–3× too slow, free fall lasts 1–2 samples), so every time-based number below is unreliable.
- **Links:**
  - paper: [Nandi et al., Data in Brief 2023, doi:10.1016/j.dib.2023.109812](https://pmc.ncbi.nlm.nih.gov/articles/PMC10709028);
  - data: [Zenodo 10.5281/zenodo.10013090](https://doi.org/10.5281/zenodo.10013090) (`full_dataset.zip`, 26,396,425 bytes, licence **cc-by-4.0** according to the Zenodo API).
- **Sensors:** custom device on the **left wrist**: Snapdragon 820c with an MPU6500 (accelerometer and gyroscope), a GY-273 magnetometer and a MAX30102 heart-rate sensor. Motion is sampled at 20 Hz, heart rate at 1 Hz. The accelerometer range is not stated [UNVERIFIED].
- **Subjects:** 41 volunteers aged 20–47 (per-subject table in the paper). The title says "geriatric", but there are **no older adults**.
- **Falls:** 8 simulated types in a padded chamber, "5 trials" of 40 s each:
  - forward fall onto the knees; right; left; forward; seated on a bed and falling to the ground; forward with weight on the hands; backward from a seated position; grabbing while falling.
  - There are no explicit recovery falls. The ADLs include "lying on back and getting up".
- **What the files actually contain [MEASURED]:**
  - 328 fall CSVs = 41 users × 8 types, i.e. one file per user and type (not 5); 656 ADL CSVs.
  - Rows interleave the sensors. The gravity-including acceleration is under the label `acg` (median |a| = 9.85 m/s²), while `acc` is close to gravity-free (median 0.19). That is the opposite of the paper's legend.
  - Timestamps are written as `1.10E+13`, so the real sampling rate cannot be checked. The numbers below assume 20 Hz.
  - File length: median 544 samples, about 27 s.
  - Post-impact length (after the maximum |a|): median **17 s**, 75th percentile **28.6 s**, **22 % ≥ 30 s**, 45 % ≥ 20 s, max 153 s.
  - Peaks: 239 of 328 files have exactly one peak above 2 g; 40 have none.
  - Peak |a|: median 6.1 g, 75th percentile 9.5 g, so it will clip at ±4 g.
  - Noise at rest: the quietest second has a median std of **0.003 g**, well below our 0.02 g threshold.
  - Stillness: of the 73 falls with ≥ 30 s post-impact, **97 %** contain a run of ≥ 10 s and **90 %** a run of ≥ 20 s in which the per-second std is below 0.02 g. In other words, subjects lay still after the fall, as scripted.
  - ADLs: about 5.6 h at 20 Hz.
- **Impact annotation:** none; use the peak.
- **Access:** direct download, no login. Format: CSV.
- **Open questions it answers:** stillness pass rate at 10, 20 and 30 s on a large set of left-wrist falls where the subject stays down (an upper bound on the pass rate, since there are no get-up attempts [INFERENCE]); stage-1 sensitivity on an unseen dataset after upsampling to 25 Hz.
- **Red flags:** 20 Hz is below 25 Hz; the effective rate cannot be verified; sensor labels look swapped; the file count does not match the paper.

#### 3. FallAllD
- **Links:** [IEEE DataPort, doi:10.21227/bnya-mn34](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living); paper [Saleh et al., IEEE Sensors J. 2021, doi:10.1109/JSEN.2020.3018335](https://doi.org/10.1109/JSEN.2020.3018335).
- **Sensors:** three identical RF-Track loggers worn on the **waist, wrist and neck**, each with an LSM9DS1. Accelerometer 238 Hz, ±8 g; gyroscope 238 Hz; magnetometer 80 Hz; barometer 10 Hz (DataPort). Wrist side [UNVERIFIED].
- **Subjects and falls:** 15 participants (7 women / 8 men) aged 21–53, 44 ADL types and 35 fall types, 6,605 samples (4,883 ADL / 1,722 falls), **each 20 s long** ([Casilari & Silva 2022, Table 2](https://doi.org/10.1016/j.measurement.2022.111843); that table lists "Waist, Wrist, Chest", DataPort lists neck). The DataPort page states 26,420 files.
- **Post-impact:** under 20 s by construction [INFERENCE]. This is the same limitation as the datasets we already use, so it does not unlock the stillness question.
- **Access (verified 2026-11-04):**
  - The page is under "Open Access". It says "Open Access dataset files are accessible to all logged in users… free IEEE account… IEEE Membership is not required". Licence: CC-BY-4.0.
  - A generic "requires an IEEE DataPort Subscription" pop-up also appears in the page source; the Open Access text is the one specific to this dataset.
- **Size and format:** main file 401.39 MB, fall videos 169.64 MB; CSV in a zip, plus scripts to convert to `.mat`, `.h5` or `.pkl`.
- **Hugging Face mirror:** [IdaFLab/FallAllD-Wrist](https://huggingface.co/datasets/IdaFLab/FallAllD-Wrist) is a single 1.37 GB "processed" CSV. Its columns are only `acc_x…gyro_z,label`: no subject or trial IDs and no licence stated. Do not use it.
- **Open questions it answers:** stage-1 sensitivity on the largest unseen set of wrist falls (35 types). The 10 s check can only be measured partially.

#### 4. SmartFallMM (Texas State, Ngu et al.)
- **Links:** <https://github.com/txst-cs-smartfall/SmartFallMM-Dataset>; paper on [IEEE Xplore 11651591](https://ieeexplore.ieee.org/abstract/document/11651591) (not read; paywalled).
- **Sensors:** smartwatch and smartphone at 32 Hz; "Meta" sensors on the wrist and hip at 50 Hz; skeleton at 30 fps (README). Which wrist and which devices are used [UNVERIFIED].
- **Subjects:** "Young" (30 multimodal-ready) and "Old" (21) groups.
  - Falls (back, front, left, right, rotate) are performed by the **young group only**. The old group did drinking, picking up, jacket, stepping, sweeping, washing, waving and TUG/walking.
  - Ages [UNVERIFIED].
- **Format:** per-trial CSVs that were manually cropped from longer recordings. The labelling description says trials are "identified by spikes and rest periods".
  - **[MEASURED]** 679 young watch fall files and 692 Meta-wrist fall files.
  - In 15 random files of each, watch clips were 201–1,189 rows (about 6–37 s at 32 Hz) and Meta-wrist clips 144–476 rows (about 3–10 s at 50 Hz).
- **Size:** about 530 MB (GitHub API `size` = 542,867 KB).
- **Licence:** "© SmartFall Group, Texas State University. All rights reserved." with an acknowledgement request. **There is no open licence.**
- **Open questions it answers:** stage-1 sensitivity on a commodity smartwatch; ADLs from older adults on the wrist (a false-alarm sanity check). It does not answer the stillness question (clips too short).
- **Older SmartFall data (2018):** [Mauldin et al., Sensors 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6210545).
  - "Smartwatch": MS Band 2 on the left wrist, 31.25 Hz, 7 subjects aged 21–55, 91 falls.
  - "Notch": 7 subjects aged 20–35, 107 falls.
  - The paper says both are continuous streams with fall labels.
  - Their URL `cs.txstate.edu/~hn12/data/SmartFallDataSet` now redirects to `userweb.cs.txstate.edu/...` and returns **HTTP 403**.

#### 5. TST Fall detection v2 (Univ. Politecnica delle Marche)
- **Link:** [DataPort, doi:10.21227/H2QP48](https://ieee-dataport.org/documents/tst-fall-detection-dataset-v2).
- **Sensors and protocol:** 11 young actors. Raw acceleration from IMUs on the **waist and right wrist**, plus Kinect v2. Falls: front, back and side ending up lying, and backward ending up sitting. Each action is repeated 3 times, giving 264 sequences (132 falls).
- **Sampling and length:** Shimmer at 100 Hz, ±8 g ([Casilari & Silva 2022, Table 2](https://doi.org/10.1016/j.measurement.2022.111843)). Sequences last 3.84–18.34 s ([Casilari 2017, Table 5](https://pmc.ncbi.nlm.nih.gov/articles/PMC5539544)).
- **Access:** a "Standard Dataset" on DataPort, so a **subscription is required** (the page says so). The original download site `tlc.dii.univpm.it/blog/databases4kinect` returns **404**.
- **Size:** 11 zips of 0.70–1.31 GB each, mostly depth data.
- **Zenodo copy:** the ["Unicomfacauca" modified version](https://zenodo.org/records/3961894) (CC-BY) contains depth image sequences only ([repo](https://github.com/CristianChilito/Fall-Detection-System-Unicomfacauca)).
- **TST v1** is depth-only ([DataPort](https://ieee-dataport.org/documents/tst-fall-detection-dataset-v1)).
- **Verdict:** right wrist, but paywalled and short. Low priority.

#### 6. AybuFall (2026)
- **Links:** [Tokgöz & Kocaoğlu, BMC Res Notes 2026, doi:10.1186/s13104-026-07679-9](https://doi.org/10.1186/s13104-026-07679-9); [Zenodo 18175644](https://zenodo.org/records/18175644): CC-BY-4.0, a single `aybu_fall.csv` of 30.3 MB.
- **Sensors:** MbientLab MetaMotionS on the **forearm (distal to the elbow)**, on the right arm according to the Zenodo description, plus the forehead. 200 Hz. Range [UNVERIFIED].
- **Protocol:** 17 subjects aged 23–43; 11 fall types, 13 ADLs and 5 prayer movements, each repeated 3 times. Trials average about 20 s, and recording is "terminated upon completion of the movement".
- **Use:** prayer movements are a hard-negative test for false alarms; the falls are a minor sensitivity test (forearm ≠ wrist). It does not answer the stillness question.

#### 7. FARSEEING real-world fall repository
- **Links:** [Klenk et al., Eur Rev Aging Phys Act 2016, doi:10.1186/s11556-016-0168-9](https://doi.org/10.1186/s11556-016-0168-9).
- **Scale:** more than 300 falls recorded and 208 verified, from 94 fallers with a mean age of 76.1. The sensor is at L5 in 72 % of cases and on the thigh otherwise; 73 % at 100 Hz and 27 % at 20 Hz.
- **Public subset:** "A dataset of 20 selected fall events is available on request". The consortium "does not share data with projects having primarily commercial interests" (same paper).
- **What the subset contains** ([Casilari & Silva 2022 §3.1.1](https://doi.org/10.1016/j.measurement.2022.111843)):
  - 22 falls from 15 subjects aged 56–86;
  - every trace is **1,200 s** long and the fall instant is annotated;
  - "Just in one of the samples the faller was able to get up without help".
- **Website:** farseeingresearch.eu is online, but its last post dates from 2018 and I found no data-request page.
- **Relevant result** ([Schwickert et al., Gerontology 2018, doi:10.1159/000478092](https://doi.org/10.1159/000478092), lower-back sensor):
  - Resting time on the ground was a median **10.5 s** in falls where the person recovered to standing, versus **34.5 s** in falls where they did not.
  - Resting longer than 24.5 s predicted that the person would not recover.
  - Fallers who did not recover showed *more* cumulative pitch movement (median 308° versus 76°), i.e. they made failed attempts to get up.
  - [INFERENCE] This directly challenges a 60 s stillness requirement: people who cannot get up may keep moving.
- **Red flags:** no wrist sensor; non-commercial restriction; request-only.

#### 8. OHSU "Real-world falls in Multiple Sclerosis" (FFFStudy)
- **Link:** [DataPort, doi:10.21227/xcnr-nz43](https://ieee-dataport.org/open-access/real-world-falls-multiple-sclerosis-ms-0): Open Access (free IEEE login), CC-BY-4.0, `ohsu-ms-fall-data.zip` of 25.65 GB.
- **Contents (readme):** raw 3-axis accelerometry at 50 Hz in one `.npz` per participant-day; `falls_groundtruth.csv` with fall timestamps; time-of-flight tracking at 4 Hz.
- **Study:** 25 participants with MS, 8 weeks, 54 falls, 0.65 false alarms/day reported (DataPort abstract).
- **Sensor and ages:** sensor on the waist or in a trouser pocket, ±8 g; ages 33–76; only 49 falls can be identified from the ground-truth file ([Silva et al., Measurement 2024](https://riuma.uma.es/rest/api/core/bitstreams/7a55637b-b075-4401-93d1-8994cb8ad032/content) §2.2.2).
- **Use:** no wrist sensor, so stage 1 cannot run on it. It is the only free, CC-BY, continuous set of **real** falls.
  - It can measure how long the body stays still after real falls.
  - [INFERENCE] If the waist moves, the wrist almost certainly moves too, so the waist-still duration is an upper bound on the wrist stillness pass rate (device noise floors differ).

#### 9. Project Gravity (SINTEF / NTNU)
- **Link:** <https://github.com/SINTEF-SIT/project_gravity> (`fallData/`, JSON). The last push was in 2015. The README gives Apache-2.0 for the code; the data licence is not stated.
- **Data:** 2 subjects aged 26–32; LG G Watch plus a thigh smartphone; 72 falls of 12 types; samples last 9–86 s ([Casilari 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5539544)). Sampling: phone 50 Hz, watch 157 Hz ([Casilari & Silva 2022 Table 2](https://doi.org/10.1016/j.measurement.2022.111843)).
- **Red flags:** the data stores only the magnitude and vertical acceleration, not three axes (same source). That is incompatible with our 3-axis CNN. Skip.

#### 10–20. Datasets without a wrist sensor (brief)
Sensor positions are from [Casilari & Silva 2022 Table 2](https://doi.org/10.1016/j.measurement.2022.111843) unless another source is given.
- **MobiAct/MobiFall:** smartphone in the trouser pocket, ±2 g. Falls are performed "with and without a long interval of lying" ([Casilari 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5539544)). Only useful as a non-wrist reference for lying.
- **KFall:** sensor on the lower back; 32 young men. The [add-in](https://sites.google.com/view/kfalldatasetadd-in) adds 10 older subjects who did ADLs only.
- **SisFall:** waist sensor; ages 19–75. Older subjects did no falls, except one judo/aikido expert over 50 ([Casilari 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5539544)).
- **DOFDA** ([Cotechini 2019](https://pubmed.ncbi.nlm.nih.gov/31372467); that this paper *is* "DOFDA" is my [INFERENCE] from the matching 8 subjects and waist sensor): waist MARG sensor. Includes "ending up lying with recovery" falls, the only explicit recovery falls found, but not on a wrist.
- **UniMiB SHAR:** 1 s windows only. **Graz:** 5 Hz. **URFD:** pelvis.
- **UNIVRFall** ([Zenodo 18346755](https://zenodo.org/records/18346755)): lower back.
- **QUIDA** ([PeerJ 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11970414/)): smartphone on the torso.
- **ShimFall&ADL** ([Zenodo 3901285](https://zenodo.org/records/3901285)): chest, CC-BY-NC-ND.

#### Leads that are not public (worth an e-mail)
- **Rennes LTSI (the FallAllD authors).** Long-term monitoring of 16 nursing-home residents over 80, with the sensor on the wrist (9) or neck (7), about 400 days, **2 real falls** ([Saleh et al. 2021, PubMed 34874864](https://pubmed.ncbi.nlm.nih.gov/34874864); counts from [Casilari & Silva 2022 Table 1](https://doi.org/10.1016/j.measurement.2022.111843)).
  - A follow-up recorded 570 days of free living in people over 80 (LIS3DH, 50 Hz, ±8 g; body position [UNVERIFIED]). It only states "All relevant data are within the paper or Supporting information", so the raw data was not released ([Abbas et al., PLOS ONE 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9886261)).
- **DARE FALLSPREDICT-GP (Bologna).** Participants over 65 wore, for one week, an Empatica EmbracePlus on the **non-dominant wrist** and a DynaPort on the lower back. The authors present **3 real-world falls** ([BGS poster, Palmerini et al.](https://www.bgs.org.uk/real-world-falls-collected-with-a-wearable-multi-sensor-and-multi-domain-setup)). Not released.
- **FARSEEING extended set (143 falls)** and the **Fall Sensing project (403 falls)** are described as "not publicly available yet" or "undisclosed" ([Casilari & Silva 2022 §3.1.1](https://doi.org/10.1016/j.measurement.2022.111843)).

---

### C. How the [MEASURED] numbers were obtained
The scripts ran in `/tmp`; nothing was added to the repo.
- **BITS-2:**
  - Downloaded `full_dataset.zip` from Zenodo and parsed every `Dataset/fall/*/*.csv`.
  - Took the rows labelled `acg` as gravity-including acceleration in m/s², divided by 9.80665 and computed |a|.
  - Impact = argmax |a|; post-impact = (N − 1 − i)/20 s.
  - Stillness = per-second std of |a| below 0.02 g, starting 2 s after the impact; reported as the longest run of consecutive still seconds.
  - Peaks = local maxima above 2 g that are more than 5 s apart.
- **SmartFallMM:** listed the repository tree through the GitHub API, then counted rows in 15 randomly chosen fall CSVs per folder (seed 0) using raw.githubusercontent.

---

### D. Ranked top 3 for our purpose

1. **CMDFALL.**
   - It is the only public set where each fall on the **left wrist** (no mirroring needed) is followed, inside a **continuous 7.5 min stream**, by lying and then getting up.
   - 400 falls at 50 Hz, ±16 g. It measures both stage-1 sensitivity on unseen data and the real cost of the stillness check, including the "fall then get up" case that currently fails the check.
   - Cost: one e-mail; check whether the licence allows commercial use.
2. **BITS-2.**
   - Downloadable today under CC-BY, left wrist, 328 falls.
   - It is the only open wrist set with a meaningful number of falls having **≥ 30 s** after the impact (73, measured). It gives a best-case pass rate for the stillness check at 10, 20 and 30 s, plus an unseen sensitivity test.
   - Caveats: 20 Hz (needs upsampling to 25 Hz), timestamps that cannot be trusted, swapped sensor labels, no older adults.
3. **FallAllD.**
   - The largest wrist fall set (1,722 fall samples, 35 fall types, 15 subjects), CC-BY, with a free IEEE login.
   - It is the strongest **stage-1 sensitivity** re-test on unseen data (resample 238 Hz to 25 Hz). With 20 s records it only partly informs the 10 s check.

*Next after these:* the **OHSU MS real-falls** set (CC-BY, free login, continuous days). It cannot test stage 1, but it is the only free source of real-fall post-impact behaviour. Combined with Schwickert 2018 (people who cannot get up still move), it is the best evidence on whether a 10–60 s stillness requirement misses real long lies.

# Part B. Free-living wrist datasets


Scope: open (or obtainable) datasets with **raw** wrist acceleration (≥ 25 Hz, not counts), worn in free living for hours or more per person. Goal: re-measure the current design's false alarms per day (1D-CNN on 4 s windows, then a 10–60 s stillness check) on people other than Capture-24's. Checked 2026-11-04. Each fact links to its primary source. **[COMPUTED]** = I worked it out from the linked source files. **[INFERENCE]** = my reasoning, not stated by the source. **[UNVERIFIED]** = I could not confirm it.

Three things apply to every candidate:
- **None has a fall recorded in the wild on the wrist.** The only real-world fall repository found (FARSEEING) has sensors on the lower back or thigh only (see "Real-world falls" below). The cost of the stillness check on real falls stays unmeasurable with these datasets.
- None records at our exact spec except ExtraSensory's Pebble watch (25 Hz, ±4 g). Everything else must be resampled to 25 Hz and clipped to ±4 g. IMU50 (±2 g) cannot be clipped *up* to ±4 g.
- "Post-impact continuous length" does not apply (no falls). What matters here is continuous wear time, because the stillness check needs 10–60 s of uninterrupted signal after each CNN decision.

## Summary table

| # | Dataset | Wrist / side | Device | Rate, range | Subjects (age) | Continuous free-living wear | Labels | License / access | Size | Red flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [PAAWS Free-Living R1+R2](https://www.paawsstudy.org/data.html) | **both** wrists, dorsal | ActiGraph GT9X Link (research) | 80 Hz, ±8 g | 40 released (of 237); R1: 7 of 20 aged 55+ | ~7 days, 24 h; 6,450 h total | second-by-second video labels (waking hours); PSG on some nights | DRS direct download (throttled); dataset license [UNVERIFIED] | R1 FL ~111 GB, R2 FL 112 GB | few older adults; research device |
| 2 | [NHANES 2011–14 PAX80_G/H](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm) | non-dominant (~99%) | ActiGraph GT3X+ (research) | 80 Hz, ±6 g | 14,693 aged 3–80+; **3,062 aged ≥60** [COMPUTED] | up to 194 h (~8 days), 24 h | none (wear/sleep predictions only) | public FTP, NCHS data-user agreement | ~176 MB/person; ~1.0 + ~1.2 TB | no activity labels; huge |
| 3 | [BIDSleep Apple Watch](https://physionet.org/content/bidsleep-dataset/1.0.1/) | wrist, side [UNVERIFIED] | **Apple Watch** (consumer) | ~50 Hz [COMPUTED], range [UNVERIFIED] | 47 adults, ages [UNVERIFIED] | nights only; 253 nights at home | EEG sleep stages (Dreem 2) | ODC-By 1.0, open | 5.9 GB zip | nights only |
| 4 | [IMU50](https://zenodo.org/records/21468410) | non-dominant | ActiGraph LEAP | 128 Hz, **±2 g** | 50 aged 40–75; 27 aged ≥60 [COMPUTED] | ~136 h (5.6 days), 24 h | none | CC BY 4.0, Zenodo direct | 46.7 GB | **±2 g clips below our ±4 g** |
| 5 | [UK Biobank field 90001](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=90001) | **dominant** | Axivity AX3 (same as Capture-24) | 100 Hz, ±8 g | ~103.6k aged 45–79 | 7 days, 24 h | none (linkable hospital records) | approved application + fee | ~100 MB/person compressed | cost/application; same device as Capture-24 |
| 6 | [ExtraSensory](http://extrasensory.ucsd.edu/) | wrist, side [UNVERIFIED] | **Pebble** smartwatch (consumer) | **25 Hz, ±4 g** | 56 with watch data; 60 total aged 18–42 | **20 s of every minute** | self-reported context labels | free if you cite; no formal license | watch accelerometer zip 800 MB | duty-cycled; young |
| 7 | [Sleep-Accel (Walch)](https://physionet.org/content/sleep-accel/1.0.0/) | wrist, side [UNVERIFIED] | **Apple Watch** | ~50 Hz [COMPUTED] | 31 | one lab PSG night | PSG sleep stages | ODC-By 1.0, open | 550 MB zip | one night, in the lab |
| 8 | [Newcastle PSG + GENEActiv](https://zenodo.org/records/1160410) | **both** wrists | GENEActiv | 85.7 Hz, ±8 g [UNVERIFIED for this study] | 28 sleep-clinic patients | one night | PSG sleep stages | CC BY 4.0, Zenodo direct | 0.96 GB | one night |
| 9 | [Mobilise-D TVS](https://zenodo.org/records/15861907) | non-dominant, **some participants only** | custom wrist IMU | rate [UNVERIFIED] | 108 (older healthy adults, PD, MS, hip fracture, COPD, CHF) | 2.5 h | gait reference (INDIP) | **CC BY-NC-ND 4.0**, Zenodo direct | 58.4 GB | non-commercial; wrist only for some |
| 10 | [BEAT-PD (CIS-PD / REAL-PD)](https://www.synapse.org/beatpdchallenge) | wrist; REAL-PD on most-affected side | Apple Watch S2 / Moto 360 (consumer) | rate [UNVERIFIED] | PD patients | **20-min segments only**, time-of-day removed | patient symptom diaries | Synapse; public data-use statement; MJFF terms; no redistribution | [UNVERIFIED] | fragments only; PD |
| 11 | [GOTOV](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/2) | **right** wrist | GENEActiv | 83 Hz (88 Hz in text), ±8 g | 35 aged >60 | ~1.5 h, semi-lab protocol | 16 activity labels | **CC BY-NC-SA 4.0**, direct | 1.25 GB | not free living; non-commercial |
| 12 | [PAMAP2](https://archive.ics.uci.edu/dataset/231) | dominant wrist | Colibri IMU | 100 Hz | 9 (mean 27 y) | scripted protocol | 18 activities | UCI [license UNVERIFIED] | — | scripted; young |
| 13 | [WISDM 2019](https://archive.ics.uci.edu/dataset/507) | wrist (smartwatch) | LG smartwatch [UNVERIFIED model] | **20 Hz** | 51 | 18 activities × 3 min | scripted labels | UCI [license UNVERIFIED] | 296 MB | **< 25 Hz**; scripted |
| 14 | [LTMM](https://physionet.org/content/ltmm/1.0.0/) | **none (lower back)** | 3D accelerometer + gyroscope | 100 Hz | 71 aged 65–87 | 3 days | faller / non-faller by history | ODC-By 1.0 | 20.8 GB | **no wrist** |
| 15 | [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) | **none (thigh, lower back)** | Axivity AX3 | 50 Hz | 18 aged 70–95 | ~40 min | video labels | UCI | 42 MB | **no wrist** |
| 16 | [MESA actigraphy (NSRR)](https://github.com/nsrr/mesa-documentation/blob/master/pages/actigraphy-introduction.md) | wrist | Actiwatch Spectrum | **30-s activity counts** | 2,237 aged 54–95 | 7 days | sleep/wake scoring | NSRR data request | — | **counts, not raw** |

## Per-dataset notes

### 1. PAAWS Free-Living (Northeastern, Intille lab)
- Free-living (FL) protocol: 8 days (7+ full days) with 5 GT9X Link on the right and left wrists (dorsal), right waist, right thigh and right ankle, plus one taped to the phone. Accelerometer at 80 Hz; the IMU was disabled to save battery. Labels are second-by-second, from chest-camera video during waking hours. [paawsstudy.org/data](https://www.paawsstudy.org/data.html)
- Range: "primary ActiGraph accelerometers sampled at a rate of 80 Hz (range ±8 g)". [Potter et al. 2025, IMWUT 9(4):204, PMC12893702](https://pmc.ncbi.nlm.nih.gov/articles/PMC12893702/) The paper states this for the lab protocol. The FL devices are the same model [INFERENCE].
- Released FL data: R1 has 20 participants and 3,288 h; R2 adds 20 more and 3,162 h. Total 6,450 h (237 participants collected). R3 is announced with 20 more. [paawsstudy.org/data](https://www.paawsstudy.org/data.html)
- R1 FL ages: 12 of 20 aged 18–34, 1 aged 35–54, **7 aged 55+**. Recruitment covered ages 18–80. [PMC12893702](https://pmc.ncbi.nlm.nih.gov/articles/PMC12893702/) R2 ages [UNVERIFIED]: the DRS data summary CSV did not download (HTTP throttling).
- Labels: R1 FL has 1,563.6 h of continuously labelled data, 78.18 ± 11.49 h per participant. [PMC12893702](https://pmc.ncbi.nlm.nih.gov/articles/PMC12893702/) A sleep sub-protocol adds home PSG nights (156 participants, 282 nights in total). [paawsstudy.org/data](https://www.paawsstudy.org/data.html)
- Access: direct download from Northeastern DRS. R1: [hdl 2047/D20806901](https://hdl.handle.net/2047/D20806901), ~111 GB FL per the [GitHub README](https://github.com/mHealth-Research-Group/paaws-study). R2: [collection](https://repository.library.northeastern.edu/collections/neu:gb19f5810), FL zip 112 GB uncompressed. The site warns of 403 errors and institutional login prompts caused by throttling; the team offers to safelist your institution. [paawsstudy.org/data](https://www.paawsstudy.org/data.html) Dataset license [UNVERIFIED]: the paper is CC BY 4.0, but the DRS page shows no data license. Format: CSV [UNVERIFIED]; a loader script exists ([read_accelerometer_data.py](https://github.com/mHealth-Research-Group/paaws-study)).
- Falls: none reported.
- What it answers:
  - False alarms/day on the **left wrist natively**, with no mirroring.
  - A direct test of the mirroring assumption: compare left wrist against mirrored right wrist on the same days.
  - Which activity caused each alarm that survives the stillness check, from the video labels.
  - Night-time alarms, from the PSG nights.

### 2. NHANES 2011–2014 raw 80 Hz (PAX80_G, PAX80_H)
- Device and wear: ActiGraph GT3X+ at 80 Hz on the non-dominant wrist. Participants wore it from the exam day until the morning of day 9. Up to 194 hourly files per participant. Axis values span −6.006 to +6.006 g. No activity logs or diaries were kept. [PAX80_H doc](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm)
- Participants with data: 6,917 (2011–12, ages 6+) and 7,776 (2013–14, ages 3+). About 99% wore it on the non-dominant arm. [PAX80_H doc](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm)
- Ages are top-coded at 80. [DEMO_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.htm) [COMPUTED] by joining PAXHD_G/H (PAXSTS = 1) with DEMO_G/H:

  | Cycle | Aged ≥60 | Aged ≥65 | Aged ≥70 | Aged 80+ |
  |---|---|---|---|---|
  | 2011–12 | 1,510 | 1,038 | 727 | 281 |
  | 2013–14 | 1,552 | 1,114 | 762 | 281 |

  [COMPUTED] PAXHAND = 1 ("worn on non-dominant hand") for 6,856 / 6,917 and 7,707 / 7,776. The left/right side is not coded. That it is mostly the left wrist is an [INFERENCE] from right-handed prevalence.
- Access: public FTP, one tar.bz2 per participant: [ftp.cdc.gov/pub/pax_h](https://ftp.cdc.gov/pub/pax_h/) (and pax_g). Each archive holds hourly `.sensor.csv.gz` files (HEADER_TIMESTAMP, X, Y, Z) plus a QC-flag CSV. Dates are re-mapped to January 2000; day of week is kept. [PAX80_H doc](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm) Use falls under the [NCHS data user agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html).
- Size: CDC states ~1.17 TB for H and ~1.04 TB for G. [COMPUTED] from the FTP listing for H: 7,776 files, 1.297 × 10¹² bytes, median 176 MB per participant (max 273 MB). A subset of 200 participants aged ≥65 is ≈ 35 GB [COMPUTED].
- Falls: none. NCHS's 3–5-year-old raw data is RDC-only ([PAX80_G_R](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAX80_G_R.htm)), which is irrelevant here.
- What it answers:
  - False alarms/day on **thousands of older adults** (60–80+), 24 h × ~7 days each.
  - A different device family (GT3X+) and US population.
  - Alarms split by age stratum.
  - Cannot attribute alarms to activities: no labels beyond NHANES's minute-level wake/sleep/non-wear predictions (PAXMIN).

### 3. BIDSleep Apple Watch (PhysioNet, 2026)
- 47 healthy adults wore an Apple Watch and a Dreem 2 EEG headband at home for up to 7 nights; 253 nights in total. `motion.csv` has a Unix timestamp plus x, y, z in g. Expert-corrected 30-s sleep stages. License ODC-By 1.0. Download is a 5.9 GB zip (27.9 GB uncompressed). [PhysioNet bidsleep-dataset 1.0.1](https://physionet.org/content/bidsleep-dataset/1.0.1/)
- [COMPUTED] Median sample interval 0.0201 s (~50 Hz), measured on the first 400 kB of `Bidslab00/1/motion.csv`.
- Ages, wrist side and range: [UNVERIFIED] (not on the PhysioNet page).
- Falls: none.
- What it answers: night-time false alarms per night on a **consumer smartwatch** at home, across several nights per person, with sleep and wake labels. The stillness check is most likely to pass in bed [INFERENCE], so this is where alarms that survive it may concentrate.

### 4. IMU50 (Zenodo, 2026)
- 50 healthy volunteers wore an ActiGraph LEAP on the **non-dominant** wrist for 5 days and nights in free living, ~136 h each. Accelerometer and gyroscope at 128 Hz. No activity labels. [Zenodo 21468410](https://zenodo.org/records/21468410), CC BY 4.0 per the [Zenodo API](https://zenodo.org/api/records/21468410). 46.7 GB zip. CSV per subject (`XX_imu.csv`).
- **Range set to ±2 g**: "setting its dynamic range to ± 2 g". Ages 40–75, mean 59.1. [La Fratta et al., Data in Brief 2026, PMC13571289](https://pmc.ncbi.nlm.nih.gov/articles/PMC13571289/)
- [COMPUTED] From the paper's Table 8: 27 of 50 aged ≥60, 18 aged ≥65, 8 aged ≥70.
- Red flag: every peak above 2 g is clipped. Our model takes ±4 g input, so the signal cannot be reconstructed, and alarm counts would be biased in an unknown direction [INFERENCE].
- What it answers: a continuous 24 h × 5-day check of the stillness-check stage (which only needs the < 0.02 g std, unaffected by clipping) on older middle-aged adults with a different device.

### 5. UK Biobank accelerometry (field 90001)
- Axivity AX3, 100 Hz, ±8 g, worn on the **dominant wrist** for 7 days; ages 45–79 at wear. [Doherty et al. 2017, PLoS ONE](https://biobank.ndph.ox.ac.uk/showcase/ukb/docs/acc_plos_0169649.pdf)
- 103,568 participants in the main instance. Raw `.cwa` files are ~250 MB each, ~100 MB compressed. [UKB field 90001](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=90001)
- Access: approved UKB application plus access fees, through the Research Analysis Platform. [UKB costs FAQ](https://community.ukbiobank.ac.uk/hc/en-gb/articles/15010425851933-How-much-does-it-cost-to-access-UK-Biobank-data) The fee amounts are in an image I could not read [UNVERIFIED].
- Same device as Capture-24, so it tests a population shift (older), not a device shift. Dominant wrist means mostly right, which needs mirroring [INFERENCE].
- Possible falls: UKB links hospital inpatient records. Fall-coded admissions inside a wear week could provide a handful of real-world wrist falls [INFERENCE; not verified that any exist].

### 6. ExtraSensory (UCSD, 2015–16)
- The app records a **20-second session every minute**. The Pebble watch accelerometer runs at **25 Hz** (~500 × 3 samples per session) in milli-g. 56 users, 210,716 labelled examples with watch accelerometer. 60 users aged 18–42 (mean 24.7). Raw watch-accelerometer zip is 800 MB. The only stated condition is to cite Vaizman 2017a; no license file found. [extrasensory.ucsd.edu](http://extrasensory.ucsd.edu/)
- Watch-wear rule: participants turned the watch app off when they removed the watch, so watch data can be assumed on-wrist. Which wrist is not stated. [Vaizman 2017 supplementary](http://extrasensory.ucsd.edu/papers/vaizman2017Pervasive_supplementary.pdf)
- Pebble accelerometer is calibrated to **±4 g**. [Pebble developer guide](http://developer.repebble.com/guides/events-and-services/accelerometer/)
- [COMPUTED] 210,716 × 20 s ≈ 1,170 h of watch signal.
- Red flags:
  - The 20 s bursts cannot support a 10–60 s stillness check [INFERENCE].
  - Young population.
- What it answers: the CNN-stage false-alarm rate per hour on a **consumer smartwatch at exactly our 25 Hz / ±4 g spec**, with self-reported context labels.

### 7. Sleep-Accel (Walch 2019, PhysioNet)
- 31 subjects with Apple Watch acceleration (g) and PSG labels from one night in the sleep lab. The watch was worn 7–14 days before, but only steps were kept from that period. License ODC-By 1.0. 550 MB zip. [PhysioNet sleep-accel](https://physionet.org/content/sleep-accel/1.0.0/)
- [COMPUTED] Subject 1066528: median sample interval 0.0200 s (~50 Hz), 13.98 h span, max |axis| 4.82 g.
- Wrist side and ages: [UNVERIFIED].
- What it answers: a small night-time consumer-watch check.

### 8. Newcastle PSG + GENEActiv (2015 data, Zenodo 2018)
- 28 sleep-clinic patients, one PSG night each. GENEActiv on **each wrist** (55 `.bin` files; participant 10's right wrist is missing) at 85.7 Hz. `participants_info.csv` holds diagnosis, age and sex. CC BY 4.0. 0.96 GB. [Zenodo 1160410](https://zenodo.org/records/1160410)
- Range: GENEActiv is ±8 g per the [GOTOV device sheet](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/2); not confirmed for this study [UNVERIFIED].
- What it answers: night-time false alarms in patients with sleep disorders (restless), and a left vs mirrored-right comparison. Too small to estimate a rate per day.

### 9. Mobilise-D Technical Validation Study (TVS)
- 108 participants: older healthy adults plus PD, MS, **proximal femoral fracture**, COPD and CHF. Lab tests plus **2.5 h unsupervised free living**. McRoberts MM+ on the lower back; "Some participants additionally wore a custom wrist-IMU at the non-dominant hand". `data.mat` files. **CC BY-NC-ND 4.0**. 58.4 GB in 6 cohort zips. [Zenodo 15861907](https://zenodo.org/records/15861907)
- Wrist IMU rate and range, and how many participants have wrist data: [UNVERIFIED].
- What it answers: false alarms per hour in frail and patient groups (walking aids, hip-fracture rehabilitation). Short recordings; the license prohibits commercial use and derivatives.

### 10. BEAT-PD challenge data (CIS-PD, REAL-PD = Parkinson@Home)
- CIS-PD: Apple Watch Series 2 worn for 6 months with continuous capture. REAL-PD: Motorola Moto 360 Sport on the most affected side for 2 weeks. [Sieberts et al. 2023, PLOS Digit Health, PMC10047543](https://doi.org/10.1371/journal.pdig.0000208)
- The release is **non-overlapping 20-minute segments** around symptom reports, with start time set to 0 to remove time of day. Acceleration in g (CIS-PD) or m/s² (REAL-PD). [Synapse wiki: Data](https://www.synapse.org/#!Synapse:syn20825169/wiki/600898)
- Access: Synapse account, a public Intended Data Use statement, MJFF conditions, and "These data may not be redistributed". [Synapse wiki: Accessing the Data](https://www.synapse.org/#!Synapse:syn20825169/wiki/600903)
- Raw sampling rate [UNVERIFIED].
- Red flags:
  - Fragments cannot give a rate per day.
  - Tremor and dyskinesia are an atypical population.

### 11. GOTOV
- 35 healthy adults aged >60 (mean ≈ 65.6). GENEActiv on the **right wrist**, right ankle and chest. About 1.5 h of 16 activities in a semi-lab setting, including lying left/right 3 min each and sofa sitting. [GOTOV Dataset Details PDF](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/2)
- Sampling rate is internally inconsistent: the figure says 83 Hz, the text says 88 Hz. Range ±8 g.
- **CC BY-NC-SA 4.0**. 1.25 GB. CSV with an activity label column. [4TU v2](https://data.4tu.nl/datasets/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/2)
- Not free living.
- What it answers: per-activity alarm behaviour in older adults, and whether lying still passes the stillness check after transitions.

### Rejected or limited (rows 12–16; red flags)
- **PAMAP2**: 9 subjects, scripted; IMU over the dominant wrist at 100 Hz. [UCI 231](https://archive.ics.uci.edu/dataset/231) The PAAWS paper lists them as 8 men and 1 woman, 27.2 ± 3.3 y. Not free living.
- **WISDM 2019**: 51 subjects, 18 activities × 3 min, smartwatch accelerometer at **20 Hz**. [UCI 507](https://archive.ics.uci.edu/dataset/507) Below 25 Hz, and scripted.
- **LTMM**: 71 community-dwelling adults aged 65–87, 3-day recordings from the **lower back**. Header shows 100 Hz, 3 acceleration and 3 angular-velocity channels. Fall history only. [PhysioNet LTMM](https://physionet.org/content/ltmm/1.0.0/) No wrist.
- **HAR70+**: 18 adults aged 70–95, ~40 min, AX3 on the **right thigh and lower back**, 50 Hz. [UCI 780](https://archive.ics.uci.edu/dataset/780/har70) No wrist.
- **MESA (NSRR)**: Actiwatch Spectrum; files are **30-second epochs of activity counts** plus light. [NSRR MESA actigraphy doc](https://github.com/nsrr/mesa-documentation/blob/master/pages/actigraphy-introduction.md) Counts, not raw signal; unusable.

### Also checked, excluded
- **Physical Activity Monitoring in Older Adults (80+)** (Twente, 2026): MOX on the thigh (25 Hz) and APDM on the lower back; one 75-min session. [Zenodo 18222938](https://zenodo.org/records/18222938) No wrist.
- **Sole-HARmony** (Sci Data 2026): instrumented insoles; article under CC BY-NC-ND, data license [UNVERIFIED]. [doi:10.1038/s41597-026-08172-5](https://www.nature.com/articles/s41597-026-08172-5) No wrist as the primary sensor.
- **FALAH (Vanuatu) GENEActiv**: participants aged 13–17; records restricted or embargoed. [Zenodo 14043332](https://zenodo.org/records/14043332)
- **Upcoming, not yet public**:
  - [OxWEARS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12747664): adults 40+, wrists among 5 sites, 3 days / 4 nights, release planned after 2026 collection ends.
  - [WEALTH](https://wearable-landscape.info/topics/open-accelerometer-datasets): 600+ participants, wrist included, "shared upon request".

### Real-world falls (bonus question): none on the wrist
- **FARSEEING**: 208 verified real-world falls (status end of 2015) from 94 fallers, mean age 76.1. Sensors were on the lower back at L5 (72%) or the thigh (28%), at 100 Hz (73%) or 20 Hz (27%). A 20-fall sample is available on request; full access requires collaboration. [Klenk et al. 2016, PMC5086409](https://pmc.ncbi.nlm.nih.gov/articles/PMC5086409/) No wrist, so it cannot measure the cost of the stillness check on wrist falls.

### Already-used datasets: newer versions?
- **Capture-24**: the [ORA record](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) and the [Sci Data 2024 descriptor](https://www.nature.com/articles/s41597-024-03960-3) still describe 151 participants. No newer version found.
- **FallAllD** (it is in the brief, though not a free-living dataset):
  - License is CC BY 4.0, accelerometer 238 Hz / ±8 g on the wrist, waist and neck, 15 participants, 26,420 files. [IEEE DataPort](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living)
  - The page contradicts itself on access. One part says "Open Access dataset files are accessible to all logged in users… IEEE Membership is not required"; a modal says "This dataset requires an IEEE DataPort Subscription".
  - Per-file recording length: [UNVERIFIED] here.

## Ranked top 3

1. **PAAWS Free-Living R1+R2.** It is the only open dataset with continuous multi-day wrist data on **both wrists**, plus second-by-second video labels. That gives false alarms/day on the native left wrist, a left vs mirrored-right check, and the activity behind every alarm that survives the stillness check. Weaknesses:
   - Only 7 of the 20 R1 participants are aged 55+ (R2 ages unknown).
   - GT9X is a research device, not a smartwatch.
   - DRS throttling may require contacting the team.
   - The data license must be confirmed before any commercial use.
2. **NHANES 2011–14 PAX80.** It has 3,062 participants aged ≥60 and 562 aged 80+ [COMPUTED], each with ~8 days of continuous 80 Hz non-dominant-wrist data. It is free public FTP under the NCHS user agreement, with a ±6 g range that we can clip to ±4 g. It gives a population-level false-alarm rate for older adults, split by age, on a different device than Capture-24. No activity labels; download only a sampled subset (~176 MB per person).
3. **BIDSleep Apple Watch.** It is the only open, multi-night (253 nights), **consumer-smartwatch** raw accelerometer set (~50 Hz) recorded at home, under ODC-By. It tests device shift to a real smartwatch where alarms that survive the stillness check plausibly concentrate (in bed, before or after taking the watch off) [INFERENCE]. Nights only; ages not documented.

Alternates:
- **UK Biobank**: best older-adult coverage plus a possible route to real falls via linked hospital records [INFERENCE]. It costs an application and fees, and uses the same AX3 device as Capture-24.
- **IMU50**: open, 24 h × 5 days, ages 40–75, but ±2 g clipping invalidates CNN-stage counts.
- **ExtraSensory**: matches our 25 Hz / ±4 g spec on a Pebble watch, but the 20 s/min duty cycle rules out the stillness check.
