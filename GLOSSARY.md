# Wrist Fall Detection

Detecting falls from the motion sensors of consumer smartwatches (Apple, Samsung, Huawei, Garmin), using either the vendor's own fall event or a model trained on wrist IMU data.

## Language

### Events and data

**Fall**:
A hard, uncontrolled descent of the wearer to the ground or a lower surface that the detector should report.
_Avoid_: Incident, accident, drop

**Simulated fall**:
A fall performed on purpose by a volunteer, usually onto a mattress. Every public wrist dataset contains only these.
_Avoid_: Fake fall, staged fall

**Impact moment**:
The instant of peak impact within a fall, located from the dataset's annotation or, failing that, from the accelerometer-magnitude peak.
_Avoid_: Fall time, impact point, trigger point

**Fall window**:
A window that contains the impact moment. Other windows cut from a fall recording are excluded from training, not labelled as ADL or fall.
_Avoid_: Positive window, fall segment

**Activity code**:
A dataset's original label for a recording (e.g. a walking or a fall-type code), kept alongside the binary fall/ADL label so errors can be broken down per activity.
_Avoid_: Class label, activity type

**ADL**:
An activity of daily living: any non-fall activity the wearer performs, such as walking, sitting or clapping.
_Avoid_: Non-fall, normal activity, background

**Window**:
A fixed-length slice of the sensor signal that is classified as one sample.
_Avoid_: Segment, frame, chunk

**Canonical format**:
The vendor-independent representation (common units, rate and axis convention) that all device and dataset signals are converted to before windowing.
_Avoid_: Standard format, normalised data

**Wrist frame**:
The axis convention of the canonical format: x along the forearm toward the fingers, z out of the watch face, and y completing a right-handed frame (toward the little-finger side on the left wrist), so a watch lying flat and face up reads +1 g on z. All data is expressed as if worn on the left wrist.
_Avoid_: Sensor frame, device axes, body frame

**Mounting alignment**:
Rotating one wearer's signal into the wrist frame, using the average gravity direction during a reference activity (standing or walking) and mirroring right-wrist data to the left-wrist convention. Without activity labels the walking is found from the signal itself (a step rhythm in |a|). A wearer whose reference average is too weak to show a direction is excluded rather than guessed.
_Avoid_: Calibration, orientation correction

**Free-living recording**:
Unscripted, continuous wear in daily life (a day or more, sleep included), with no falls marked. Used to measure false alarms and as a source of training ADL.
_Avoid_: Real-world data, background data, wild data

**Non-wear time**:
Recorded time when the device was off the wrist, found from the signal (an hour with almost no variation on at least two axes). Excluded from exposure and from detections, as a watch with off-wrist detection would not alert then.
_Avoid_: Idle time, missing data

### Detection quality

**Alarm event**:
One alert the detector would show the wearer. Detections closer together than the refractory period form a single alarm event; its alarm time is when the device would alert (for a confirmed fall, after the confirmation context has been seen).
_Avoid_: Trigger, detection (for the merged alert)

**Refractory period**:
The time after an alarm event during which further detections are not shown as new alarms.
_Avoid_: Cooldown, debounce

**Stillness check**:
Holding a detection until the wrist has stayed still for a wait period after the detector's decision; only detections that pass become alarm events. It delays the alert by the wait and misses falls after which the wearer moves.
_Avoid_: Immobility timer, post-fall confirmation (stage 2 is the confirmation)

**False alarm**:
An alarm event while the wearer did not fall. Rates are stated per day of exposure.
_Avoid_: False positive (in user-facing discussion); counting fired windows as false alarms

**Exposure**:
The wear time over which false alarms are counted, the denominator of a false-alarm rate.
_Avoid_: Duration, recording length

**Missed fall**:
A fall the detector fails to report.
_Avoid_: False negative (in user-facing discussion)

**Hard negative**:
An ADL window that the model scores as a fall, typically a high-impact wrist motion such as clapping or slamming a table.
_Avoid_: Difficult sample, adversarial sample

**Hard negative mining**:
Collecting hard negatives after a training pass and adding them to training to reduce false alarms. It does not reduce missed falls.
_Avoid_: Negative mining, error mining

### Models and evaluation

**Baseline**:
The first end-to-end model that every later model is measured against, trained and evaluated on the same splits.
_Avoid_: Prototype, v1, benchmark model

**Threshold reference**:
A hand-written rule detector (free-fall dip followed by an impact peak) evaluated on the same splits as the baseline, to show whether the learned model earns its complexity.
_Avoid_: Heuristic baseline, rule baseline

**Subject-wise split**:
A train/validation division in which no person appears on both sides.
_Avoid_: Person split, user split

**Leave-one-dataset-out**:
An evaluation that trains on all datasets but one and tests on the held-out dataset; the headline measure of generalisation.
_Avoid_: Cross-dataset test, LODO

**Adaptation**:
Further training of a base model on a few volunteers recorded on the target device, so it fits that device's signal.
_Avoid_: Fine-tuning (as the general term), calibration, personalisation

**Calibration set**:
Target-device volunteers held apart from adaptation, used to choose when adaptation stops and where the decision threshold sits.
_Avoid_: Validation set (for the target device), tuning set
