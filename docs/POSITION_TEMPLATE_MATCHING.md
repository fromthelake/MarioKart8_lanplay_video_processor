# Position Template Matching

This note explains how the left-side position number check works on score screens.

## Goal

The position template check is used as the main score-presence signal for score-screen detection and as an extra visual signal for player-count detection.

It helps answer:

- Does this scoreboard row show a real rank number?
- Is this lower row likely empty noise?

It does **not** replace the rest of OCR by itself, but it now drives:

- initial score-screen detection
- RaceScore player-count recovery
- TotalScore drop confirmation

## Current ROI

The default path now uses the black/white tile templates, not the old strip-derived row slices.

- Tile size: `52 x 52`
- Match ROI size: `52 x 52`
- The tile ROI is anchored from the same left-side position strip and shifted into the fixed tile grid used by the templates
- Each lower row advances by `52 px`

## Template source

The default position templates are loaded from:

- `assets/templates/Score_template_white.png`
- `assets/templates/Score_template_black.png`

The matcher tries:
- white tile first
- black tile second
- then keeps the better masked match score for that row

## Metrics

For every row ROI, the code compares that row against the matching row number and the monotone fallback candidate from both tile sets.

The report shows three scores:

### 1. Coeff

This is the OpenCV normalized correlation score:

- `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`

This is useful as a broad visual similarity score.

### 2. White IoU

This only looks at the foreground digit shape:

- template white on ROI white = good
- template white on ROI black = bad
- template black on ROI white = bad
- template black on ROI black = ignored

Formula:

- `TP / (TP + FP + FN)`

### 3. Weighted White IoU

This is the same as White IoU, but it penalizes missing template strokes harder.

That means:

- template white on ROI black counts heavier than template black on ROI white

Current weight:

- `FN weight = 2.0`

Formula:

- `TP / (TP + FP + 2 * FN)`

## Why black-on-black is not a main score

If black-on-black is rewarded too strongly, empty rows can score artificially well just because most of the background is dark.

That is why the foreground-based IoU scores are more useful than plain accuracy for this task.

## Official method

After testing the current 10-player and 12-player reference cases, the official method now has two separate steps:

1. row presence gate
2. template choice inside a present row

### Step 1. Row presence gate

The row is treated as present only when:

- `Coeff >= 0.50`

Anything below that threshold is treated as empty.

This is intentionally simple and strict. It prevents weak OCR noise on the lower rows from keeping those rows alive.

### Step 2. Template choice inside a present row

Once a row passed the presence gate, the official template-ranking method is:

1. evaluate the expected row rank and the monotone fallback rank
2. for each candidate, compare both the white and black tiles
3. keep the stronger masked match for that candidate
4. keep the best monotone-consistent candidate overall

This is now implemented as the official position-guided template ranking.

### Step 3. Monotone fallback

Rows may stay equal or increase as the table goes down.

That means:

- lower than the row above is not allowed
- equal to the row above is allowed
- higher than the row above is allowed

If the preferred template would drop below the row above, the code now tries the next
logical high-coefficient candidate first instead of stopping immediately.

This is mainly useful for close-shape neighbours such as `3` and `8`.

### Step 4. Tie-aware prefix acceptance

For some TotalScore checks, the code intentionally allows tied rank numbers.

That means the row is still acceptable when:

- the shown rank does not go down compared with the row above
- the shown rank is not larger than the row number itself

Examples:

- row 2 may show `1` or `2`
- row 3 may show `1`, `2`, or `3`
- row 6 may show any non-decreasing rank in `1..6`

This tie-aware rule is used for confirming when the RaceScore signal has really disappeared,
so temporary transition frames and tied totals do not create a false TotalScore drop.

## Current status

The stricter ROI and the new template slicing made the row-level results much stronger.

On the tested 10-player and 12-player reference screenshots:

- real rows `1..10` are now strong and stable
- true 12-player rows `11` and `12` are strong
- empty 10-player rows `11` and `12` are much weaker

The one remaining mismatch from the raw test table turned out to be a real tie on the
`TotalScore` screen: row 8 and row 9 both showed place `8`. That means the stricter
position method is consistent with the corrected expectation.

The position-template method is now the official row-count guide, with the older OCR-only
count retained only as a legacy debug reference.

Additional guard in the row-count path:

- non-finite template-match scores (`inf` / `NaN`) are treated as invalid row support
- this prevents malformed match values from creating phantom extra players during vote/recovery

For initial score detection, the same left-side row boxes now act as the score trigger:

- rows `1..N` must all match their own rank
- each row must clear the configured row floor
- the prefix average must clear the configured average floor

The current 10-player reference video now validates cleanly with this rule set:

- all 11 races resolve to `10 / 10`
- the player-count summary is fully consistent

## Late-frame recovery

Some `2RaceScore` frames briefly hide the last row behind the black "results are in" banner.

When the selected RaceScore frame looks suspicious, the OCR layer now:

1. keeps the normal position-template count as the first pass
2. checks a slightly later RaceScore frame, usually `+3`
3. accepts that later count only when it is clearly better

This recovery only changes the RaceScore player-count decision. It does not replace the
whole OCR payload for the race.

## TotalScore drop confirmation

TotalScore selection no longer treats a single missing score frame as the end of the
RaceScore phase.

Instead the selector now:

1. watches for a sustained drop in the score signal
2. uses tie-aware rank acceptance on rows `1..6`
3. confirms the drop only after `5.0 * fps` worth of continuous absence
4. anchors TotalScore from the start of that confirmed drop, then applies the existing
   `-2.7s` timing offset

This prevents short transition animations from triggering an early `3TotalScore` frame.

## RaceScore frame split

RaceScore OCR now uses different frame subsets for different signals inside the same
7-frame consensus bundle.

Current split:

- player names: all 7 nearby frames
- race points: first 3 frames
- character icons: last 3 frames
- left-side position templates: last 3 frames
- RaceScore player count: first 3 frames

Why this split exists:

- early RaceScore frames keep the points table clean before later drift starts pulling
  values downward
- late RaceScore frames are better for stable character icons and left-side rank shapes
- names benefit most from the larger 7-frame vote

This was validated directly on `Divisie_2` race 1:

- frames `8299..8302` read the correct race points
- frames `8303..8305` drifted downward by `-1` and then `-2`

So race points now use the early clean subset, while later frames are reserved for the
signals that visibly improve later in the score-screen animation.
