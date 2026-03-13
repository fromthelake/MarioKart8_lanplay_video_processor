# Position Template Matching

This note explains how the left-side position number check works on score screens.

## Goal

The position template check is used as an extra visual signal for player-count detection.

It helps answer:

- Does this scoreboard row show a real rank number?
- Is this lower row likely empty noise?

It does **not** replace the other OCR signals by itself.

## Current ROI

Each row uses the same window structure as `Score_template.png`.

- Template size: `56 x 36`
- Match ROI size: `58 x 41`
- Horizontal padding: `1 px` left and `1 px` right
- Vertical padding: `1 px` above and `4 px` below

The row starts come from the fixed template slicing:

- `0, 50, 102, 154, 206, 258, 310, 362, 414, 466, 518, 570`

## Template source

The current position templates are sliced from:

- `assets/templates/Score_template.png`

Each exported `template_row_*.png` is `56 x 36`.

## Metrics

For every row ROI, the code compares that row against all 12 `template_row_*.png` images.

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

- `Coeff >= 0.60`

Anything below that threshold is treated as empty.

This is intentionally simple and strict. It prevents weak OCR noise on the lower rows from keeping those rows alive.

### Step 2. Template choice inside a present row

Once a row passed the presence gate, the official template-ranking method is:

1. shortlist the top 3 templates by `Coeff`
2. choose the best of those 3 by `Weighted White IoU`

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
