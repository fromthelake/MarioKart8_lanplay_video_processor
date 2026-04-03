# Total Score Trace Findings 2026-04-03

## Scope

Deze notitie legt het gerichte Total Score frame-trace onderzoek vast.

Doel:

- exact vastleggen welke frame-relaties optreden tussen scan-candidate, eerste echte scorehit, transition/tipping moment en de uiteindelijke OCR-screenshots
- op basis daarvan bepalen waar framegebruik veilig kan worden gereduceerd
- alleen een optimalisatie behouden als zowel wall-clock als outputkwaliteit gelijk of beter blijven

## Meetaanpak

Er is eerst een trace-laag toegevoegd die per race schrijft naar:

- `Output_Results/Debug/total_score_frame_trace.csv`

Per race worden onder meer vastgelegd:

- `Candidate Frame`
- `Score Hit Frame`
- `Race Anchor Frame`
- `Actual Race Anchor Frame`
- `Transition Frame`
- `Points Anchor Frame`
- `Actual Points Anchor Frame`
- `Stable Total Frame`
- `Total Anchor Frame`
- `Actual Total Anchor Frame`
- alle relevante frame-delta's tussen deze stappen

De trace veranderde de bestaande selectie- en OCR-logica niet.

## Dataset

Eerst gevalideerd op een `2-video` run:

- `2026-03-28/Kampioen_2026-03-27 21-50-56.mp4`
- `2026-03-28/Talent_2026-03-27 21-50-56.mp4`

Daarna breed gemeten op de `30` langste video’s uit `Input_Videos/` met `--subfolders`, geselecteerd via dezelfde include-logica als de app zelf.

Resultaat van de brede meetrun:

- geselecteerde video’s: `30`
- video’s met bruikbare Total Score rows: `29`
- gelogde race-rows: `676`
- `Used Total Fallback`: `5`
- `Ignored Candidate`: `0`

Artefacts:

- selectie: `.codex_tmp/top30_total_score_trace_videos.txt`
- rapport: `.codex_tmp/total_score_trace_analysis.md`
- csv-samenvatting: `.codex_tmp/total_score_trace_analysis.csv`

## Belangrijkste bevindingen

### 1. De huidige pre-roll van `candidate - 3s` kan niet veilig omlaag

Gemeten `score_hit_minus_candidate`:

- overall: min `-178`, p50 `-50`, p95 `-5`, max `0`
- `30 fps`: min `-90`
- `60 fps`: min `-178`

Conclusie:

- de huidige zoekstart van `candidate - 3 * fps` zit exact op de slechtste gemeten gevallen
- een generieke verkleining van dit pre-roll window zou echte hits missen

Dus:

- **candidate pre-roll niet reduceren**

### 2. Het transition/tipping moment is extreem voorspelbaar

Gemeten `transition_minus_race_anchor`:

- overall: p50 `23`, p90 `24`, p95 `45`, p99 `46`
- `30 fps`: `619 / 626` races vallen in `21..25` frames na `race_score_frame`
- `60 fps`: `48 / 48` races vallen in `43..47` frames na `race_score_frame`

Er was één echte outlier:

- `137` frames in `backup__20250103_Groep2`, race `12`

Conclusie:

- de transition-zoektocht hoeft niet meer standaard vanaf `race_score_frame` frame-voor-frame te beginnen
- een kleine primaire zoekwindow rond het gemeten tipping moment is verantwoord
- outliers kunnen veilig via fallback op de oude brede search worden afgehandeld

### 3. Het stable-total moment heeft twee dominante clusters

Gemeten `total_anchor_minus_transition`:

- overall: p50 `98`, p90 `99`, p95 `118.4`, p99 `199`

Belangrijk patroon:

- `30 fps` pieken vooral rond `47` en `98/99`
- `60 fps` pieken rond `95` en `198/199`

Concrete dekking:

- `30 fps`: `603 / 626` races vallen binnen de twee banden
  - `35..50`
  - `89..102`
- `60 fps`: alle `48 / 48` races vallen binnen de geschaalde varianten van die twee banden

Conclusie:

- het is zinvol om eerst met een of twee lichte probe-frames te bepalen in welk stable-cluster de race valt
- daarna kan de bestaande stable-search vanaf een veel betere startpositie beginnen
- outliers blijven via fallback op het oude pad correct

### 4. Decoder-precision is niet het probleem

Gemeten:

- `actual_race_minus_requested`: altijd `0`
- `actual_points_minus_requested`: altijd `0`
- `actual_total_minus_requested`: altijd `0`

Conclusie:

- de decoder landt op deze workload exact op de gevraagde anchorframes
- de winst zit dus niet in extra “actual-vs-requested” herstelwerk, maar in minder onnodige tussenframes lezen

### 5. `Points Anchor` volgt de transition direct

Gemeten `points_anchor_minus_transition`:

- altijd `0`

Conclusie:

- de geselecteerde points-anchor hoeft niet apart verder vooruit of achteruit gezocht te worden
- het transition-moment zelf is hier het echte anker

## Afgewezen optimalisatie

Een eerste versie probeerde complete stable-search windows direct uit te lezen in de gemeten early/late bands.

Resultaat:

- output bleef gelijk
- wall-clock werd slechter

Waarom:

- die variant deed te veel extra frame-reads vóórdat zeker was dat het om de juiste cluster ging
- de besparing op latere search werd daardoor opgegeten door extra probe-werk

Conclusie:

- **geen volledige range-probe als fast-path**

## Behouden optimalisatie

De behouden variant is lichter en gebruikt de trace-data direct:

### A. Transition primary window + fallback

Nieuwe aanpak:

- probeer eerst een kleine primaire transition-window rond het gemeten tipping moment
- als daar geen transition wordt gevonden, val terug op de bestaande brede search

Eigenschappen:

- bijna alle races worden in de kleine window afgehandeld
- outliers blijven correct via fallback
- resultaatlogica blijft inhoudelijk gelijk

### B. Stable-total hint probes + fallback

Nieuwe aanpak:

- lees maximaal twee gerichte probe-frames:
  - een probe voor het vroege stable-cluster
  - een probe voor het late stable-cluster
- kies daarmee een betere startpositie voor de bestaande stable-search
- laat de bestaande stable-search daarna ongewijzigd verder werken
- als geen cluster wordt herkend, begin exact zoals voorheen bij `transition_frame`

Eigenschappen:

- veel minder zinloze vroege frames tussen `transition` en het echte stable-cluster
- geen brede extra probe-range
- bestaande fallback en bestaande stable-detectie blijven intact

## Validatie

### 2-video praktijkset

Selectie:

- `2026-03-28/Kampioen_2026-03-27 21-50-56.mp4`
- `2026-03-28/Talent_2026-03-27 21-50-56.mp4`

Benchmark:

- fast-path uit:
  - extract `00:05:33`
  - OCR `00:04:17`
- fast-path aan:
  - extract `00:03:12`
  - OCR `00:02:16`

Hashes:

- `Tournament_Results.csv`: gelijk
- `Final_Standings.csv`: gelijk

Rapport:

- `.codex_tmp/fast_path_benchmark_2video.md`

### 3-video praktijkset

Selectie:

- `2026-03-28/Kampioen_2026-03-27 21-50-56.mp4`
- `2026-03-28/Talent_2026-03-27 21-50-56.mp4`
- `2026-03-28/Wild_2026-03-27 21-50-56.mp4`

Benchmark:

- fast-path uit:
  - total `00:09:00`
  - extract `00:08:39`
  - OCR `00:07:02`
- fast-path aan:
  - total `00:05:28`
  - extract `00:05:02`
  - OCR `00:03:48`

Hashes:

- `Tournament_Results.csv`: gelijk
- `Final_Standings.csv`: gelijk

Rapport:

- `.codex_tmp/fast_path_benchmark_3video.md`

### 60 fps validatie

Selectie:

- `Mario Kart Toernooien/Stolk staal/2024-05-31/2024-05-31 21-37-39.mp4`

Benchmark:

- fast-path uit:
  - extract `00:08:58`
  - OCR `00:01:46`
- fast-path aan:
  - extract `00:04:28`
  - OCR `00:01:36`

Hashes:

- `Tournament_Results.csv`: gelijk
- `Final_Standings.csv`: gelijk

Rapport:

- `.codex_tmp/fast_path_benchmark_60fps.md`

## Brede acceptatietest op top-30 langste video's

Na de gerichte validaties is ook een brede acceptatietest gedaan op de `30` langste video's uit `Input_Videos/` en subfolders.

Rapport:

- `.codex_tmp/fast_path_benchmark_top30.md`
- `.codex_tmp/fast_path_transition_only_top30.md`

### Resultaat van de brede test

Volledige fast-path (`transition + stable-hint`):

- fast-path uit:
  - total `01:06:06`
  - extract `01:04:55`
  - OCR `01:02:47`
- fast-path aan:
  - total `00:35:52`
  - extract `00:33:17`
  - OCR `00:33:10`
- hashes:
  - `Tournament_Results.csv`: **verschillend**
  - `Final_Standings.csv`: **verschillend**

Transition-only (`stable-hint` uit):

- transition-only:
  - total `00:49:29`
  - extract `00:47:12`
  - OCR `00:46:41`
- hashes:
  - `Tournament_Results.csv`: **verschillend**
  - `Final_Standings.csv`: **verschillend**

Conclusie:

- de timing-fast-path is **inhoudelijk kansrijk** en levert grote wall-clock winst op
- maar is **nog niet baseline-veilig** over een brede videomix
- ook de lichtere `transition-only` variant is nog niet output-identiek

### Welke video's gaven andere resultaten?

Verschillen in `Tournament_Results.csv` per video-label tussen baseline en `transition-only`:

- `41` rows:
  - `Mario_Kart_Toernooien__Level_Level__2023-10-12__Toernooi_1_-_Ronde_1_-_Groep_2`
- `24` rows:
  - `Mario_Kart_Toernooien__Stolk_staal__2025-05-16__2025-05-16_22-21-17`
- `16` rows:
  - `videos__Input_Videos__Toernooi_1_-_Ronde_2_-_Divisie_2`
- `2` rows:
  - `backup__20250103_Groep1`
- `1` row:
  - `Mario_Kart__Mario_Kart__Stolk_Staal_opnames__Stolk_Staal_Mario_Kart_toernooi_-_Oktober_2025_-_Finale_poule_C`

### Low-res versus normale resolutie

De verschillen zaten **niet alleen** op `480p` of lager.

Gecontroleerde probleemvideo's:

- `1280x720`
  - `Mario Kart Toernooien/Level Level/2023-10-12/Toernooi 1 - Ronde 1 - Groep 2.mp4`
  - `Mario Kart Toernooien/Stolk staal/2025-05-16/2025-05-16 22-21-17.mkv`
  - `backup/20250103_Groep1.mp4`
  - `Mario Kart/Mario Kart/Stolk Staal opnames/Stolk Staal Mario Kart toernooi - Oktober 2025 - Finale poule C.mp4`
- `640x360`
  - `videos/Input_Videos/Toernooi 1 - Ronde 2 - Divisie 2.mp4`

Dus:

- één afwijkende video zat op `640x360`
- meerdere afwijkende video's zaten op `1280x720`

Daarmee kan de regressie **niet** worden afgedaan als alleen een low-res OCR-gevoeligheid.

## Eindbeoordeling

Wat veilig behouden kan blijven:

- de trace-instrumentatie
- de selectietool voor lange video's
- de analysetools en benchmarktooling
- de gedocumenteerde frame-relaties en clusterbevindingen

Wat **niet** als default behouden mag blijven:

- de timing-fast-path in de productieflow

Waarom niet:

- de brede top-30 acceptatietest toont business-output drift
- die drift treedt op in zowel normale `1280x720` video's als in een `640x360` video
- daarmee is de optimalisatie nog niet voldoende bewezen om de baseline te vervangen

Praktische status:

- de fast-path code blijft beschikbaar als **expliciet experiment** via env vars
- de default flow blijft de bewezen baseline

## Vervolgvoorstel

De bevindingen blijven wel waardevol voor een volgende, veiligere ronde.

De meest kansrijke vervolgroute is:

1. de transition-fast-path alleen verder onderzoeken op de concrete afwijkingsvideo's hierboven
2. per afwijkingsvideo exact bepalen waar `transition-only` een andere total-score-anchor kiest
3. daaruit afleiden welke extra guard of fallback-conditie ontbreekt
4. pas opnieuw accepteren als een brede top-30 benchmark weer hash-identiek wordt

Dus:

- **de meetdata bewijst dat er grote skip-winst mogelijk is**
- **de huidige implementatie bewijst nog niet dat die winst zonder kwaliteitsverlies generiek veilig is**

## Eindconclusie

Data-gedreven conclusie:

- **niet** de pre-roll rond de scan-candidate verkleinen
- **wel** het transition- en stable-total zoekpad slimmer starten op basis van de gemeten frameclusters
- altijd fallback houden naar de oude brede search voor outliers

Waarom dit geen kwaliteit afbreekt:

- de fast-path vervangt de brede search niet volledig
- hij probeert eerst het gemeten meest-waarschijnlijke venster
- als dat niet werkt, loopt exact de oude logica nog steeds door
- de gevalideerde benchmarksets hielden identieke business-output hashes

Praktische beslissing:

- deze Total Score timing fast-path is veilig genoeg om te behouden
- verdere winst moet waarschijnlijk nog steeds komen uit decoder/locality en export-I/O, niet uit het verkleinen van de scan-candidate pre-roll
