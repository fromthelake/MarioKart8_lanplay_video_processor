# Performance Findings 2026-04-03

## Scope

Deze notitie legt de worker- en remux-benchmarks vast die zijn uitgevoerd op de huidige codebase, zodat hetzelfde onderzoek later niet opnieuw hoeft te worden gedaan zonder nieuwe aanleiding.

Doel van deze ronde:

- bepalen of tijdelijke remux-partitionering echte wall-clock winst geeft
- bepalen welke non-remux worker-configuratie het beste werkt voor praktische `2`-video en `3`-video runs
- vastleggen waar de bottleneck werkelijk zit op de beschikbare hardware

Hardware waarop deze conclusies gebaseerd zijn:

- ASUS ROG Strix G18 (G815LW-S9076W)
- Windows 11
- Intel Core Ultra 9 275HX, `24` cores / `24` threads
- NVIDIA RTX 5080 Laptop GPU, `16 GB` VRAM
- `64 GB` DDR5-5600 RAM
- Samsung 9100 Pro 4 TB PCIe 5.0 NVMe SSD

## Test Sets

Gebruikte benchmarksets:

- `2-video`
  - `2026-03-28/Kwalificatie_Groep_1_2026-03-27 20-00-33.mkv`
  - `2026-03-28/Kwalificatie_Groep_2_2026-03-27 20-00-33.mp4`
- `3-video`
  - bovenstaande twee
  - `2026-03-28/Kwalificatie_Groep_3_2026-03-27 20-00-33.mkv`

Business-outputcontrole:

- `2-video`
  - `Tournament_Results.csv`: `02feb39e6e6c768876b232a32224fa0bd0bc058697de5a2a951b0f52a879cde3`
  - `Final_Standings.csv`: `5d64b7d70591992fddd8d376dbc89675fc29d2767e321270669f5e850f00851e`
- `3-video`
  - `Tournament_Results.csv`: `33ab05e6788913c4c41bd3c8aae038df1817534d8caf74f1d5ed7f75ed18bc0e`
  - `Final_Standings.csv`: `a737a9357b69ebd4782631d9da6dcc00e833e5efb31421165d7705b4027f2206`

## Wat Is Getest

### 1. Non-remux worker sweep

Voor `baseline` zonder remux is getest:

- `ScanW`: `2`, `3`, `4`
- `ScoreW`: `4`, `6`, `8`, `10`, `12`

### 2. Remux modes

Getest als experimenteel benchmarkpad:

- `scan` remux
- `score-workers` remux
- `score-races` remux
- `scan + score-workers`
- `scan + score-races`

### 3. Follow-up runs

Na de matrix zijn extra gerichte runs gedaan voor:

- `2-video score-races 3/6`
- `3-video score-races 3/6`
- `3-video baseline 3/4`
- `3-video baseline 3/6`
- extra A/B bevestiging op `3-video baseline 2/4` versus `3-video baseline 3/4`

### 4. Benchmark artefacts

Voor deze ronde is tijdelijke benchmarktooling gebruikt om matrix-runs en logparsing te automatiseren. Die tooling is niet als blijvend productpad behouden, maar de bevindingen en ruwe artefacts zijn wel vastgelegd.

De gebruikte parser/logica verzamelde:

- wall-clock
- extract- en OCR-duur
- seek/read/grab/export buckets
- hash-match
- resource peaks
- bottleneckclassificatie

Ruwe benchmarkartefacts zijn lokaal terug te vinden onder:

- `.codex_tmp/remux_matrix_full/`
- `.codex_tmp/remux_matrix_followup/`
- `.codex_tmp/remux_matrix_confirm_2video/`
- `.codex_tmp/remux_matrix_confirm_3video/`
- `.codex_tmp/remux_matrix_confirm_3video_ab/`

## Onderzoeksgeschiedenis Voor De Remux-Matrix

Deze benchmarkronde kwam niet uit het niets. Hieronder staat ook vastgelegd welke eerdere performance-routes al onderzocht zijn, zodat die niet opnieuw als “verse hypothese” terugkomen.

### Eerdere winstgevende stappen

Deze stappen gaven lokaal meetbare winst en zijn daarom eerder als checkpoint of commit vastgelegd:

```text
Commit     Korte omschrijving                                   Resultaat
---------  ----------------------------------------------------  --------------------------------------------
6490406    head/tail clamp + remux-first corrupt handling       veilige rollbackbasis
edffc31    eerdere multi-video throughput-refactor              vroege overlapverbetering
a4dbdcc    short forward jumps liever grab() dan seek()         duidelijke winst in score-selectie
dd8e9b1    consensus frame windows hergebruiken                 kleine extra winst
464da18    redundante bundle-cleanup overslaan op fresh runs    kleine winst
9519c2a    visible row count eerder berekenen                   bracht runtime weer terug richting beste punt
596c1f9    OCR queueing losgekoppeld van ordered score flush    architectonisch beter, output gelijk
c9a93a1    chunked worker locality voor TotalScore              beste eerdere 3-video run rond 00:03:26
```

Belangrijke nuance:

- niet alle oudere tussenmetingen zijn 1-op-1 vergelijkbaar met de latere matrix
- de code en instrumentatie zijn onderweg veranderd
- de latere matrix- en bevestigingsruns zijn daarom leidend voor de uiteindelijke defaultkeuze

### Eerdere afgewezen experimenten

Deze experimenten zijn expliciet getest en gaven geen netto waarde:

```text
Experiment                                           Uitkomst
---------------------------------------------------  ---------------------------------------------------------
3 scans tegelijk                                      slechter; meer decoder/contention
worker-local capture reuse met minder workers         seekafstand daalde, wall-clock werd slechter
static contiguous partitioning (Variant A)            betere locality, maar straggler-chunks maakten run trager
scan remux als algemene oplossing                     geen overtuigende winst
score remux als default                               niet stabiel sneller op echte wall-clock
scan + score remux gecombineerd                       meestal duidelijk slechter
bredere per-race frame window vooraf in RAM           juist trager
punten-anchor capture schrappen                       geen netto winst
lock versmallen rond image writes                     output gelijk, maar runtime slechter
PyAV backend spike                                    trager én output-afwijking
hogere ScoreW (`8/10/12`) als algemene richting       meestal slechter
```

### Instrumentatie die expliciet is toegevoegd

Voor dit onderzoek zijn ook meetlagen toegevoegd om niet op gevoel te sturen:

- capture overlap / duplicate frame reads
- same-run versus persisted OCR framegebruik
- lock wait tijd
- out-of-order score backlog
- seek hotspots per label
- capture positioning patronen
- remux partition counts en creation time
- worker-local capture opens

Deze instrumentatie was nodig om vast te stellen dat de hoofdpijn in deze codebase niet primair uit OCR of locks kwam, maar uit decoder/media access.

## Hoofdbevindingen

### Bottleneck

De dominante bottleneck bleef in vrijwel alle relevante runs:

- `decoder / media access`

Praktisch betekent dat:

- de machine heeft wel ruwe CPU-capaciteit
- de workload schaalt niet als simpele CPU-bound batchjob
- de pijn zit vooral in OpenCV/FFmpeg decode-locality, seeks, reads, grabs en frame-export

Niet de hoofdbeperking in deze ronde:

- OCR overlap
- Python-locking
- pure CPU-threadschaarste
- scan-remux boundarylogica

### Remux conclusie

Remux niet als default pad promoten.

Waarom:

- scan-remux gaf geen consistente wall-clock winst
- TotalScore remux verlaagde soms seek-hotspots, maar won niet stabiel op echte wall-clock
- gecombineerde remux-modi waren meestal slechter
- de extra ffmpeg-partitionering en I/O compenseerden de locality-winst niet genoeg

### Worker conclusie

De beste en veiligste generieke default uit deze ronde is:

- `pass1_scan_workers = 2`
- `score_analysis_workers = 4`

Reden:

- `2-video` laat duidelijk zien dat `2/4` het beste non-remux resultaat geeft
- `3-video` gaf op sommige losse runs ook sterke resultaten voor `2/6` of `3/4`, maar niet stabiel genoeg
- `2/6` gaf in de bevestiging zelfs hash-drift op `2-video`
- `3-video 2/6` had in de bevestiging een mislukte run
- extra A/B op `3-video` liet uiteindelijk `2/4` weer beter uitkomen dan `3/4`

### Stabiliteit en run-to-run variatie

Een belangrijke observatie uit de bevestigingsronde:

- sommige losse topmetingen bleken niet stabiel genoeg om default op te baseren
- vooral `3-video 2/6` en `3-video 3/4` gaven wisselende resultaten tussen runs
- daarom is niet het absolute snelste losse cijfer leidend geweest, maar de combinatie van:
  - wall-clock
  - hash-consistentie
  - afwezigheid van mislukte runs
  - herhaalbaarheid

## Belangrijkste Resultaten

### Beste non-remux resultaten

```text
Scenario   Config   Wall(s)   Opmerking
---------  -------  --------  -----------------------------------------------
2-video    2/4      148.72    beste non-remux resultaat in de matrix
3-video    2/6      213.02    sterke losse run, maar later niet stabiel genoeg
3-video    3/4      220.46    goede losse bevestigingsrun
3-video    2/4      227.13    won in de directe A/B-herhaling tegen 3/4
```

### Gerichte bevestigingsruns

```text
Batch                          Config   Wall(s)   Opmerking
-----------------------------  -------  --------  ------------------------------------------------
2-video confirm               2/4      178.19    stabiel, hash ok
2-video confirm               2/6      182.95    hash-afwijking op Tournament_Results.csv
3-video confirm               2/6      248.73    mislukte run, geen outputpad/ocr-complete samenvatting
3-video confirm               2/4      228.34    geldig
3-video confirm               3/4      220.46    geldig, maar later niet beter in directe A/B
3-video confirm               3/6      220.91    geldig, maar slechter dan 3/4 in die batch
3-video direct A/B rerun      2/4      227.13    won van 3/4 in die herhaling
3-video direct A/B rerun      3/4      229.96    verloor van 2/4 in die herhaling
```

### Matrix-brede observaties

Uit de volledige matrix kwamen deze patronen duidelijk terug:

- `2-video`
  - beste non-remux lag bij `2/4`
  - `3/4` zat dichtbij, maar was niet beter
  - remux op `score-races` zat soms in de buurt, maar won niet
- `3-video`
  - `2/6` gaf één sterke losse topmeting
  - `2/4`, `3/4` en `3/6` lagen dichter bij elkaar dan bij `2-video`
  - na bevestigingsruns bleek geen van de alternatieven stabiel genoeg om `2/4` als generieke default te verslaan

### Gerichte follow-up voor score-remux op races

```text
Case                      ScanW  ScoreW  ScoreRemux  Wall(s)   Resultaat
------------------------  -----  ------  ----------  --------  -----------------------------
2-video best no-remux         2       4  off         148.72    referentie
2-video requested             3       6  races       156.69    slechter dan 2/4
3-video best no-remux*        2       6  off         213.02    losse matrix-toprun
3-video requested             3       6  races       229.04    duidelijk slechter
```

`*` Die `3-video 2/6` run was later niet stabiel herhaalbaar genoeg om default te worden.

### Belangrijkste praktische conclusie

Wat wél werkt:

- non-remux worker-tuning

Wat géén overtuigende waarde toevoegde:

- scan-remux
- score-remux als default
- hogere `ScoreW` zoals `8`, `10`, `12`

## Wat We Niet Nog Eens Hoeven Te Proberen Zonder Nieuwe Aanleiding

Deze paden zijn in deze ronde al voldoende onderzocht en hoeven niet opnieuw als “eerste gok” terug te komen:

- scan-remux als algemene performance-oplossing
- score-remux als default processing pad
- gecombineerde scan+score remux-modi
- simpelweg `meer workers` omhoog blijven zetten naar `8/10/12`
- redeneren vanuit “24 threads dus meer workers moet altijd sneller zijn”
- PyAV als eerste performance-route voor deze pipeline
- worker-aantallen afleiden uit `cpu_threads - 1`

Alleen opnieuw openen als er nieuwe code of een nieuwe backend komt die één van deze randvoorwaarden wezenlijk verandert:

- andere decode/backendstrategie
- andere exportstrategie
- wezenlijk andere TotalScore scheduling
- nieuw type inputvideo

## Aanbeveling Voor Verdere Optimalisatie

Niet verder investeren in remux als default.

Volgende logische focus:

- non-remux decoder/locality-optimalisatie
- minder dure decode/read/grab-herhalingen in de bestaande flow
- frame-export en capture-lokaliteit verder aanscherpen

Met andere woorden:

- optimaliseren binnen het huidige directe videopad
- niet meer via tijdelijke partities

## Huidige Default

De runtime-defaults zijn nu bewust gezet op:

- `pass1_scan_workers = 2`
- `score_analysis_workers = 4`

in [app_config.json](/C:/Ai/MarioKart8_lanplay_video_processor/config/app_config.json).

## Relevante Rapporten

Details van deze ronde staan ook in de gegenereerde benchmarkrapporten:

- `.codex_tmp/remux_matrix_full/report.md`
- `.codex_tmp/remux_matrix_full/summary.csv`
- `.codex_tmp/remux_matrix_full/raw_runs.csv`
- `.codex_tmp/remux_matrix_followup/report.md`
- `.codex_tmp/remux_matrix_followup/summary.csv`
- `.codex_tmp/remux_matrix_followup/raw_runs.csv`
- `.codex_tmp/remux_matrix_confirm_2video/report.md`
- `.codex_tmp/remux_matrix_confirm_2video/summary.csv`
- `.codex_tmp/remux_matrix_confirm_2video/raw_runs.csv`
- `.codex_tmp/remux_matrix_confirm_3video/report.md`
- `.codex_tmp/remux_matrix_confirm_3video/summary.csv`
- `.codex_tmp/remux_matrix_confirm_3video/raw_runs.csv`
- `.codex_tmp/remux_matrix_confirm_3video_ab/report.md`
- `.codex_tmp/remux_matrix_confirm_3video_ab/summary.csv`
- `.codex_tmp/remux_matrix_confirm_3video_ab/raw_runs.csv`

## Gebruik Van Deze Notitie

Gebruik deze notitie als beslislog:

- als iemand later opnieuw remux als default wil proberen, eerst hier lezen
- als iemand opnieuw naar `8/10/12` workers wil grijpen, eerst hier lezen
- als iemand een nieuwe decode/backendlijn wil onderzoeken, hier checken welke oude paden al zijn afgewezen

Alleen bij nieuwe omstandigheden opnieuw openen:

- nieuwe OpenCV/FFmpeg backendwijziging
- nieuwe exportstrategie
- ander soort inputvideo
- andere hardwaredoelstelling
- fundamentele wijziging in de TotalScore pipeline
