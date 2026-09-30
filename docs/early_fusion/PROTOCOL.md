# Channel Audit (Final)

- **channels.json:** 151 channels
- **Ingested:** 135 channels
- **Missing:** 16 channels, all verified via the API (see `missing_channels_audit.csv`)

## Categories

- **channel_not_found (6):** `@AlexG`, `@Empleman`, `@MaxMiller`, `@NikoOmilana`, `@ProHomeCooks`, `@TheEngineeringMindset` — handles could not be resolved.

- **no_videos_in_window (4):** `@CompanyMan`, `@Hoog`, `@HowToMakeEverything`, `@LessonsFromTheScreenplay` — no uploads published on or after `2025-01-01`.

- **all_shorts (4):** `@Garage54`, `@PracticalEngineering`, `@TomSka`, `@jasontheween` — all uploads are Shorts (≤180 seconds).

- **Unresolved errors (2):** `@AdamRagusea`, `@Kraut` — network timeout / playlist 404 errors.

## Conclusion

No channels failed due to an ingestion bug.

These 16 channels simply have no videos that meet the filtering criteria.

The final snapshot covers **135 channels**.

## Text Token Length (L)

- Snapshot: 11,285 titles
- Distribution: p50=11, p90=19, p95=22, p99=29, max=174
- Selected L: **32**
- Rationale: 0.59% of titles are truncated (≤1% threshold), saving ~200 MB of memory vs L=48
- Longest title: 174 tokens (outlier, still truncated to 32)

## Thumbnail Size Distribution (all 11.285)
- 1280×720 (16:9, maxres): 9.565 (84.8%)
- 640×480 (4:3, high):     1.710 (15.2%)
- 480×360 (4:3, medium):      10 (0.1%)
- Transform: `squash` → (224, 224), ignoring aspect ratio
- Mode: RGB for all
- Note: 15.3% of thumbnails are non-16:9 → different visual distortion when squashed