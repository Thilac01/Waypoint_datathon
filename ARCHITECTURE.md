# Model architecture and proposed deployment

```mermaid
flowchart TD
  A["Organizer CSVs"] --> B["Schema and join checks"]
  B --> C["Task1 labels and planned features"]
  B --> D["Order-date demand panel"]
  B --> E["Scenario and fleet rules"]
  C --> F["Chronological model comparison"]
  D --> G["Ten-week forecast backtests"]
  E --> H["CP-SAT allocation"]
  F --> I["Saved prediction ensembles"]
  G --> I
  I --> J["Template-aligned prediction CSVs"]
  H --> K["Allocation CSV and policy"]
  J --> L["Validation and audit reports"]
  K --> L
```

Training is a local batch job. Task1 and Task2A final models are serialized with their fitted preprocessing and ensemble weights. Allocation is a separate deterministic constraint model with time-limited optimization, explicit status and bounds.

Proposed deployment: run the saved Task1 models after the order cutoff once planned routes and current road inputs are available. Run demand forecasts as a weekly batch using only observations available at that date. Send predictions to a dispatcher application, retain input/model versions, and monitor service error, lateness calibration and demand error by brand/depot. Use observed new outcomes only in a later retraining batch. This proposal is documented; no HTTP service or live logistics integration is claimed in this deliverable.
