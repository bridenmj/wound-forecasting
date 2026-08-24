# Private notebook manifest

The complete paper-era notebooks are archived outside Git in Google Drive.
They contain exploratory cells, output history, machine-specific paths, and
training/evaluation development code. This repository exposes the cleaned
implementation instead.

Local reference aliases and SHA-256 identities:

| Experiment | Local alias | SHA-256 |
|---|---|---|
| LLaMA-Adapter | `llama_adapter_final.ipynb` | `c327c574b02061e4a074f35928ced3e595c9492ac5bd33a6df744cd7185873d4` |
| DyneODE | `dyneode_variable_context_final.ipynb` | `f0b5316835c1748fdd923e39430baf9e9ab0a81d6f0cafebf83a6b2c8ba82cd8` |
| River | `river_final.ipynb` | `3026f564953de99e58320ee54959f972339c6f88556b04c47adc7a10a40f9f60` |

These hashes allow a private notebook to be matched to the exact artifact used
during repository extraction without publishing the notebook itself.
