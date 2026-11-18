# Production semantic mapping

- Use `python run.py build` for native HMSG, ConceptGraphs and VLMaps. Keep the original serial stage order and native file contracts.
- Keep vendored sources and local patches traceable. Record changes to model identity, calibration, sampling, detector settings and association explicitly.
- Use a new run directory for each recording. Preserve stage failures and keep class-locked copies outside native outputs.
- Keep model weights, downloaded caches, virtual environments, recordings and run results outside source control. Use synthetic assets for CPU tests.
- Report input preflight, clean setup and real GPU mapping separately. No robot motion, service activation or editing deployment is part of this project.
