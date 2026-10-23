# Retired legacy environments

Retired on 2026-10-23. This 2018 coursework is preserved as a historical learning record, not a supported application or environment template.

The original notebooks, data, Python requirements and Conda manifests remain unchanged. Git history preserves their provenance. The security scan identified 22 affected package/version groups in the old environments; retirement does not patch those packages or make them safe.

Do not install the original environment, start its notebook server, deploy it, or reuse its requirements in another project. Existing setup instructions are historical documentation. No active deployment was found for this repository; no infrastructure is started by this change.

To revisit an exercise, create a separate modernization project with a supported Python runtime, a newly resolved and scanned dependency lock, and tests for that exercise. TensorFlow/Keras API migrations and model compatibility must be validated before execution. Do not load untrusted historical model or serialized data files.

Repository archival is a separate release action after this retirement notice is reviewed and merged. Archival preserves the files; it does not resolve the historical advisories. Security reporting must retain the retirement record rather than describe the old dependencies as patched.
