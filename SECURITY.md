# Security Policy

## Supported versions

Cortica is pre-alpha. Only the latest release on PyPI receives fixes.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| < 0.1 | ❌ |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub's
[security advisory form](https://github.com/samnemati/cortica/security/advisories/new),
which notifies the maintainer without disclosing the report publicly.

Expect an acknowledgement within a week. If a fix is needed, the advisory will be
published alongside the release that carries it, crediting you unless you ask
otherwise.

## Scope

Cortica is a desktop research tool: it reads neuroimaging recordings, runs a local
analysis pipeline, and writes reports. It has no server component, no network
listener, and no authentication. The things worth reporting are therefore mostly
about handling untrusted input:

- Code execution or path traversal triggered by opening a crafted recording,
  pipeline YAML, montage file, or behavioral table.
- HTML or script injection into an exported report (report values are escaped —
  a bypass is a valid report).
- Unsafe deserialization in pipeline loading.

Out of scope: vulnerabilities in dependencies — report those upstream (MNE-Python,
PySide6, NumPy, …), though do tell us if Cortica needs a version bump to pick up
a fix.

## A note on data

Cortica runs entirely locally and does not transmit your recordings anywhere.
Exported HTML reports **embed your data** as figures and values, so treat a report
like the recording it came from before sharing it.

Cortica is for research use only. It is not a medical device and must not be used
for clinical diagnosis or treatment.
