# Contributing to Measured RAG

Thank you for considering a contribution. Measured RAG is maintained as a
focused portfolio reference implementation, so contributions should be
well-scoped, evidence-based, and straightforward to review.

## Before contributing

Read [Architecture](docs/architecture.md),
[Development](docs/development.md), and [Limitations](docs/limitations.md)
before changing the system. Report suspected vulnerabilities privately under
the [security policy](SECURITY.md), not through a public issue.

Discuss large behavioral, architectural, dependency, user-interface, or
benchmark changes before implementation. Small documentation corrections and
focused fixes may proceed directly as narrowly scoped pull requests. Submission
does not guarantee acceptance.

## Development workflow

1. Fork the repository and create a focused branch.
2. Follow the environment instructions in [Development](docs/development.md).
3. Make one coherent change without unrelated cleanup.
4. Add or update tests for changed behavior.
5. Update the canonical owning document when behavior or a public interface
   changes.
6. Run the relevant offline tests and formatting or structural checks.
7. Submit a focused pull request.

The application and evaluation dependencies intentionally use separate
environments. Preserve that isolation, and do not import `app.py` from headless
tests or benchmark modules. The canonical offline test commands and platform
notes are maintained in [Development](docs/development.md#run-offline-tests).

## Change expectations

Contributions should:

- avoid hidden network requests or model downloads;
- preserve the hosted privacy and telemetry boundaries documented in
  [Telemetry and diagnostics](docs/telemetry.md);
- keep credentials, sensitive data, and personal paths out of source, logs,
  fixtures, screenshots, and documentation;
- include tests for behavioral changes and explain user-visible or compatibility
  effects;
- avoid unrelated refactors and generated-file churn;
- preserve accessibility and the established user-interface system; and
- use non-sensitive fixtures and examples.

Keep code, documentation, dependency, deployment, and generated-artifact changes
independently reviewable where practical. Do not weaken validation, provenance,
coverage, resume, or defensive failure handling merely to simplify a change.

## Benchmark and data policy

The benchmark has stricter provenance and redistribution boundaries than normal
application development. Follow [Benchmarking](docs/benchmarking.md) and the
[published results interpretation](docs/benchmark-results.md).

- Do not commit downloaded research-paper PDFs or generated QASPER source
  datasets.
- Do not commit application logs, retrieved document text, vector stores,
  caches, credentials, or private artifacts.
- Do not manually modify canonical benchmark-run artifacts.
- Use a new run ID for a changed generation condition and a new judge ID for
  a changed judging condition. Record the relevant configuration, dataset, evaluator,
  and source provenance.
- Apply the redaction and attribution boundary demonstrated by the
  [curated publication bundle](benchmarking/publication/qasper-publication-v1-28p/README.md)
  to public benchmark artifacts.
- Retain the existing CC BY 4.0 attribution for QASPER questions and derivative
  material. Independently authored papers retain their own terms; an arXiv URL
  does not itself authorize redistribution.
- Support benchmark claims with denominators, coverage, configuration,
  limitations, and reviewable artifacts.
- Do not trigger provider-backed benchmark runs as routine contribution
  validation.

See [Third-party notices](THIRD_PARTY_NOTICES.md) for evaluator, dataset, model,
and redistribution details.

## Dependencies and licensing

Contributors retain copyright in their contributions. By submitting a contribution,
they agree that it may be distributed under the project’s [`AGPL-3.0-only`](LICENSE)
license and confirm that they have the right to submit it.

Third-party code, data, models, assets, and adapted material must have accurate
provenance and compatible license treatment. A dependency addition or update
must include a license review and changes to
[Third-party notices](THIRD_PARTY_NOTICES.md) when necessary. Do not add copied
code, paper text, icons, images, datasets, generated material, or model assets
with unclear rights.

AI-assisted contributions remain the contributor's responsibility. Review them
for correctness, security, provenance, and licensing before submission.

## Pull-request quality

A concise pull-request description should cover:

- the problem and rationale;
- the implementation;
- tests performed;
- documentation changes;
- security and privacy impact;
- dependency and license impact;
- benchmark impact; and
- screenshots for user-interface changes when useful.

Before submitting, confirm that:

- [ ] the change is focused and contains no secrets or private artifacts.
- [ ] relevant offline tests and structural checks pass.
- [ ] behavioral and user-visible changes are documented.
- [ ] dependency, license, security, privacy, and benchmark effects are stated.
- [ ] generated files and screenshots are intentional and safe to publish.

## Conduct

Communicate respectfully and constructively. Focus review discussions on the
technical change, its evidence, and its effect on users and maintainers.
