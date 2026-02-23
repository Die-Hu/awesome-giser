# Contributing to Awesome GISer

Welcome, and thank you for considering a contribution! Every addition — whether it is a single resource link or a full comparison table — helps the GIS community.

## How to Contribute

1. **Fork** this repository.
2. **Create a branch** for your changes:
   ```bash
   git checkout -b add/my-new-resource
   ```
3. **Make your edits** following the content standards below.
4. **Commit** with a clear message:
   ```bash
   git commit -m "Add <resource name> to <module>"
   ```
5. **Push** to your fork and open a **Pull Request** against `main`.

A maintainer will review your PR. Small fixes are usually merged within a few days; larger additions may go through a round of feedback.

---

## Content Standards

### Resource Entry Format

Every resource entry must include the following fields:

```markdown
- **[Resource Name](https://link.example.com)** — One-line description of what it does.
  - License / Cost: MIT / Free / $99 per year / etc.
  - Pros: Fast, open-source, active community.
  - Cons: Limited 3D support, steep learning curve.
```

### Comparison Table Format

When comparing multiple tools or datasets, use a Markdown table:

```markdown
| Tool | Type | Cost | Platform | Best For |
|------|------|------|----------|----------|
| QGIS | Desktop GIS | Free | Win/Mac/Linux | General-purpose analysis |
| ArcGIS Pro | Desktop GIS | $100+/yr | Windows | Enterprise workflows |
```

### SOTA vs Practical Labels

Where applicable, tag entries so readers can quickly gauge the trade-off:

- **SOTA** — State-of-the-art; best quality or performance, but may require expensive hardware, licenses, or expertise.
- **Practical** — Good-enough quality for most use cases, often free or low-cost.

### General Rules

- **All links must be verified.** Broken links will be rejected.
- **Code examples must be tested.** Include the runtime version (e.g., Python 3.11, Node 20) and any dependencies.
- **Keep descriptions concise.** One sentence is ideal; two sentences maximum.
- **Use English.** Bilingual annotations (English + Chinese) are welcome but the primary text must be in English.
- **No affiliate or referral links.**

---

## Suggesting Resources or Content

If you do not want to edit files directly, you can open an issue instead:

- [Suggest a Resource](.github/ISSUE_TEMPLATE/resource-suggestion.md) — Recommend a tool, dataset, or library.
- [Request Content](.github/ISSUE_TEMPLATE/content-request.md) — Ask for a new guide, comparison, or section.

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold a welcoming, inclusive, and harassment-free environment for everyone.

---

Thank you for helping make Awesome GISer better!
