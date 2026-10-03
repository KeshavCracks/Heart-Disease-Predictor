# Design system master: independent heart disease demo

## Source and identity

This interface adapts the user-provided `design.md` and the published Vercel Brand Guidelines stylesheet. The project is independently built and is not authored, endorsed, or operated by Vercel. Do not use Vercel wordmarks, logos, or wording that implies affiliation. Use the copied, byte-identical CSS foundation at `assets/vercel-brand.css` and its documented public classes and tokens; do not modify or recreate its token system.

## Reader and purpose

Visitors are learning how a classifier behaves on fictional values. They need to see what the output means, how the model was evaluated, and what the evidence cannot establish. The strongest safety boundary is that a score from historical labels is not a personal probability or a health assessment.

## Composition

- Open directly on the fictional-input tool and its default output, with the educational boundary in the first read.
- Continue through one scrollable reading path: example tool, grouped-CV candidate comparison, held-out test results, then source data and limitations.
- Preserve exact lookup in semantic tables. Do not add a chart when the comparison table answers the question more clearly.
- Avoid metric-card grids, decorative icons, banners, gradients, testimonials, visual risk bands, and duplicate summaries.
- Keep source, methods, and caveats available without making them louder than the tool.

## Foundation and page-owned styling

- Visual direction (2026-10 refresh): a self-contained **white-primary / blue-secondary** theme inspired by the UI/UX Pro Max skill and the React Bits aesthetic (soft aurora glow, faint dot-grid hero, gentle micro-interactions). Implemented with page-owned CSS only; no third-party JavaScript or component libraries.
- Use Inter for reading, headings, labels, inputs, and numeric results (tabular numerals for metrics); JetBrains Mono only for code identifiers such as `num` and `ca`.
- White is the main surface; blue (#2563eb / #1d4ed8) is the secondary accent for the brand mark, links, focus rings, the score, selected states, and section overlines. Do not use red/green risk colours; class meaning is carried by text.
- Keep the page white regardless of OS theme (`color-scheme: light`). Preserve visible keyboard focus, semantic landmarks, clear labels, reduced-motion support, and mobile reflow.
- Use native Streamlit controls and preserve the Streamlit framework; page fragments use the self-contained `hd-*` classes. The earlier Vercel Brand Guidelines stylesheet at `assets/vercel-brand.css` is retained for reference but is not loaded.

## Safety and evidence

- Use fictional values only. Do not persist submitted fields or prediction history.
- Describe results as source-data labels, not diagnoses, personal probabilities, or recommendations.
- State that the 0.50 threshold is a software default and the patient-level test split is not external validation.
- Keep the UCI source, cohort grouping, missingness, 11/13-field comparison, and no-clinical-use boundary easy to find.
