# Introduction

## Background
Underwater imaging has a fundamentally different visual physics profile compared to terrestrial imaging. Water rapidly absorbs and scatters light, especially in red wavelengths, causing:
- dominant blue/green color casts
- low local contrast
- blurred edge boundaries
- reduced object separability for detectors

Traditional image pipelines are not sufficient for these conditions, especially when the goal is reliable AI inference and not only visual beautification.

## Motivation
The motivation behind JalDrishti is practical and research-driven:
- make underwater scenes interpretable for operators and reviewers
- improve machine detection quality by improving image quality first
- demonstrate a complete, production-style AI workflow in an academic project
- provide a visually impressive and technically sound system for evaluations/demos

## Existing Systems
Most existing systems fall into one of these categories:
1. **Enhancement-only systems** that improve visuals but do not provide semantic understanding.
2. **Detection-only systems** that run directly on degraded underwater inputs, causing unstable outputs.
3. **Research prototypes** that are difficult to deploy for non-technical users.

## Limitations of Existing Systems
- Poor robustness across scene types (reef, open water, diver shots, fish schools)
- Frequent color artifacts when enhancement is too aggressive
- High false-positive rate in noisy marine backgrounds
- Lack of integrated UI for explainable outputs
- Weak reproducibility for first-time setup on new machines

## Proposed Solution
JalDrishti addresses these limitations with an integrated, layered architecture:
- **Hybrid enhancement engine** (deep model + classical fallback + tone balancing)
- **Marine + diver detection pipeline** with post-processing guards
- **Scene-aware label refinement** to reduce common underwater misclassifications
- **Depth and environmental analysis modules** for richer output context
- **Web-based interface** for upload, analysis, and visual explanation

The key philosophy is not to rely on a single model blindly, but to combine learned and heuristic intelligence for stable performance in unpredictable underwater scenes.
