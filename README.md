# DMAP

## Overview

![workflow](./workflow.png)

We propose DMAP, a novel multi-modal framework for document question answering. We designed a Structured-Semantic Understanding Agent to construct DMAP by organizing textual content together with figures, tables, charts, etc into a human-aligned hierarchical schema that captures both semantic and layout dependencies. Building upon this representation, a Reflective Reasoning Agent performs structure-aware and evidence-driven reasoning, dynamically assessing the sufficiency of retrieved context and iteratively refining answers through targeted interactions with DMAP. Extensive experiments on MMDocQA benchmarks demonstrate that DMAP yields document-specific structural representations aligned with human interpretive patterns, substantially enhancing retrieval precision, reasoning consistency, and multimodal comprehension over conventional RAG-based approaches.

## Requirements

1. Clone this repository and navigate to DMAP folder

   ```bash
   git clone https://github.com/Forlorin/DMAP.git
   cd DMAP
   ```

2. Install package with conda.

   ```bash
   conda create -n dmap python=3.12
   conda activate dmap
   bash install.sh
   ```

3. Download the dataset and place it in the data directory. The Dataset we use is same to [MDocAgent](https://github.com/aiming-lab/MDocAgent).

4. Download & install [pdffigure2](https://github.com/allenai/pdffigures2), put it in "./mydatasets/pdffigure2/" and configure it well. Make sure that the "run_pdffigure2.sh" is in the same folder with the "src" folder of pdffigure2.

## Index
