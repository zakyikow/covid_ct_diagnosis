# Step 4 · The demo page

> **Live:** [zakyikow.github.io/covid_ct_diagnosis](https://zakyikow.github.io/covid_ct_diagnosis/)<br>
> A one-page site that tells the project's story and lets anyone watch the honest model diagnose 6 patients.

## What happens

- **Pick a patient** (drag, click, tap or keyboard), watch the model read the scan, then see its diagnosis, its
  5 most suspicious slices and where it looked.
- **Each patient is judged by the fold model that never saw it** in training.
- **Nothing runs in the browser:** `bash scripts/run.sh step4` computes the results in advance and saves them in
  `public/data/`. The page reads every number from there; none is typed by hand.

## The 6 patients

| Patient | True diagnosis | The model says | Why this one |
|---|---|---|---|
| P002 | COVID-19 | COVID-19, 94% | typical case, lesions confirmed by radiologists |
| P114 | COVID-19 | COVID-19, 94% | severe, both lungs |
| cap007 | Pneumonia | Pneumonia, 96% | dense patch, upper right lung |
| cap040 | Pneumonia | Pneumonia, 97% | dense patch, lower right lung |
| normal056 | Healthy | Healthy, 95% | no slice scores high enough for a heatmap |
| P021 | COVID-19 | **Pneumonia, 78%** | the "hard case": the most common mistake |

The list is in `config.yaml`: change it and run step 4 again to update the page.

## Run

Needs Node.js 22 or newer. From this folder:

```bash
npm install      # once
npm run dev      # the page on a local address, reloaded on every change
npm run build    # the finished page, in dist/
npm run preview  # the finished page, as it will be published
```

At every push to `main`, GitHub builds the page and publishes it on GitHub Pages.

## Inside

| | |
|---|---|
| **Stack** | React, TypeScript, Vite, Tailwind CSS, Motion (drag and drop, animation), NumberFlow (counting numbers), Geist fonts |
| **Data** | `public/data/`: per patient, every slice's scores, the diagnosis, small frames for the animation, the top 5 slices and their heatmaps |
| **Checks** | the export stops if the model doesn't reproduce the saved results; Lighthouse 100 in all four categories on desktop; no accessibility issue found by axe |
| **Code** | `src/components/demo/` the demo, `src/components/` the other sections, `src/lib/` data loading, labels, scrolling |
