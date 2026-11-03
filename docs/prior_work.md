# Prior work and benchmarks

Compiled 2026-11-04. Status key: **V** = I opened the page/PDF and read the content cited; **P** = partially verified (abstract/README only, numbers not seen); **U** = seen only in a search snippet, not opened (not relied on).

Headline: I found no public SF 311 resolution-time model with a credible held-out metric, and no public benchmark of CLIP or a vision LLM on 311-style photos. The comparison this project plans (zero-shot CLIP vs fine-tuned small model vs frontier VLM, with cost) appears to be unoccupied.

## Summary table

| # | Source | Area | Predicted | Data | Method | Reported metric | Status |
|---|---|---|---|---|---|---|---|
| 1 | kimberlytanyh/Predicting_311_Call_ResolutionTime (GitHub) | (a) SF resolution time | Resolution time (regression) | SF 311 cases, 2020 only, joined to businesses/facilities/socio-economic data | Linear reg, RF, XGBoost | Best (XGBoost) R2 about 0.26; RMSE/MAE values not shown | V (README) |
| 2 | pjandir/CapstoneProject2 | (a) SF volume | Weekly request counts per type per census tract | SF 311 + Census | SARIMA | MAE (values not shown) | V (README) |
| 3 | sfbrigade/data-science-wg SF_311 exploratory README | (a) SF descriptive | none (EDA) | SF 311 full history | Medians by type/neighborhood, KL divergence | none | V (commit) |
| 4 | Yi 2023 USF dissertation | (b) SF 311 photos | Object detection: garbage, cardboard, cans, overflowing cans; drug-related objects | SF 311 citizen photos (size not seen) | Faster R-CNN, RetinaNet; Faster R-CNN, EfficientDet, YOLOv5 | RetinaNet mAP 0.87 (garbage); YOLOv5 mAP 0.64 (drug objects) | P (abstract page) |
| 5 | Kumar et al. 2019, arXiv 1901.10124 | (b) civic photos (FixMyStreet) | 10-class issue category | 80,640 train / 4,992 test images from civic complaint portals | VGG-16 (COCO-pretrained) baseline; scene-graph proposal | Accuracy 47.13%, F1 38.76; best classes 86.5/83.6/75.2%, 4 of 10 classes below 17% | V (PDF text) |
| 6 | Vrabie 2025, arXiv 2504.08972 | (b) civic photos (Romanian city) | 3 classes: infrastructure damage, waste, illegal parking/misc | 5,712 images, 2022 | Fine-tuned ResNet (Keras) | Accuracy 92.8%, P 0.93, R 0.91 (single 80/20 split, no test set) | V (PDF text) |
| 7 | MuST2-Learn, arXiv 2508.16503 | (a) 311 service time | Service time for new requests | Chattanooga 311, about 170k requests with completion time (second dataset unnamed) | Spatio-temporal-type learning | MAE reduced at least 32.5% vs unnamed baselines; absolute MAE not seen | P (abstract + PDF grep) |
| 8 | jarrarx/nyc311-72h-shadow (Hugging Face) | (a) NYC late closure | Not closed within 72h | NYC 311, train 2025, test 64,751 Mar-Apr 2026 | HistGradientBoosting; features complaint_type, borough, month/hour/weekday | Acc 0.845, late P 0.672, R 0.871, F1 0.759; no AUC | V |
| 9 | vbopardi/311ETCModel | (a) LA completion time | Estimated time of completion | LA 311 | not described | none | V (README, thin) |
| 10 | Sriphon et al. 2023, Traffy Fondue + CLIP Prefix Caption | (b)/(c) Bangkok | Image captions, then embedding clustering | Traffy Fondue images (size not seen) | CLIP prefix + GPT-2 | BLEU 0.93%, ROUGE-1 16.39% (authors: not good enough); embeddings suggested instead | V (abstract in PDF) |
| 11 | Zois et al., ACTION, SeeClickFix | (b) SeeClickFix | Category (Code Violations / Signs / Parking), text-feature-driven | SeeClickFix reports | Sequential hypothesis testing vs Bayes/SVM | Numbers garbled in extraction | P |
| 12 | Urban monitoring VLM review, arXiv 2510.12400 | (c) review | none | 32 studies | PRISMA review | Cites one few-shot Faster R-CNN on Jakarta JAKI reports: 2,402 images, 42.5% AP at 30-shot, 69.8% AP novel classes | V (PDF text, one table row) |
| 13 | Programmer-RD-AI/road-issues-detection-dataset (HF) | (c) data | 7 classes: pothole, damaged road, broken sign, illegal parking, litter, vandalism, mixed | 9,660 images, CC0, compiled from 8 mostly Roboflow sources | n/a (dataset) | n/a | V |
| 14 | Roboflow forum thread: zero-shot CLIP, 0.0 accuracy | (c) CLIP | 4 classes: encampment, garbage, graffiti, mural | about 150 own photos | Roboflow zero-shot CLIP notebook | Accuracy 0.0, unresolved | V (anecdotal, likely a pipeline bug) |

## Notes by area

### (a) SF 311 and 311 time/category prediction

- **Dataset itself (V, via data.sf.gov API):** vw6y-z8j6 is "311 Cases", PDDL license, cases since 2008-07-01, updated nightly. Columns include `requested_datetime`, `closed_date`, `status_description`, `agency_responsible`, `service_name`, `service_subtype`, `service_details`, `supervisor_district`, `analysis_neighborhood`, `lat`, `long`, `source`, `media_url`. A SODA count query returned about 9.04M rows on 2026-11-04. `media_url` gives the citizen photo link, which is what makes the image task possible.
- **#1 weaknesses:** one year (2020, a pandemic year with atypical operations); R2 0.26 is a weak bar; no per-service-type breakdown; features not listed; author says not production-grade; tree-model files missing from the repo; split method (random vs time-based) not stated.
- **#2:** forecasts volume, not resolution; MAE values not visible in the README.
- **#3:** useful only as descriptive context: median resolution times vary a lot by request type and neighborhood, so type is likely a dominant feature.
- **#7 (Chattanooga):** closest methodological peer for "predict service time for a new request". Heterogeneous request types, spatio-temporal demand features, text description. Baselines and absolute MAE not seen, and it is a different city.
- **#8 (NYC late-closure):** the best template for the binary "late" framing: thresholded label, time-based split, decision threshold 0.20. Weaknesses stated by the author: features limited to type, borough, time; fails completely on HEAT/HOT WATER (0 of 552 late cases found); test period burned; no AUC/PR-AUC. It is a classroom/shadow model, not a research benchmark.
- **Unopened leads (U):** Alteryx community post on NYC 24h closure with XGBoost; Boston 311 NEU course proposal (course.ccs.neu.edu/ds2500sp23/proposals/27.html); IIETA LightGBM incident-resolution paper (IT service management, not 311). I did not open these and did not rely on them.
- **Kaggle:** I found no Kaggle notebook on SF 311 closure time. My search tooling could not query Kaggle directly, so treat "none exists" as unconfirmed.

### (b) 311 / civic photo classification

- **#5 is the cleanest multi-class benchmark:** a 10-class FixMyStreet-category VGG-16 gets 47.13% accuracy, with four classes under 17%. It shows noisy citizen labels cap plain-classifier accuracy. Old backbone (VGG-16, 2019), so a modern fine-tuned small model should beat it, but the dataset is not clearly public (they release derived datasets; I did not verify access).
- **#6 is a high-looking number with weak support:** 92.8% on 3 coarse classes, single split, no independent test set, and a catch-all "illegal parking / miscellaneous" class. Do not treat it as a bar for a finer-grained task.
- **#4 is the only SF 311 photo work found, and it is detection, not category/agency prediction.** It confirms SF 311 photos are usable and that "Street and Sidewalk Cleaning" dominates (36% of requests per the abstract). Full text not read, so dataset size and splits are unknown.
- **Not verified:** an EUDL/Springer chapter (doi 10.1007/978-3-030-76063-2_35) surfaced in search as an SF 311 dataset of about 2,500 annotated images; EUDL and Springer blocked my fetch (403 / login redirect), so I cannot confirm authors, title, or numbers. Marked U.
- **Traffy Fondue (#10):** the only Traffy Fondue paper I opened is about captioning, with very poor text-generation scores. A GitHub README claiming a 10-class Traffy classifier appeared in search but I did not open it (U). No reported Traffy classification accuracy found.
- **SeeClickFix (#11):** text-feature work; the photo is only a data field. No SeeClickFix image classifier found.
- **FixMyStreet:** only via #5.

### (c) CLIP / vision-LLM on urban-issue photos

- I found **no paper reporting zero-shot CLIP or a GPT/Gemini/Claude-class VLM on litter/graffiti/pothole/encampment 311 photos.** The review (#12) covers zero-shot VLMs for urban monitoring in general; the abstract and the text I grepped show no 311 citizen-report VLM benchmark.
- StreetSurfaceVis (arXiv 2407.21454, road surface quality, GPT-4o used for label prediction): I opened the abstract page; it reports no GPT accuracy numbers, so it is not a benchmark. SAGAI (2504.16538), 2409.18286, 2409.02278 and 2408.12821 (GPT-4V/Gemini on street view / transport) appeared in search; I did not open them (U).
- Encampment is an obvious gap: no labeled dataset or classifier found. The only encampment mention is the Roboflow thread (#14).
- #13 is a ready CC0 set for litter/vandalism/pothole sanity checks, but it is Roboflow-sourced imagery, not citizen phone photos, so it will not reflect the 311 distribution.

## Gaps this project can fill

1. **Head-to-head on SF 311 photos with cost:** zero-shot CLIP vs fine-tuned small model vs frontier VLM, with accuracy, macro-F1, and $/1k images and latency. Nothing comparable found.
2. **Official labels at scale:** use `agency_responsible` and `service_name`/`service_subtype` as labels; prior photo work uses ad hoc classes. About 9M rows, `media_url` present on a subset.
3. **Label noise honesty:** citizens often pick the wrong category (see #5's weak classes). Report a human-audited clean subset alongside raw labels.
4. **Time-based split for resolution time:** #1 used a single year and #8 is the only time-split example. Use forward splits across years, report MAE (hours or log-hours) vs a per-type median baseline, plus AUC/PR-AUC and calibration for "late", broken out by `service_name`.
5. **Define "late" cleanly:** no SF source defines it. Options: exceeds the per-type 75th percentile, or a fixed SLA; document the choice.
6. **Does the photo help the time model?** Add image embeddings/predicted category to LightGBM features and measure lift over type+location+time alone. Not found in prior work.
7. **Encampment and other sensitive categories:** per-class error analysis and privacy discussion (photos contain people).

## Benchmarks to beat

- Resolution time (SF): **R2 about 0.26** (XGBoost, 2020 only, #1). Beat it on a forward time split across multiple years and also beat a per-type median baseline, which is the fairer bar.
- Late closure: **NYC F1 0.759 / P 0.672 / R 0.871 at 72h, acc 0.845** (#8) is a loose reference only, since city, label, and base rate differ. Report PR-AUC against the base rate.
- Photo category: **47.13% acc / 38.76 F1 on 10 classes** (#5, VGG-16) as a floor; 92.8% on 3 coarse classes (#6) is not comparable.
- Detection (if extended): garbage mAP 0.87, drug objects mAP 0.64 on SF 311 photos (#4).
- Frontier VLM and CLIP: no prior numbers; this project sets the baseline.

## Sources

1. https://github.com/kimberlytanyh/Predicting_311_Call_ResolutionTime
2. https://github.com/pjandir/CapstoneProject2
3. https://github.com/sfbrigade/data-science-wg/commit/535ad3c153a6cfd2fcf21f1eca967fb68996b299
4. https://digitalcommons.usf.edu/etd/10009
5. https://arxiv.org/abs/1901.10124 (PDF: https://arxiv.org/pdf/1901.10124)
6. https://arxiv.org/abs/2504.08972 (PDF: https://arxiv.org/pdf/2504.08972)
7. https://arxiv.org/abs/2508.16503
8. https://huggingface.co/jarrarx/nyc311-72h-shadow
9. https://github.com/vbopardi/311ETCModel
10. https://ph02.tci-thaijo.org/index.php/JSTNSRU/article/download/252002/169782/919746
11. https://par.nsf.gov//servlets/purl/10073980
12. https://arxiv.org/abs/2510.12400
13. https://huggingface.co/datasets/Programmer-RD-AI/road-issues-detection-dataset
14. https://discuss.roboflow.com/t/zero-shot-clip-example-works-on-default-ds-but-on-my-photos-accuracy-is-0-0/2123
15. https://data.sf.gov/api/views/vw6y-z8j6 (dataset metadata; FAQ at https://sfdigitalservices.gitbook.io/dataset-explainers/311-cases not read)
16. https://arxiv.org/abs/2407.21454 (opened; no usable GPT metrics)
17. Not opened (U): https://eudl.eu/pdf/10.1007/978-3-030-76063-2_35 (403), https://github.com/NonKhuna/NonKhuna/blob/main/README.md, arXiv 2504.16538 / 2409.18286 / 2409.02278 / 2408.12821
