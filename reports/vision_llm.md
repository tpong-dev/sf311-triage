# Frontier vision LLM baseline (Claude)

Part of question 1 in the README: route a 311 request from the citizen's photo, comparing zero-shot CLIP, a small fine-tuned model, and a frontier vision LLM on accuracy **and** cost per 1,000 requests. This report covers the vision LLM arm.

**Status:** the harness is built and unit-tested offline. **No paid API calls have been made.** The results table at the end stays TODO until someone decides to spend the money.

Code: `src/sf311/vision_llm.py` · Tests: `tests/test_vision_llm.py` · Cached predictions: `data/llm_preds/{model}.parquet`

## Method

- **Task:** single-label classification into the 12 classes in `sf311.dataset.LABELS` plus `Other` (13 in total), using only the photo. Location, time and request text are not used, so the comparison with the image-only models is fair.
- **Data:** `data/images/manifest.parquet` rows with `split == "test"` and `ok == True`. The test split is time-based (requests from 2026-09-01 onward), the same split every other model uses.
- **Sample:** `--n` photos drawn as a stratified sample of the test split. Allocation is proportional to label frequency, with at least 5 photos per class when that many exist so that rare-class recall is not computed from one or two photos. The fixed seed is 42. Because of the rare-class floor, the sample is slightly less skewed than the real traffic. Accuracy on the sample is therefore an estimate, and macro-F1 and per-class recall are the headline numbers.
- **Models:**
  - `claude-sonnet-5-5` is the default.
  - `claude-haiku-5-5` is the cheap option.
  - `claude-opus-5-5` is priced in the code but not part of the plan.
- **One request per photo**, sent through the Messages API:
  - **System prompt:** describes what each label looks like and how to settle the confusable pairs. It carries `cache_control: ephemeral`. The first request runs on its own to write the cache, and the remaining requests then read it at 0.1x the input price.
  - **Image:** read from `data/images/{split}/{id}.jpg` and sent as base64. Any photo larger than `--max-side` px (1024 by default) is downscaled in memory first. Image bytes are never written anywhere new.
  - **Structured output:** `output_config.format` with a JSON schema whose `label` is an enum of the 13 class names, plus `confidence` (0 to 1) and a one-sentence `reason`. Forced `tool_choice` is not used because Sonnet 5.5 rejects it with a 400; the schema guarantees parseable JSON anyway.
  - **Effort:** `output_config.effort = "low"` by default. Classification rarely benefits from long thinking, and thinking tokens are billed as output.
  - **Refusal fallback:** on Sonnet and Opus the request sends `fallbacks: "default"` with the beta header `server-side-fallback-2026-07-01`. If a safety classifier declines a photo, the API re-runs it on a fallback model instead of returning nothing. `--no-fallback` turns this off. Haiku 5.5 has no server-side fallback. A refusal that survives is recorded as `Other` with confidence 0 and `error = "refusal"`.
- **Reliability:**
  - The SDK retries 429, 5xx and connection errors (`--sdk-retries`, default 4).
  - The harness adds `--retries` (default 2) attempts with backoff on top, and also retries when the JSON is malformed or empty.
  - An asyncio semaphore limits requests in flight (`--concurrency`, default 8).
  - Photos that still fail are logged by request id and are not cached, so a rerun tries them again.
- **Result cache:**
  - Every prediction is appended to `data/llm_preds/{model}.parquet`, flushed every 25 results and on exit.
  - Rows are keyed by `(service_request_id, prompt_hash)`. `prompt_hash` covers the system prompt, the user text and the schema, so editing the prompt automatically invalidates old predictions, while reruns and bigger `--n` values pay only for photos not yet classified.
  - Stored columns: ids and labels, the prediction, the confidence, the model's reason, the token usage fields, the model that actually served the request, and the cost. Image bytes and URLs are never stored.
- **Metrics:**
  - Accuracy.
  - Macro-F1 over the classes present in the sample.
  - Per-class recall, precision, F1 and support. These are implemented directly in the module, so scikit-learn is not needed.
- **Cost:** computed per request from `usage` using the per-model rates below. Each request's cost is input tokens × input rate + cache-write tokens × write rate + cache-read tokens × read rate + output tokens × output rate. The script reports the total and the cost per 1,000 images.

### Pricing used

These are Anthropic first-party API rates, taken from the model and pricing table cached in Claude Code's `claude-api` skill on **2026-10-06** (canonical page: https://platform.claude.com/docs/en/about-claude/pricing). Re-check before budgeting.

| Model | Input $/MTok | Output $/MTok | 5-min cache write | Cache read |
|---|---:|---:|---:|---:|
| claude-sonnet-5-5 | 2.00 | 10.00 | 2.50 (1.25x) | 0.20 |
| claude-haiku-5-5 | 0.10 | 0.50 | 0.125 (1.25x) | 0.01 (0.1x) |

Haiku 5.5 rates apply to prompts of 100K tokens or fewer, which is always true here. If the server-side fallback serves a request, that request is priced using the serving model's rates.

## Prompt

The system prompt below is printed in full by `--dry-run` (prompt hash `8737691632ae`):

```
You route San Francisco 311 service requests. A resident submitted the attached photo with a request;
your job is to choose the single service category the city would file it under, using the photo only.

Categories (use the exact name):
- Street and Sidewalk Cleaning: loose garbage or debris on a street or sidewalk that needs pickup
  (litter, bags, dumped furniture, mattresses, appliances, food waste, glass, needles, waste)...
- Parking Enforcement: a vehicle is the subject (on sidewalk, blocking driveway, double-parked,
  bike lane, red zone, oversized, apparently abandoned)...
- General Request: catch-all addressed to a department; most often a person needing outreach without
  a tent setup, other public-works or transit issues; choose only when no specific class fits...
- Graffiti Public: graffiti on public property (poles, signal boxes, mailboxes, shelters, public cans...)
- Graffiti Private: graffiti on private property (walls, storefronts, shop gates, garage doors...)
- Blocked Street and Sidewalk: objects, not trash or parked cars, obstructing the way
  (construction materials, debris boxes, scaffolding, A-frames, tables and chairs)...
- Encampment: tents, tarps, makeshift shelters, bedding set up, carts with belongings...
- Sidewalk and Curb: damage to the pavement itself (raised or cracked sidewalk, curb ramp,
  vent cover, manhole cover off, public stairway)...
- Tree Maintenance: fallen, broken, leaning or overgrown street trees, roots, landscaping...
- RPD General: problem inside a city park, playground, field or rec facility...
- Litter Receptacle Maintenance: the trash can itself (overflowing, tipped, damaged, liner or lock,
  toters left out, cans at transit shelters)...
- Sewer: storm drain or catch basin, flooding, water leak, sewage back-up, sewer odor...
- Other: streetlights, traffic signals, signs, potholes, transit shelter damage, posters on poles,
  noise, and photos with no visible city issue.

Guidance for hard cases: main subject wins; trash on ground vs trash CAN; parked vehicle vs other
obstruction; graffiti public vs private by surface owner (default public); tent/shelter vs lone person;
sidewalk damage vs roadway pothole (Other); park setting -> RPD General; unclear -> Other, low confidence.

Return JSON only: label, confidence (0-1), reason (one short sentence; no faces, identities,
licence plates or addresses).
```

The listing above is abridged; the exact text is `SYSTEM_PROMPT` in the module. The label descriptions are based on the most common `service_subtype` values per label in the training months. For example, `Parking Enforcement` includes abandoned vehicles and parking on the sidewalk; `General Request` is mostly homelessness outreach, public works and MTA requests; and `Other` is mostly noise, streetlights, transit shelters, posters on poles and traffic signals.

## How to run

```bat
set PYTHONPATH=src

:: 1. free: print prompt + cost estimates for both models at n=500 and n=1300 (no API call, no key needed)
.venv\Scripts\python -m sf311.vision_llm --dry-run

:: 2. offline tests (mocked client)
.venv\Scripts\python -m unittest tests.test_vision_llm -v

:: 3. PAID: cheapest first, with a hard cap on the estimate
set ANTHROPIC_API_KEY=...
.venv\Scripts\python -m sf311.vision_llm --model claude-haiku-5-5 --n 500 --max-cost 1
.venv\Scripts\python -m sf311.vision_llm --model claude-sonnet-5-5 --n 500 --max-cost 5

:: 4. re-print metrics from cached predictions (no API call)
.venv\Scripts\python -m sf311.vision_llm --model claude-sonnet-5-5 --n 500 --report-only
```

Useful flags:

- `--concurrency 8`: the number of requests in flight.
- `--effort low|medium|high`.
- `--max-side 1024`: use 0 to send photos at their original size.
- `--no-fallback`: turns off the server-side refusal fallback.
- `--retries` and `--sdk-retries`: the harness's and the SDK's retry counts.

Before any paid call, the script prints the number of photos already cached and the estimated cost of the remaining ones. It aborts if that estimate exceeds `--max-cost`.

## Cost estimate (dry run, before any paid call)

The estimate assumes:

- **System prompt:** about 1,670 tokens per request, counting the prompt and the schema. The first request writes it to the cache and every later request reads it.
- **Output:** about 250 tokens per photo, covering the JSON plus low-effort thinking.
- **Images:** the photos downloaded so far are **256-px-long-edge thumbnails**, measured on the val photos already in the manifest. At that size an image costs only about 66 tokens. The full-size row shows what sending roughly 1024-px photos (about 1,050 tokens each) would cost instead.

| Model | Image size | n = 500 | n = 1,300 | $ / 1,000 images |
|---|---|---:|---:|---:|
| claude-sonnet-5-5 | 256-px thumbnails (current download) | $1.50 | $3.90 | $3.00 |
| claude-sonnet-5-5 | ~1024 px | $2.49 | $6.46 | $4.97 |
| claude-haiku-5-5 | 256-px thumbnails (current download) | $0.08 | $0.20 | $0.15 |
| claude-haiku-5-5 | ~1024 px | $0.12 | $0.32 | $0.25 |

Caveats:

- **Output tokens dominate Sonnet's cost at thumbnail size,** at about $2.50 per 1,000 images. If low-effort thinking runs longer than assumed, at 500 output tokens per photo, Sonnet's cost rises to about $5.50 per 1,000; Haiku's stays under $0.30.
- **Token counts are heuristic.** The script estimates them from characters and from w×h/750, without calling `count_tokens`. The real figures come from `usage` after a run.
- **Thumbnail resolution may cap LLM accuracy,** because small text, graffiti detail and pavement cracks are hard to see at 256 px. The CLIP and fine-tuned models see the same thumbnails, so the comparison stays fair, but this is worth noting as a limitation.

## Results

TODO: fill this in after a paid run, which needs the user's approval. Paste the output of `--report-only`.

| Model | n | Accuracy | Macro-F1 | Errors / refusals | Actual $ / 1,000 images | Mean tokens per image (in / cache-read / out) |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | TODO | TODO | TODO | TODO | TODO | TODO |
| claude-sonnet-5-5 | TODO | TODO | TODO | TODO | TODO | TODO |

Per-class recall: TODO. Use the table that `--report-only` prints.

| Label | Support | Haiku recall | Sonnet recall |
|---|---:|---:|---:|
| Street and Sidewalk Cleaning | | | |
| Parking Enforcement | | | |
| General Request | | | |
| Graffiti Public | | | |
| Blocked Street and Sidewalk | | | |
| Encampment | | | |
| Sidewalk and Curb | | | |
| Tree Maintenance | | | |
| RPD General | | | |
| Graffiti Private | | | |
| Litter Receptacle Maintenance | | | |
| Sewer | | | |
| Other | | | |

## Limitations and privacy

- **Photo only.** The city's label also reflects the reporter's text and the agency's routing rules (for example, `General Request` is defined by department, not by what the photo shows), so some classes have a low ceiling for any image model.
- **Data leaves the machine.** Each photo is sent to the Anthropic API for classification. Photos can show people and licence plates, so the prompt tells the model not to describe faces or transcribe plates or addresses. Its `reason` text is stored locally only. This report contains no photos, URLs or per-request text.
- **Pricing date.** The rates above are dated 2026-10-06. The rates in the code (`PRICING` in the module) must be updated if Anthropic changes them.
