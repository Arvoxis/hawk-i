# Multi-Query YOLO-World

## Why multi-query?

YOLO-World uses CLIP text embeddings to match visual regions against class labels.
CLIP was trained on natural image-text pairs scraped from the web, not on
construction or civil-engineering inspection literature.  Domain-specific labels
like "Efflorescence" or "Exposed_reinforcement" produce weak embeddings because
CLIP has rarely seen those words paired with relevant images.

Replacing each technical label with 5 plain-language descriptions that CLIP
understands well — "white powder on wall", "salt deposit on surface" — gives the
model a much better chance of matching the correct visual region.

## How it works

```
QUERY_MAP = {
    "Crack": [
        "thin line in concrete",   ← sub-queries
        "hairline crack",
        "fracture in wall",
        ...
    ],
    ...
}
```

1. All sub-queries across all classes are flattened into one list and passed to
   `model.set_classes()` **once at init**.  YOLO-World's CLIP text encoder
   processes all of them in a single forward pass.

2. Each `predict()` call runs **one** vision forward pass regardless of how many
   classes or sub-queries are defined.

3. The detected sub-query label is reverse-looked-up to its canonical class name.

4. Per-class NMS merges overlapping boxes that were fired by different sub-queries
   of the same class.  Boxes from different classes are never merged.

## Adding a new defect class

Edit `QUERY_MAP` in `multi_query_yoloworld.py`:

```python
QUERY_MAP["Delamination"] = [
    "concrete layer separating",
    "surface sheet peeling from slab",
    "hollow sound area on wall",
    "debonded layer in concrete",
    "bubbling paint on cement",
]
```

Guidelines for writing sub-queries:
- Describe what the defect *looks like*, not what it *is* ("brown stain on concrete",
  not "ferrous oxide deposit").
- Use phrases of 3–7 words — CLIP performs best in this range.
- Cover different visual presentations of the same defect (colour, shape, context).
- Avoid jargon that CLIP is unlikely to have seen in image captions.

## YOLO-World batching limitations

| Limit | Detail |
|-------|--------|
| ~80 classes max | CLIP's context window can overflow silently above ~80 text tokens total. The default QUERY_MAP uses 30 sub-queries, well within this limit. Monitor confidence scores if you add many new classes. |
| No sequential fallback needed | `set_classes()` accepts an arbitrary list; all strings are encoded in one CLIP forward pass. There is no need to loop over sub-queries individually. |
| Jetson note | Do **not** pass `torch.compile` or `half()` to the model on Jetson — the default FP32 path is stable across JetPack versions. |

## Running the benchmark

```bash
python multi_query_yoloworld.py \
    --benchmark \
    --model yolov8s-worldv2.pt \
    --frames data/test_frames \
    --conf 0.25 \
    --iou 0.4
```

## Running the tests

```bash
python -m pytest tests/test_multi_query.py -v
```

No GPU, no model file, and no `ultralytics` installation required to run the tests.
