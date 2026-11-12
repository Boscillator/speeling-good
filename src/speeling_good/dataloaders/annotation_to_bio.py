"""Align spelling spans with token offsets for token classification."""

from collections.abc import Sequence

from speeling_good.preprocessing.types import SpellingAnnotation

LABEL2ID = {"O": 0, "B": 1, "I": 2}
ID2LABEL = {label_id: label for label, label_id in LABEL2ID.items()}


def annotation_to_bio(
    offset_mapping: Sequence[tuple[int, int]],
    annotations: Sequence[SpellingAnnotation],
    *,
    special_tokens_mask: Sequence[int] | None = None,
) -> list[int]:
    """Return one BIO label ID per token, ignoring special tokens with -100.

    Offsets must be half-open Unicode code-point spans into the original passage,
    as returned by a fast Hugging Face tokenizer with return_offsets_mapping=True.
    Tokenize the complete raw passage, rather than pre-split words. Supply one
    sequence of offsets at a time, before converting tokenizer outputs to tensors.
    Non-special token offsets are assumed to advance through the passage.

    Any positive overlap marks a token as erroneous. If a token touches multiple
    annotations, the earliest span wins (shortest first for equal starts). The
    first token assigned to each annotation receives B; later tokens receive I.
    Corrections are ignored, and neither input is modified.

    Zero-width offsets, including ordinary special/padding token offsets, receive
    -100. Supply special_tokens_mask to also ignore special tokens with nonempty
    offsets. Truncated annotations label only their visible tokens, starting at B.
    An empty annotation sequence represents a clean passage.

    Attach the result as the encoding's labels and remove offset_mapping and
    special_tokens_mask before passing it to a model. Configure the model with
    num_labels=3, id2label=ID2LABEL, and label2id=LABEL2ID.
    """
    if special_tokens_mask is not None and len(special_tokens_mask) != len(
        offset_mapping
    ):
        raise ValueError("special_tokens_mask must have one entry per token")

    spans = sorted((item.span_start, item.span_end) for item in annotations)
    for start, end in spans:
        if start < 0 or end <= start:
            raise ValueError(
                f"annotation span must satisfy 0 <= start < end: {(start, end)}"
            )

    labels: list[int] = []
    annotation_index = 0
    previous_annotation_index: int | None = None
    for token_index, (start, end) in enumerate(offset_mapping):
        if start == end or (
            special_tokens_mask is not None and special_tokens_mask[token_index]
        ):
            labels.append(-100)
            continue

        while annotation_index < len(spans) and spans[annotation_index][1] <= start:
            annotation_index += 1

        if annotation_index == len(spans) or spans[annotation_index][0] >= end:
            labels.append(LABEL2ID["O"])
            continue

        if annotation_index == previous_annotation_index:
            labels.append(LABEL2ID["I"])
        else:
            labels.append(LABEL2ID["B"])
        previous_annotation_index = annotation_index

    return labels
