import json

import torch

from wound_forecasting.llama_data import (
    WoundLlamaEvaluationDataset,
    WoundLlamaTrainingDataset,
    load_prompts,
    prepare_text_tokens,
)


def records():
    days = [f"Day_{index}" for index in range(8)]
    return [
        {
            "wound_id": "ID1323_Wound_I",
            "days": days,
            "tokens": {
                day: [torch.full((256,), day_index), torch.full((256,), day_index + 10)]
                for day_index, day in enumerate(days)
            },
            "images": {
                day: [torch.zeros(3, 8, 8), torch.ones(3, 8, 8)] for day in days
            },
        }
    ]


def prompts():
    return [
        {
            "context": [0, 1, 2, 3],
            "target": [4, 5, 6, 7],
            "text_tokens": [1, 2, 3],
        }
    ]


def test_prompt_loading_and_padding(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(prompts()[0]) + "\n", encoding="utf-8")
    assert load_prompts(path) == prompts()
    padded = prepare_text_tokens([1, 2], maximum_length=4, padding_token=99)
    assert padded.tolist() == [1, 2, 99, 99]


def test_training_dataset_masks_context_and_padding():
    dataset = WoundLlamaTrainingDataset(records(), prompts(), number_of_samples=1)
    example, labels, mask, text = dataset[0]
    assert example.shape == labels.shape == mask.shape == (2048,)
    assert (labels[:1024] == -100).all()
    assert (labels[1024:] >= 0).all()
    assert mask.sum().item() == 2048
    assert text.shape == (32,)


def test_evaluation_dataset_preserves_bursts():
    dataset = WoundLlamaEvaluationDataset(records(), prompts())
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first["targets"] == [4, 5, 6, 7]
    assert torch.equal(first["token_maps"][0], torch.zeros(256, dtype=torch.long))
    assert torch.equal(second["token_maps"][0], torch.full((256,), 10))
