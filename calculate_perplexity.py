import argparse

import torch
from tqdm import tqdm
from transformers import pipeline


def calculate_perplexity(model_id: str, text: str, stride: int = 512) -> float:
    """
    Calculate perplexity of a model on given text using Hugging Face's
    pipeline and accelerate.
    Adapted from https://huggingface.co/docs/transformers/pipeline_tutorial

    Args:
        model_id (str): The HF model ID (e.g., 'meta-llama/Meta-Llama-3-8B')
        text (str): The text to calculate perplexity on
        stride (int): The stride to use for calculating perplexity

    Returns:
        float: The perplexity of the model on the given text
    """
    model_pipeline = pipeline("text-generation",
                              model=model_id,
                              tokenizer=model_id,
                              trust_remote_code=False,
                              device_map="auto")
    assert model_pipeline.tokenizer is not None
    encodings = model_pipeline.tokenizer(text, return_tensors="pt")
    max_length = model_pipeline.model.config.max_position_embeddings
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride),
                          desc="Calculating perplexity...",
                          leave=False,
                          colour="green"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        # We don’t want the log-likelihood for the tokens we’re just treating
        # as context to be included in our loss, so we can set these targets
        # to -100 so that they are ignored.
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model_pipeline.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Calculate perplexity of a model on given text using Hugging Face's "
        "pipeline and accelerate.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The model ID (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct')")
    parser.add_argument("--text",
                        type=str,
                        required=True,
                        help="The text to calculate perplexity on")
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="The stride length to use for calculating perplexity")
    args = parser.parse_args()
    perplexity = calculate_perplexity(args.model_id, args.text, args.stride)
    print(f"Perplexity: {perplexity:.2f}")
