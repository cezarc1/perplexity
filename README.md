# Perplexity Calculator

A command-line tool to locally calculate the [perplexity](https://en.wikipedia.org/wiki/Perplexity) (PPL) of a given text using a specified language model.

>...perplexity is a measure of uncertainty in the value of a sample from a discrete probability distribution. The larger the perplexity, the less likely it is that an observer can guess the value which will be drawn from the distribution.

_This is not to be confused with [Perplexity](https://www.perplexity.ai/), the search engine product._

This repo largely follows the code provided on the excellent [HuggingFace documentation on perplexity](https://huggingface.co/docs/transformers/en/perplexity).

Supports cuda, mlx (mac m-series) and cpu inference on recurrent llms (Llama, Mistral, etc) and encoder-decoder LLMs (BERT).

## Coming Soon

Non-HF hosted models (OpenAI, Anthropic, Gemmini-series)

Masked LLMs

## Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/cezarc1/perplexity
   cd perplexity
   ```

2. (Optional) If you plan to use the shell script, ensure you have `uv` installed. If not, the script will prompt you to install it if it's not found. See [here](https://github.com/astral-sh/uv?tab=readme-ov-file#highlights) for more info on `uv`.

## Usage

### Option 1: Running as a Shell Script

1. Make the script executable:

   ```shell
   chmod +x calculate_perplexity.sh
   ```

2. Run the script with text:

   ```shell
   ./calculate_perplexity.sh --model_id "google/gemma-2-2b-it" \
     --text "It's simple: Overspecialize, and you breed in weakness. It's slow death."
   ```

   Or with a text file:

   ```shell
   ./calculate_perplexity.sh --model_id "google/gemma-2-2b-it" \
     --text_file "path/to/your/text_file.txt"
   ```

### Option 2a: Running as a Python Script

Run the Python script directly with uv:

   ```shell
   uv run --with-requirements requirements.txt calculate_perplexity.py \
     --model_id "google/gemma-2-2b-it" \
     --text "It's simple: Overspecialize, and you breed in weakness. It's slow death."
   ```

   Or with a text file:

   ```shell
   uv run --with-requirements requirements.txt calculate_perplexity.py \
     --model_id "google/gemma-2-2b-it" \
     --text_file "path/to/your/text_file.txt"
   ```

### Option 2b: Running as a Python Script (venv)

   ```shell
   python -m venv .venv
   ```

   ```shell
   source .venv/bin/activate 
   ```

   ```shell
   pip install -r requirements.txt
   ```

   ```shell
   python calculate_perplexity.py --model_id "google/gemma-2-2b-it" \
     --text "It's simple: Overspecialize, and you breed in weakness. It's slow death."
   ```

   Or with a text file:

   ```shell
   python calculate_perplexity.py --model_id "google/gemma-2-2b-it" \
     --text_file "path/to/your/text_file.txt"
   ```

## Arguments

- `--model_id`: The ID of the model to use (e.g., "meta-llama/Meta-Llama-3-8B")
- `--model_type`: The type of model to use (choices: "recurrent", "encoder_decoder", "masked")
- `--text`: The text to calculate perplexity on
- `--text_file`: Path to a text file to calculate perplexity on
- `--stride` (optional): The stride length to use for calculating perplexity (default: 512)

Note: You must provide either `--text` or `--text_file`, but not both.

## Notes

- The shell script version uses `uv` to manage dependencies and run the Python script.
- The Python script version requires you to manually install the dependencies listed in `requirements.txt`.
- Make sure you have sufficient permissions to download and use the specified model on HuggingFace.
