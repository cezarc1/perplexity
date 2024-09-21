# Perplexity Calculator

A command-line tool to locally calculate the [Perplexity](https://en.wikipedia.org/wiki/Perplexity) (PPL) of a given text using a specified language model. _This is not to be confused with [Perplexity](https://www.perplexity.ai/), the search engine product._

This repo largely follows the code provided on the excellent [HuggingFace documentation on Perplexity](https://huggingface.co/docs/transformers/en/perplexity).

Supports cuda, mlx (mac m-series) and cpu inference.

## Coming Soon

Non-HF hosted models (OpenAI, Anthropic, Gemmini-series)

## Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. (Optional) If you plan to use the shell script, ensure you have `uv` installed. If not, the script will prompt you to install it if it's not found.

## Usage

### Option 1: Running as a Shell Script

1. Make the script executable:

   ```shell
   chmod +x calculate_perplexity.sh
   ```

2. Run the script:

   ```shell
   ./calculate_perplexity.sh --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --text "It's simple: Overspecialize, and you breed in weakness. It's slow death."
   ```

### Option 2a: Running as a Python Script

Run the Python script directly with uv:

   ```shell
   uv run --with-requirements requirements.txt calculate_perplexity.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --text "It's simple: Overspecialize, and you breed in weakness. It's slow death."
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
   python calculate_perplexity.py --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" --text "It's simple: Overspecialize, and you breed in weakness. It's slow death."
   ```

## Arguments

- `--model_id`: The ID of the model to use (e.g., "meta-llama/Meta-Llama-3-8B")
- `--text`: The text to calculate perplexity on
- `--stride` (optional): The stride length to use for calculating perplexity (default: 512)

## Notes

- The shell script version uses `uv` to manage dependencies and run the Python script.
- The Python script version requires you to manually install the dependencies listed in `requirements.txt`.
- Make sure you have sufficient permissions to download and use the specified model on HuggingFace.
