# LoRec

## Environment
- **python >= 3.8** (Required for compatibility and correct JSON file ordering)
- numpy >= 1.22.2
- scikit-learn >= 1.0.2
- scipy >= 1.8.0
- torch >= 1.10.1

If you want to use LLM for encoding:
- transformers >= 4.35.2
- tqdm  >= 4.66.1


## Usage (Quick Start)
1. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script with the desired backbone model and dataset:

    ```bash
    python main.py --model=<backbone model> --dataset=<dataset>
    ```

   Replace `<backbone model>` with the name of your model, and `<dataset>` with the name of your dataset.

3. To utilize LLM for encoding, ensure you have the optional dependencies installed. If you require embeddings of items and users encoded by a Large Language Model, please contact us for more information.
