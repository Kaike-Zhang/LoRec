# LoRec
This paper has been accepted by SIGIR24. [Link to the paper on Arxiv](https://arxiv.org/pdf/2401.17723) / [Link to the paper on ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3626772.3657684)

## Authors
- **Kaike Zhang**
- Qi Cao
- Yunfan Wu
- Fei Sun
- Huawei Shen
- Xueqi Cheng

## Abstract
Sequential recommender systems stand out for their ability to capture users' dynamic interests and the patterns of item transitions. However, the inherent openness of sequential recommender systems renders them vulnerable to poisoning attacks, where fraudsters are injected into the training data to manipulate learned patterns. Traditional defense methods predominantly depend on predefined assumptions or rules extracted from specific known attacks, limiting their generalizability to unknown attacks. To solve the above problems, considering the rich open-world knowledge encapsulated in Large Language Models (LLMs), we attempt to introduce LLMs into defense methods to broaden the knowledge beyond limited known attacks. We propose **LoRec**, an innovative framework that employs **L**LM-Enhanced Calibration to strengthen the r**o**bustness of sequential **Rec**ommender systems against poisoning attacks. LoRec integrates an LLM-enhanced CalibraTor (LCT) that refines the training process of sequential recommender systems with knowledge derived from LLMs, applying a user-wise reweighting to diminish the impact of attacks. Incorporating LLMs' open-world knowledge, the LCT effectively converts the limited, specific priors or rules into a more general pattern of fraudsters, offering improved defenses against poisons. Our comprehensive experiments validate that LoRec, as a general framework, significantly strengthens the robustness of sequential recommender systems. 

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

3. To utilize LLM for encoding, ensure you have the optional dependencies installed. If you require embeddings of items and users encoded by a Large Language Model, feel free to contact us (at kaikezhang99@gmail.com or zhangkaike21s@ict.ac.cn).


## Citation
If you find our work useful, please cite our paper using the following BibTeX:

```bibtex
@article{zhang2024lorec,
	author = {Zhang, Kaike and Cao, Qi and Wu, Yunfan and Sun, Fei and Shen, Huawei and Cheng, Xueqi},
	journal = {ArXiv},
	year = {2024},
	pages = {},
	publisher = {},
	title = {LoRec: Large {Language} {Model} for {Robust} {Sequential} {Recommendation} against {Poisoning} {Attacks}},
	volume = {abs/2401.17723},
}

@inproceedings{zhang2024lorecSIGIR,
  title={LoRec: Combating Poisons with Large Language Model for Robust Sequential Recommendation},
  author={Zhang, Kaike and Cao, Qi and Wu, Yunfan and Sun, Fei and Shen, Huawei and Cheng, Xueqi},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1733--1742},
  year={2024}
}
