# Fed-PEKD: Federated Pseudo-Embedding Knowledge Distillation
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


This repository contains the Python implementation of the Fed-PEKD algorithm, a novel approach to federated learning designed for privacy, communication efficiency, and robustness against non-IID data distributions. 🔐📡📊

## 🎯 Project Goal

The primary objective of Fed-PEKD is to enable collaborative training of a global model across distributed clients (e.g., mobile devices, hospitals) without sharing raw private data. It aims to achieve this while being:

* **Data-Free:** No raw client data leaves the local device.
* **Communication-Efficient:** Minimizes data transmission compared to methods like FedAvg.
* **Privacy-Preserving:** Protects client data by sharing aggregated summaries or generated synthetic data.
* **Robust to Non-IID Data:** Performs stably even when clients have different data distributions.

---
## 💡 Core Idea: The `prototype-gmm` Method

After several iterations and overcoming significant challenges, the final and most successful version of Fed-PEKD implemented here uses a hybrid **Prototype-GMM** approach:

1.  **Local Training:** Each client trains a local feature extractor (e.g., a CNN backbone) on its private data, keeping a globally shared classifier head frozen. 🧠
2.  **Prototype Extraction:** Instead of sending raw data or embeddings, each client calculates the *mean* feature embedding (**prototype**) and the *mean* output logits for each data class it possesses locally. 📊➡️🔢
3.  **Efficient Communication:** Clients transmit only these prototypes (e.g., 10 feature vectors + 10 logit vectors) to the central server. This is **ultra-low bandwidth**. 📉
4.  **Server-Side Generation:** The server collects prototypes from all clients. For each class, it fits a simple Gaussian Mixture Model (GMM) using the received prototypes for that class as input. It then generates a large synthetic (pseudo-embedding) dataset by sampling from these GMMs. 👻 A robust fallback mechanism handles cases where no prototypes are received for a class, preventing crashes.
5.  **Knowledge Distillation:** The server trains its global classifier head on this large synthetic dataset, using the mean prototype logits received from clients as the "soft teacher" signal (Knowledge Distillation). 🧑‍🏫
6.  **Global Update:** The server sends the updated global classifier head back to all clients for the next round. 🌍

---
## 🚧 Development Journey: Challenges & Solutions

The path to the final `prototype-gmm` method involved overcoming several key challenges:

1.  **"Untrained Body" Problem:** Initial attempts to train a full global model on the server failed because only the final layer received gradients.
    * **Solution:** Adopted a **"Classifier-Only"** design where clients train extractors and the server trains only the classifier. ✅
2.  **Noisy Statistics:** Generating pseudo-embeddings from simple mean/variance statistics proved too noisy and ineffective.
    * **Solution:** Moved to methods using higher-quality information derived from client data (real embeddings or prototypes). ✅
3.  **GMM Instability:** Fitting GMMs to raw embedding samples was prone to crashing due to insufficient data or feature collapse.
    * **Solution:** Switched to fitting GMMs on more stable **prototypes** and added robust error handling. ✅
4.  **Server Overfitting ("Prototypes Only" Collapse):** Training the server directly on the small set of received prototypes led to severe overfitting and model collapse.
    * **Solution:** Used prototypes as a *basis* to **generate a large synthetic dataset** via GMMs, preventing server overfitting. ✅
5.  **"0-Prototype" Decay/Crash:** Sparse data distributions (even `alpha=1.0`) meant sometimes no client had data for a class, leading to generator failures, `NaN` poisoning, and accuracy collapse.
    * **Solution:** Implemented a **robust fallback** in the generator (using a global mean) and made the server **skip updates** for missing classes, preventing crashes and ensuring stability. ✅

---
## 📊 Results Summary (FashionMNIST, Dirichlet `alpha=5.0`)

Our final, robust `prototype-gmm` method demonstrated a compelling trade-off:

| Method                       | Peak Accuracy | Stability            | Privacy      | Communication |
| :--------------------------- | :------------ | :------------------- | :---------   | :------------ |
| FedAvg (`alpha=1.0`)         | ~86%          | Very Poor            | None         | Very High     |
| **Fed-PEKD `prototype-gmm`** | **~70%**      | **Code=OK, Acc=Good**| **Excellent**| **Ultra Low** |
| Fed-PEKD `real-samples`      | ~61%          | Excellent            | Medium       | Medium        |

**Key Finding:** Fed-PEKD (`prototype-gmm`) achieves **significant accuracy (~70%)** while being **data-free**, **ultra-communication-efficient** (~150x less than FedAvg), and **more stable** than FedAvg, although it showed sensitivity to extreme data sparsity in earlier tests (`alpha=1.0`). It clearly **outperforms** the less private `real-samples` baseline.

---
## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Krishal743/Fed-PEKD.git
    cd fed-pekd
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure `requirements.txt` includes `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`)*

---
## ▶️ Running Experiments

Use the `main.py` script with command-line arguments. Key arguments:

* `--method`: `fedavg` or `fed-pekd`
* `--agg_method`: (for `fed-pekd`) `real-samples` or `prototype-gmm` (and others we tested)
* `--exp_name`: A name for the output files (e.g., "my_test")
* `--num_rounds`: Number of communication rounds
* `--local_epochs`: Client local training epochs
* `--distill_epochs`: Server distillation epochs (for `fed-pekd`)
* `--alpha`: Dirichlet distribution alpha (e.g., `1.0`, `5.0`)

**Example Commands:**

1.  **Run FedAvg Baseline (`alpha=1.0`):**
    ```bash
    python main.py --method fedavg --exp_name "fedavg_baseline_alpha1" --num_rounds 40 --local_epochs 2 --alpha 1.0
    ```

2.  **Run Best Fed-PEKD (`prototype-gmm`, `alpha=5.0`):**
    ```bash
    python main.py --method fed-pekd --agg_method prototype-gmm --exp_name "fedpekd_protogmm_alpha5" --num_rounds 50 --local_epochs 5 --distill_epochs 30 --alpha 5.0
    ```

3.  **Run Fed-PEKD (`real-samples` baseline, `alpha=1.0`):**
    ```bash
    python main.py --method fed-pekd --agg_method real-samples --exp_name "fedpekd_realsamples_alpha1" --num_rounds 50 --local_epochs 5 --distill_epochs 30 --alpha 1.0 --pekd_samples 500
    ```

Results (CSVs and final model `.pth` files) will be saved in the `results/` directory.

---
## 📊 Generating Plots & Evaluations

1.  **Accuracy vs. Rounds Comparison:**
    * Make sure the relevant CSV files are in the `results/` folder.
    * Modify `plot_results.py` to point to the correct filenames.
    * Run: `python plot_results.py`
    * Output: `results/ALL_VARIANTS_Comparison.png`

2.  **Confusion Matrix:**
    * Ensure the final model `.pth` files are saved in `results/`.
    * Run `evaluate_model.py`, pointing to the correct model files and specifying the method and alpha used during training.
    * **Example (Fed-PEKD ProtoGMM alpha=5.0):**
        ```bash
        python evaluate_model.py --method fed-pekd --classifier_path results/final_classifier_fedpekd_protogmm_alpha5.pth --extractor_path results/final_client0_model_fedpekd_protogmm_alpha5.pth --alpha 5.0 --exp_name fedpekd_protogmm_alpha5
        ```
    * Output: `results/cm_fedpekd_fedpekd_protogmm_alpha5.png` (or similar name)

---
## Further work

* Implement explicit **Communication Cost** tracking.
* Conduct formal **Privacy Analysis** (e.g., inversion attacks).
* Add **Ablation Studies** (e.g., effect of confidence weighting on `real-samples`).
* Implement **Projection Heads** to enable true heterogeneous client models.
* Scale experiments to **CIFAR-10**.
* Explore advanced generative techniques (e.g., contrastive loss).
