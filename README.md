# 🛡️ AI HEIST: Model Stealing Using API Queries

> **Mini Project – Final Phase | 2025–2026**  
> **Department of Computer Science & Engineering**  
> **JSS Science and Technology University (JSS STU), Mysuru**  
>
> **Project Team:** Sthuthi Sheela · Hemalatha H B · Bhavani · Vaishnavi Panchal  
> **Project Guide:** Prof. Shruthi N M  
> **Global Alignment:** UN Sustainable Development Goal 9 (Industry, Innovation & Infrastructure)

---

## 📌 Abstract & Research Overview

Machine Learning (ML) models deployed behind cloud REST APIs are conventionally presumed secure under the premise that black-box isolation hides their weights, hyperparameters, architectures, and proprietary training datasets.

**AI HEIST** demonstrates that black-box deployment alone is insufficient to protect algorithmic intellectual property. Operating with **zero access** to the original model or training data, an adversary generates **bounded synthetic queries**, streams automated requests to a public prediction endpoint, captures **hard binary (Yes/No) responses**, and trains a local **Random Forest clone** that replicates a proprietary **Logistic Regression** target model with **88.5%+ Target Agreement Rate**.

```
┌─────────────────────────┐          Bounded POST Queries (500+)          ┌─────────────────────────┐
│     ADVERSARY /         │ ─────────────────────────────────────────────> │       TARGET API        │
│     LOCAL CLONE         │                                                │  (Logistic Regression)  │
│  (Random Forest Clone)  │ <───────────────────────────────────────────── │    /predict Endpoint    │
└─────────────────────────┘             Hard Binary (Yes / No)             └─────────────────────────┘
```

---

## 🎯 Key Attack Characteristics

- **Zero Model Internals Required:** Zero access to model weights, source code, architecture, or training data.
- **Hard Labels Only:** API returns strictly binary (`Yes` / `No`) attrition classifications — no probabilities or confidence scores are provided to the attacker.
- **Architectural Divergence:** The clone uses a non-linear ensemble (**Random Forest**) to reverse-engineer a linear sigmoid boundary (**Logistic Regression**).
- **80%+ Target Agreement Rate:** Demonstrates that 800+ out of 1,000 queries can be cloned with exact mathematical parity.

---

## 🏗️ 10-Stage Project Pipeline

```
  01. Dataset Collection (Kaggle Employee Attrition)
        │
  02. Data Preprocessing (StandardScaler & OneHotEncoder)
        │
  03. Train Logistic Regression (Algorithm 1: Target Model)
        │
  04. Deploy Flask REST API (/predict & /api/predict)
        │
  05. Bounded Synthetic Query Generation (Slide 10 Distributions)
        │
  06. High-Frequency API Probing (500 - 5,000 Streams with Retry Logic)
        │
  07. Create Stolen Dataset (stolen_dataset.csv flushes every 50 queries)
        │
  08. Train Random Forest Clone (Algorithm 2: Substitute Model)
        │
  09. Mathematical Parity Evaluation (Accuracy, Precision, Recall, F1, AUC, Agreement Rate ★)
        │
  10. Web & Security Analytics Dashboard (Flask + HTML5 / Bootstrap / Chart.js)
```

---

## 📊 Dataset & 10 Key Features

- **Source:** Kaggle – Employee Attrition Dataset (10,000 Employee Records)
- **Purpose:** Predict employee "flight risks" before turnover materializes
- **Target Variable:** `Attrition` → `Yes` (departed) / `No` (retained)

| # | Feature | Domain | Type | Why It Matters |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `Age` | Demographic | Numerical (18–65) | Career stage → younger employees show higher mobility |
| **2** | `Gender` | Demographic | Categorical | Demographic baseline for diversity stratification |
| **3** | `Department` | Organizational | Categorical | Identifies high-attrition cultural pockets (Sales, HR, IT) |
| **4** | `Job Role` | Job-Related | Categorical | Links turnover to external market demand and role mobility |
| **5** | `Monthly Income` | Compensation | Numerical ($1,500–$25,000) | Primary financial retention lever; lower income → flight risk |
| **6** | `Years at Company` | Organizational | Numerical (0–40, $\le \text{Age} - 18$) | Identifies burnout milestones (1-yr, 3-yr tenure peaks) |
| **7** | `Job Satisfaction` | Satisfaction | Categorical / Ordinal (1–4) | Low score (1/4) = imminent voluntary departure trigger |
| **8** | `Work-Life Balance` | Satisfaction | Categorical / Ordinal (1–4) | Poor balance accelerates resignation timelines |
| **9** | `Overtime` | Job-Related | Categorical (Yes / No) | **Single strongest binary trigger for attrition** |
| **10** | `Num Companies Worked` | Job-Related | Numerical (0–10) | Proxy for job-hopping propensity and external mobility |

### 🔥 Top 3 Most Impactful Features (Slide 6)
1. **⏱ #1 Trigger — Overtime:** Employees logging regular overtime show dramatically higher flight risk. Strongest predictor in both target and clone models.
2. **💰 #2 Continuous — Monthly Income:** Lower income brackets strongly correlate with higher attrition probability. Consistently ranks as the top continuous variable.
3. **⚖ #3 Ordinal — Work-Life Balance:** Scores of 1 or 2 (out of 4) translate directly to accelerated resignations; key split point for decision boundaries.

---

## 🤖 Algorithms & Mathematical Foundations

### Algorithm 1: Logistic Regression – Target Model
- **Role:** Proprietary enterprise model deployed behind the API.
- **Linear Combination:**
  $$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
- **Sigmoid Activation Function:**
  $$P(Y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Hard-Label Decision Rule:**
  $$\hat{y} = \begin{cases} 1 \quad (\text{Attrition / Flight Risk}) & \text{if } P \ge 0.5 \\ 0 \quad (\text{Retained}) & \text{if } P < 0.5 \end{cases}$$
- **Why Chosen:** Smooth, linear boundaries are ideal for testing whether a non-linear clone can reverse-engineer them from binary labels alone.

### Algorithm 2: Random Forest – Substitute Clone Model
- **Role:** Attacker's local surrogate model trained purely on stolen API responses.
- **Method:** Bootstrap Aggregation (Bagging) + Random Subspace Method.
- **Bootstrap Samples:** Draw $B$ samples $\to$ train $B$ independent trees:
  $$\{T_1(X), T_2(X), \dots, T_B(X)\}$$
- **Node Splitting (Gini Impurity):**
  $$I_G(t) = 1 - \sum_{i=1}^C p_i^2 \quad \text{split on } m = \sqrt{d} \text{ random features per node}$$
- **Ensemble Majority Voting:**
  $$Y_{\text{substitute}} = \text{majority\_vote}\{T_1(X), T_2(X), \dots, T_B(X)\}$$
- **Why Chosen:** Non-linear, high-capacity ensemble reconstructs complex boundaries from hard-label binary inputs alone.

---

## ⚡ Attack Mechanics

### Bounded Synthetic Query Generation (Slide 10)
Operating with **zero access to real employee records**, the attack generates realistic profiles bounded by domain rules:
- **Age:** $\text{Uniform}(18, 65)$
- **Monthly Income:** $\text{Uniform}(1500, 25000)$
- **Years at Company:** $\text{Uniform}(0, 40)$ constrained logically by $\text{Years} \le \text{Age} - 18$
- **Overtime:** $\text{Bernoulli}(p = 0.30)$
- **Job Satisfaction & Work-Life Balance:** $\text{Discrete Uniform}(1, 4)$
- **Num Companies Worked:** $\text{Discrete Uniform}(0, 10)$

### High-Frequency Stream Pipeline (Slide 11)
- **Retry Logic:** Automated backoff with exception handling for network timeouts during 1,000+ query loops.
- **Checkpoint Saves:** Data flushed to `attack/stolen_dataset.csv` every 50 queries to prevent data loss.
- **Attack Summary:** `attack/attack_summary.json` logs total queries, success rates, and average endpoint latencies.
- **Auto-Trigger:** JSON summary auto-triggers the Random Forest training loop and syncs dashboards.

---

## 📈 Evaluation & Results Comparison

### 6 Core Performance Metrics
- **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$
- **Precision:** $TP / (TP + FP)$
- **Recall:** $TP / (TP + FN)$
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve
- **Target Agreement Rate ★:** $\frac{\text{Identical Predictions}}{\text{Total Queries}}$ (**Primary Metric**)

### Performance Parity Benchmark (Slide 13)

| Metric | Target Model (Logistic Regression) | Clone Model (Random Forest) |
| :--- | :---: | :---: |
| **Accuracy** | **82.4%** | **80.8%** |
| **Precision** | **81.1%** | **79.3%** |
| **Recall** | **79.5%** | **78.1%** |
| **F1-Score** | **80.3%** | **78.7%** |
| **ROC-AUC** | **0.84** | **0.81** |
| **Target Agreement Rate ★** | — | **88.5%** |

> **Result:** The clone model achieved an **88.5% Target Agreement Rate** using only 500+ binary-label queries with zero internal model access.

---

## ⚖️ Feature Alignment Analytics (Slide 15)

Without knowing the target's internal structure or weights, the clone learned the **exact same feature sensitivities**:
1. **#1 Overtime:** Both models isolate regular overtime as the primary positive driver for flight risk classification.
2. **#2 Monthly Income:** Lower income brackets recognized as the primary continuous indicator of attrition probability.
3. **#3 Work-Life Balance:** Clone decision trees heavily use low WLB scores (1 or 2) to mirror the target's threshold splits.

---

## 🛡️ Active Security Defenses (Future Work — Slide 17)

1. **Rate Limiter:** Restricts high-frequency queries per client/IP, making 1,000+ extraction sweeps impractical.
2. **Anomaly Detection:** Analyzes input distributions to identify automated synthetic scanning behavior.
3. **Adaptive Misinformation:** Adds calculated label perturbations near decision boundaries when extraction scanning is detected, corrupting the stolen dataset.
4. **Multi-Class & Deep Models:** Extension to multi-grade risk classifications (Low/Med/High) and testing against Deep Neural Networks and XGBoost.

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Sthuthi1310/Employee-Attrition-Model-Stealing.git
cd Employee-Attrition-Prediction

# Virtual environment setup
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Train Target Model (Algorithm 1)
```bash
python employee_attrition_pipeline.py
```
*Generates `models/target_model.pkl`, confusion matrix, and feature sensitivities.*

### 3. Start Flask REST API Endpoint
```bash
python web/app.py
```
*Serves the API at `http://127.0.0.1:5000/predict` and `http://127.0.0.1:5000/api/predict`.*

### 4. Run Model Stealing Attack & Retrain Clone (Algorithm 2)
```bash
# In a separate terminal
python attack/collect_responses.py --count 500 --auto-train
```
*Streams 500 bounded queries, checkpoints to `attack/stolen_dataset.csv`, and trains `models/substitute_model.pkl`.*

### 5. Access Interactive Web Dashboards (Stage 10)
- **Employee Attrition Prediction UI:** Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
- **Model Stealing Security Analysis Dashboard:** Open [http://127.0.0.1:5000/analysis](http://127.0.0.1:5000/analysis) to inspect live parity metrics, agreement progression curves, confusion matrices, and feature importance alignments.

---

## 📂 Project Directory Structure

```
Employee-Attrition-Prediction/
├── attack/
│   ├── logs/                       # Attack run execution logs
│   ├── surrogate/                  # Substitute training modules
│   │   ├── config.py               # Surrogate parameters
│   │   └── train_substitute.py     # Algorithm 2: Random Forest clone trainer
│   ├── attack_summary.json         # Attack query statistics
│   ├── collect_responses.py        # High-frequency API query streamer
│   ├── config.py                   # Bounded synthetic distributions
│   ├── generate_queries.py         # Realistic employee profile generator
│   ├── run_attack_pipeline.py      # End-to-end extraction orchestrator
│   └── stolen_dataset.csv          # Checkpointed captured binary responses
├── data/
│   └── raw/
│       └── employee_attrition_dataset_10000.csv
├── models/
│   ├── target_model.pkl            # Proprietary Logistic Regression model
│   └── substitute_model.pkl        # Stolen Random Forest clone model
├── results/
│   ├── attack/                     # Substitute evaluation charts & metrics
│   │   ├── agreement_analysis.png  # Target agreement analysis
│   │   ├── analysis_summary.json   # Full experiment metrics
│   │   ├── model_comparison.csv    # Target vs Clone performance table
│   │   └── roc_comparison.png      # ROC comparison curves
│   ├── class_distribution.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── victim_confusion_matrix.png
├── utils/                          # Helper functions, metrics, preprocessing
├── web/
│   ├── static/                     # CSS & JS assets
│   ├── templates/                  # Flask HTML dashboards (index & analysis)
│   ├── analysis_data.py            # Analysis aggregation service
│   └── app.py                      # Flask REST API microservice
├── employee_attrition_pipeline.py  # Target model training & evaluation
├── interactive_predictor.py        # Console-based interactive test tool
├── requirements.txt                # Python dependencies
└── README.md                       # Comprehensive project documentation
```

---

## 📜 Citation & References

```bibtex
[1] J. B. Truong, P. Maini, R. J. Walls, and N. Papernot, "Data-Free Model Extraction," CVPR, 2022.
[2] M. Jagielski et al., "High-Fidelity Extraction of Neural Network Templates," ACM CCS, 2020.
[3] T. Orekondy, B. Schiele, and M. Fritz, "Knockoff Nets: Stealing Black-Box Mobile Models," CVPR, 2019.
[4] Y. Wang et al., "Training a Secure Model Against Data-Free Model Extraction," LNCS, 2023.
```

---

> *"Security is not about hiding everything — it's about making extraction expensive."*  
> **AI HEIST · Final Phase Mini Project · JSS STU, Mysuru · 2025–2026**