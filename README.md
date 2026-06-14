# ⚖️ AI Fairness Auditor

**AI Fairness Auditor** is an intelligent web-based platform designed to detect and analyze bias in datasets and machine learning model predictions. It helps developers, organizations, and researchers build **fair, transparent, and responsible AI systems** before deploying them in real-world scenarios.

Built as a submission for the **Google Solution Challenge 2026**, this project aligns with the vision of leveraging technology to create inclusive and ethical solutions for society.

---

## 🌍 Problem Statement

Artificial Intelligence increasingly influences critical decisions in areas such as hiring, lending, healthcare, and education. However, biased datasets and unfair models can unintentionally discriminate against certain groups, reinforcing existing inequalities.

Many developers lack accessible tools to evaluate fairness during the development lifecycle.

**AI Fairness Auditor addresses this challenge by enabling users to identify, quantify, and understand bias through an intuitive interface.**

---

## 💡 Solution

AI Fairness Auditor empowers users to:

* Upload datasets and assess fairness.
* Evaluate bias in machine learning predictions.
* Identify favored and disadvantaged groups.
* Measure fairness using standard metrics.
* Receive AI-generated explanations to interpret results.
* Make informed decisions to improve model fairness before deployment.

---

## 🚀 Features

### 📂 Dataset Bias Analysis

* Upload CSV datasets for fairness evaluation.
* Analyze distributions across sensitive attributes.
* Detect potential biases present in historical data.

### 🤖 Model Prediction Bias Detection

* Assess fairness in machine learning predictions.
* Compare outcomes between different demographic groups.
* Identify whether a model systematically favors certain groups.

### ⚖️ Fairness Metrics

* Calculate the **Disparate Impact Ratio (DIR)**.
* Measure approval rates across groups.
* Highlight favored and disadvantaged populations.
* Flag fairness concerns using established thresholds.

### 📊 Interactive Dashboard

* Clean and intuitive user interface built with Streamlit.
* Visual representation of fairness statistics.
* Easy-to-understand summaries and charts.

### 🧠 AI-Powered Insights

* Generate natural language explanations of bias findings.
* Help users understand the implications of fairness metrics.
* Provide actionable interpretation of results.

---

## 🛠️ Tech Stack

| Category              | Technologies        |
| --------------------- | ------------------- |
| Frontend              | Streamlit           |
| Programming Language  | Python              |
| Data Processing       | Pandas              |
| Machine Learning      | Scikit-learn        |
| AI Explanation Engine | Gemini              |
| Gemini Integration    | google-generativeai |
| Model Persistence     | Pickle              |

---

## 📂 Project Structure

```
bias-checker/
│
├── app.py
├── bias/
│   └── bias_check.py
│
├── model/
│   ├── model.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   └── train_model.py
│
├── data/
│   └── sample_data.csv
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bias-checker.git
cd bias-checker
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📊 How It Works

### Step 1: Upload Dataset

Upload a CSV file containing:

* Features used for prediction
* Sensitive attributes (e.g., gender)
* Prediction outcomes or target labels

### Step 2: Bias Analysis

The system automatically:

* Calculates approval rates.
* Computes the Disparate Impact Ratio.
* Detects favored and disadvantaged groups.
* Evaluates fairness thresholds.

### Step 3: Interpret Results

Users receive:

* Fairness metrics.
* Visual insights.
* Bias detection outcomes.
* AI-generated explanations.

---

## 📈 Fairness Metric Used

### Disparate Impact Ratio (DIR)

Disparate Impact Ratio is calculated as:

```
DIR = Approval Rate (Disadvantaged Group)
      ------------------------------------
      Approval Rate (Favored Group)
```

Interpretation:

* **DIR ≥ 0.80** → Fair outcome.
* **DIR < 0.80** → Potential bias detected.

The 80% rule is widely used as an initial fairness assessment criterion.

---

## 🎯 Impact

AI Fairness Auditor promotes:

* Responsible AI development.
* Transparency in decision-making systems.
* Early detection of unintended discrimination.
* Increased trust in machine learning applications.

By enabling fairness assessments before deployment, the platform contributes toward building AI systems that are more equitable and inclusive.

---

## 🏆 Google Solution Challenge 2026

This project was developed as part of the **Google Solution Challenge 2026**, an initiative encouraging students to solve real-world problems using Google technologies and the United Nations Sustainable Development Goals (SDGs).

### Relevant SDGs

* **SDG 10 – Reduced Inequalities**
* **SDG 16 – Peace, Justice and Strong Institutions**
* **SDG 9 – Industry, Innovation and Infrastructure**

---

## 🔗 Live Demo

https://bias-checker-8qar.onrender.com


---

## 👥 Team

**CodeN.JS**

Google Solution Challenge 2026 Submission


---

### Building Fair AI for Everyone ⚖️🤖✨
