# Harnessing ML to Break the Unclonable Code (PUFs)

This project explores **Machine Learning modeling of Multi-Level Physically Unclonable Functions (ML-PUFs)** — hardware primitives often described as *silicon fingerprints* that provide device-level uniqueness and security.  
The work was done as part of **CS771: Introduction to Machine Learning (IIT Kanpur, 2025)**.

---

## 🚀 Project Highlights
- Derived a **tensor (Khatri–Rao) feature map** to linearize the XOR of arbiter PUF responses.  
- Implemented **linear classification models** (`LogisticRegression`, `LinearSVC`) using scikit-learn.  
- Achieved accurate prediction of ML-PUF responses from challenge–response pairs.  
- Extended work to **arbiter PUF delay inversion**, recovering feasible non-negative delays from learned linear models.  

---

## 📂 Repository Structure
- `mp1.pdf` → Report with mathematical derivations, experiments, and analysis.  
- `submit.py` → Python implementation (`my_map`, `my_fit`, `my_decode`).  

---

## 🛠️ Tech Stack
- Python, NumPy, scikit-learn  
- Feature Engineering, Linear Models  
- Hardware Security, Physically Unclonable Functions  

---

## 📖 References
- Assignment statement and datasets: CS771, IIT Kanpur (2025).  
- Related concept: [Khatri–Rao product (Wikipedia)](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product).  

---

## 🔗 Author
Developed by *Rajender* as part of coursework under **Prof. Purushottam Kar, IIT Kanpur**.
