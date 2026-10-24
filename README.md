# 🪙 Gold Authenticity Prediction | ML Project

A beginner-friendly Machine Learning project that uses a **Random Forest Classifier** to predict whether a gold sample matches a *Genuine* or *Fake* demonstration class.

> ⚠️ **Disclaimer:** Trained on synthetic data for learning purposes only. It cannot authenticate real gold.

## ✨ Features

* 🤖 Random Forest classification
* 📊 Dataset generation and model evaluation
* 🎯 Prediction using karat, purity, density, and hallmark
* 💾 Save and load the trained model
* 📈 Accuracy and classification report

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `Joblib`

## 📁 Project Structure

```text
gold_authenticity/
├── backend/train_model.py
├── dataset/
│   ├── golddataset.csv
│   ├── gold_training.csv
│   └── hallmarks.json
├── models/gold_model.pkl
├── test_model.py
└── README.md
```

## 🚀 Run the Project

**1. Install dependencies**

```bash
python -m pip install pandas scikit-learn joblib
```

**2. Train the model**

```bash
python backend/train_model.py
```

**3. Test predictions**

```bash
python test_model.py
```

Enter the karat, purity, density, and hallmark status when prompted.

## 📊 Model Performance

| Metric             |                  Result |
| ------------------ | ----------------------: |
| Algorithm          |           Random Forest |
| Dataset            | 1,000 synthetic samples |
| Training / Testing |               80% / 20% |
| Test Accuracy      |                    99%* |

<sub>*Observed result from one run on synthetic data. It does not represent real-world gold authentication accuracy.</sub>

## 🔮 Future Improvements

* Use verified gold measurement data
* Clean and validate the dataset
* Compare different ML algorithms
* Improve model evaluation

---

**Built for learning Machine Learning with Python.** 💻
