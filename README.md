
# 📊 Linear Regression Playground

An interactive **Streamlit web app** to upload CSV data, select features and target, visualize regression, and test predictions.
Supports both **manual line fitting (w & b sliders)** for 1 feature and **automatic linear regression** for multiple features.
You can also **download the trained model** as a `.pkl` file.

---

## 🚀 Features

* Upload your own CSV dataset.
* Choose **target column (y)** and **feature column(s) (X)**.
* If you select **one feature**:

  * Adjust **Weight (w)** and **Bias (b)** manually with sliders.
  * Visualize how the regression line changes.
* Run **auto-fit regression** with scikit-learn.
* View regression metrics:

  * Equation of best fit
  * R² score
  * Mean Squared Error (MSE)
* Make **custom predictions** by entering feature values.
* **Download the trained model** (`.pkl`) for reuse.

---

## 🛠️ Tech Stack

* [Streamlit](https://streamlit.io/) (frontend + backend)
* [Pandas](https://pandas.pydata.org/) (data handling)
* [Matplotlib](https://matplotlib.org/) (visualization)
* [Scikit-learn](https://scikit-learn.org/) (linear regression)
* [NumPy](https://numpy.org/) (math operations)

---

## 📂 Project Structure

```
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── example_multifeature.csv  # Sample dataset
```

---

## 📦 Installation & Usage

1. **Clone the repo**

   ```bash
   git clone https://github.com/santhanakrishnan2004/ml-linearregression.git
   cd ml-linearregression
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

5. Open the app in your browser at
   👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Example Dataset

You can use the included **`example_multifeature.csv`**:

* `Hours_Studied`
* `Sleep_Hours`
* `Coffee_Cups`
* `Exam_Score` (target)

---

## 💡 Future Improvements

* Add correlation heatmap and pairplots for better feature visualization.
* Support polynomial regression.
* Save/load model + preprocessing pipeline.

---


