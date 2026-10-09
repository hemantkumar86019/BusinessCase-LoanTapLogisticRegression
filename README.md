# LoanTap Personal Loan Default Prediction & Risk Analytics

## 📌 Executive Summary
This project delivers an end-to-end Machine Learning solution to predict loan default risks for **LoanTap**, an online fintech credit platform. Using a dataset of **396,030 borrower records**, the primary objective is to identify high-risk applicants before loan origination, balancing default detection with minimal loss of profitable loan approvals.

Through rigorous Exploratory Data Analysis (EDA), domain-specific feature engineering, and class-imbalance mitigation using **SMOTE**, a **Logistic Regression** model was trained and tuned. By optimizing the decision threshold to **~0.30**, the model achieves an optimal trade-off between Precision and Recall—maximizing default detection while mitigating credit risk exposure.

---

## 🎯 Business Problem & Objectives
* **Credit Risk Exposure:** Minimizing Non-Performing Assets (NPAs) caused by borrower defaults (`Charged Off` loans).
* **Class Imbalance:** Managing a highly imbalanced dataset where **~80.4%** of applicants fully pay their loans and only **~19.6%** default.
* **Optimal Threshold Strategy:** Shifting from the default 0.50 probability cutoff to a lower risk threshold (~0.30) to prioritize high-recall default capture.
* **Documentation & Delivery:** Formatting analytics into compact, publication-ready documentation adhering to strict submission constraints ($\le 50$ PDF pages and $\le 20\text{ MB}$ file size).

---

## 📂 Data Architecture & Dictionary Summary

| Feature Category | Features Included | Description & Analytical Value |
| :--- | :--- | :--- |
| **Loan Parameters** | `loan_amnt`, `term`, `int_rate`, `installment` | Principal requested, tenure (36/60 mos), interest rate, and monthly payment. |
| **Risk Grading** | `grade`, `sub_grade` | LoanTap internal risk tiers (Grades A–G, Sub-grades A1–G5); primary risk driver. |
| **Borrower Profile** | `emp_title`, `emp_length`, `annual_inc`, `home_ownership` | Occupation, job stability, gross annual earnings, and housing status. |
| **Leverage & Debt** | `dti`, `revol_bal`, `revol_util` | Debt-to-Income ratio, revolving balance, and credit utilization rate. |
| **Credit History** | `earliest_cr_line`, `open_acc`, `total_acc`, `mort_acc` | Credit age, active credit lines, total lifetime accounts, and mortgage counts. |
| **Derogatories** | `pub_rec`, `pub_rec_bankruptcies` | Public negative legal records and bankruptcy filings. |
| **Application Info** | `verification_status`, `purpose`, `title`, `application_type`, `address` | Income verification, borrowing intent, joint/individual status, and zip code. |

---

## 🔬 Exploratory Data Analysis & Key Insights

### 1. Bivariate & Target Risk Drivers
* **Loan Grade & Sub-grade:** Default rates increase monotonically from **Grade A (~6%)** to **Grade G (~50%)**, validating internal risk-based pricing.
* **Loan Term:** 60-month loans exhibit double the default rate of 36-month loans (**~32% vs ~16%**) due to long-horizon economic risk.
* **Loan Purpose:** **`small_business`** loans carry the highest risk (**~30% default**), whereas asset/lifestyle loans (**`car`**, **`wedding`**) show the lowest defaults.
* **Income & Leverage:** Defaulted borrowers exhibit higher median Debt-to-Income (**DTI ~19.5% vs ~16.8%**) and higher revolving utilization (**~59% vs ~53%**).
* **Verification Selection Bias:** Verified applicants show a slightly higher default rate (~22%) than unverified applicants (~14%) because LoanTap selectively enforces strict verification on higher-risk or high-amount profiles.

### 2. Feature Skewness & Outlier Treatments
* **Categorical Artifacts (`term`, `loan_status`):** Zero IQR ($Q1=Q3$) mechanically flags minority classes; retained without trimming.
* **Zero-Inflated Risk Signals (`pub_rec`, `pub_rec_bankruptcies`):** Heavily zero-inflated ($>85\%$ at 0). Converted into binary flags ($>1.0 \rightarrow 1$) to capture repeat risk.
* **Monetary Features (`annual_inc`, `revol_bal`):** Right-skewed power-law distributions transformed using $\log(1+x)$ to compress variance.
* **Extreme Data Anomalies (`dti`, `revol_util`):** Operational penalty states and logging errors (e.g., $\text{DTI} = 9,999$, $\text{utilization} > 800\%$) capped at **100%**.

### 3. Collinearity & Feature Interactions
* **`loan_amnt` vs `installment` ($r = 0.95$):** Severe linear dependence driven by amortization math; mitigated via regularization and feature ratio engineering.
* **`open_acc` vs `total_acc` ($r = 0.68$):** Consolidated into a new activity ratio (`open_to_total_acc_ratio`).

---

## 🛠️ Feature Engineering Pipeline

* **Binary Risk Flags:** Created `pub_rec_flag`, `mort_acc_flag`, and `pub_rec_bankruptcies_flag` for values $>1.0$ to isolate high-risk repeat derogatories.
* **Temporal Features:** Derived `issue_year` and `issue_month` from `issue_d` to track macro credit cycles and seasonality.
* **Credit History Age:** Computed `credit_history_year` ($\text{issue\_d} - \text{earliest\_cr\_line}$) to quantify credit maturity at application.
* **Geographic Parsing:** Extracted 5-digit zip codes from `address` and mapped them to regional economic indicators.
* **Financial Ratio Features:**
  $$\text{Debt Service Burden} = \frac{\text{installment} \times 12}{\text{annual\_inc}}$$
  $$\text{Account Activity Ratio} = \frac{\text{open\_acc}}{\text{total\_acc}}$$

---

## ⚙️ Model Architecture & Threshold Optimization

1. **Preprocessing & Scaling:** Standardized numerical features and one-hot encoded categorical variables. Missing values imputed (median for continuous, mode/missing-category for strings).
2. **Class Imbalance Handling:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on training splits to balance default representation.
3. **Primary Model:** Trained a **Logistic Regression** classifier with L2 regularization to ensure stable coefficient estimations despite collinearity.
4. **Optimal Decision Threshold (~0.30):**
   * Standard $0.50$ threshold yields low default recall due to class imbalance.
   * Lowering the classification threshold to **~0.30** maximizes default capture (Recall) while maintaining acceptable precision, optimizing LoanTap's net financial risk exposure.

---

## 💼 Business Impact & Recommendations

* **Underwriting Policy:** Automatically reject or require manual executive sign-off for applicants possessing $\ge 2$ public derogatory marks or bankruptcies.
* **Product-Specific Pricing:** Apply strict credit caps and higher interest premiums on 60-month terms and `small_business` loan requests.
* **Early Warning System:** Monitor borrowers whose credit card utilization breaches **75%** post-origination as a key signal for proactive outreach.
* **Deployment Workflow:** Embed the Logistic Regression threshold model ($\text{Threshold} = 0.30$) into LoanTap’s origination API for real-time risk scoring.

---

## 🧰 Tech Stack
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Environment:** Visual Studio Code / Jupyter Notebook