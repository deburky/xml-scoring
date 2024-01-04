import pandas as pd
import numpy as np

url = (
    "https://drive.google.com/file/d/1q-_Py-1BoZHXwShM6puKFdZdSsGohL1Y/view?usp=sharing"
)
url = "https://drive.google.com/uc?id=" + url.split("/")[-2]

dataset = pd.read_csv(url, index_col=False)

# Define cut-off
CUT_OFF = 600

# Select approved loans
dataset['approved_model_1'] = np.where(dataset['woe_score'] > CUT_OFF, 1, 0)
dataset['approved_model_2'] = np.where(dataset['xgb_score'] > CUT_OFF, 1, 0)

# NPV * approval
npv_approval_model_1 = np.sum(dataset['approved_model_1'] * dataset['npv'])/len(dataset)
npv_approval_model_2 = np.sum(dataset['approved_model_2'] * dataset['npv'])/len(dataset)

print(f"Model 1 NPV for approvals: {npv_approval_model_1:.4f}")
print(f"Model 2 NPV for approvals: {npv_approval_model_2:.4f}")

# ROC analysis
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_ROC_NPV(dataset, score_column, target_column, npv_column):
    thresholds = sorted(np.linspace(0, 1, 40), reverse=True)
    TPRs = []
    FPRs = []
    NPVs = []
    Thresholds = []
    
    y_score = 1-dataset[score_column].values # probability of bad
    y_true = 1-dataset[target_column].values # 0 = bad, 1 = good
    npv_values = dataset[npv_column].values

    for th in thresholds:
        y_pred = np.where(y_score < th, 0, 1)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        TPRs.append(TPR)
        FPRs.append(FPR)
        Thresholds.append(th)
        
        # Calculate NPV using boolean indexing
        correctly_identified_npv = np.sum(npv_values[(y_pred == 1) & (y_true == 1)])
        NPVs.append(correctly_identified_npv)
    
    return pd.DataFrame(dict(
        TPR=TPRs,
        FPR=FPRs,
        NPV=NPVs,
        Thresholds=Thresholds  # Use thresholds in ascending order
    ))

# Get Results for Model 1
Model_1_results = calculate_ROC_NPV(
    dataset=dataset,
    score_column="woe_pd",
    target_column="is_bad",
    npv_column="npv",
)

# Get Results for Model 2
Model_2_results = calculate_ROC_NPV(
    dataset=dataset,
    score_column="xgb_pd",
    target_column="is_bad",
    npv_column="npv",
)

# Plotting
plt.plot(Model_1_results['Thresholds'], Model_1_results['NPV'], label="Model 1")
plt.plot(Model_2_results['Thresholds'], Model_2_results['NPV'], label="Model 2")
plt.xlabel("Thresholds")
plt.ylabel("NPV")
plt.gca().invert_xaxis()
plt.legend()
plt.show()