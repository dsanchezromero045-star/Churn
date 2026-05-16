"""
Bank Churn v2 - heavy feature engineering + stacked ensemble.
Saves submission.csv with blended probabilities.
"""
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

DATA_DIR = '/home/user/Churn'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f'Train: {train.shape}  Test: {test.shape}')

# --- Basic cleaning ---
for df in (train, test):
    for col in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in ['Geography', 'Gender', 'Surname']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

train = train.dropna(subset=['Exited']).reset_index(drop=True)
train['Exited'] = train['Exited'].astype(int)


# --- Feature engineering ---
def engineer(df, train_ref=None):
    df = df.copy()
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScoreAge'] = df['CreditScore'] / (df['Age'] + 1)
    df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1)
    df['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)
    df['SalaryPerProduct'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100],
                             labels=[0, 1, 2, 3, 4]).astype(int)
    df['HighBalance'] = (df['Balance'] > 100000).astype(int)
    df['IsSenior'] = (df['Age'] >= 60).astype(int)
    df['Inactive_NoCard'] = ((df['IsActiveMember'] == 0) & (df['HasCrCard'] == 0)).astype(int)
    df['Geo_Gender'] = df['Geography'].astype(str) + '_' + df['Gender'].astype(str)
    df['LowTenure_HighBalance'] = ((df['Tenure'] <= 2) & (df['Balance'] > 100000)).astype(int)
    return df

train_fe = engineer(train)
test_fe = engineer(test)

# Surname frequency encoding (combinando train+test para que no haya leak de etiqueta)
all_surnames = pd.concat([train_fe['Surname'], test_fe['Surname']])
surname_freq = all_surnames.value_counts(normalize=True).to_dict()
train_fe['SurnameFreq'] = train_fe['Surname'].map(surname_freq).fillna(0)
test_fe['SurnameFreq']  = test_fe['Surname'].map(surname_freq).fillna(0)

# Drop identifiers
drop_cols = ['id', 'CustomerId', 'Surname', 'Exited']
y = train_fe['Exited'].values
test_ids = test_fe['id'].values

X = train_fe.drop(columns=drop_cols)
X_test = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns])

# One-hot
X = pd.get_dummies(X, columns=['Geography', 'Gender', 'Geo_Gender'], drop_first=False)
X_test = pd.get_dummies(X_test, columns=['Geography', 'Gender', 'Geo_Gender'], drop_first=False)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

print(f'Features: {X.shape[1]}')

# Class balance for scale_pos_weight
neg, pos = (y == 0).sum(), (y == 1).sum()
spw = neg / pos
print(f'scale_pos_weight = {spw:.3f}')

# --- K-fold OOF + test averaging for 3 models ---
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = {'xgb': np.zeros(len(X)), 'lgb': np.zeros(len(X)), 'cat': np.zeros(len(X))}
test_pred = {'xgb': np.zeros(len(X_test)), 'lgb': np.zeros(len(X_test)), 'cat': np.zeros(len(X_test))}

X_np = X.values.astype(np.float32)
Xt_np = X_test.values.astype(np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y)):
    print(f'\n=== Fold {fold + 1}/{N_FOLDS} ===')
    Xtr, Xva = X_np[tr_idx], X_np[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=1.2, gamma=0.1,
        scale_pos_weight=1.0,
        eval_metric='auc', random_state=SEED, n_jobs=-1,
        early_stopping_rounds=80, tree_method='hist',
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    oof['xgb'][va_idx] = xgb.predict_proba(Xva)[:, 1]
    test_pred['xgb'] += xgb.predict_proba(Xt_np)[:, 1] / N_FOLDS

    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=48, max_depth=-1,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=1.2,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
            callbacks=[__import__('lightgbm').early_stopping(80, verbose=False)])
    oof['lgb'][va_idx] = lgb.predict_proba(Xva)[:, 1]
    test_pred['lgb'] += lgb.predict_proba(Xt_np)[:, 1] / N_FOLDS

    # CatBoost
    cat = CatBoostClassifier(
        iterations=2000, learning_rate=0.04, depth=6,
        l2_leaf_reg=3.0, eval_metric='AUC',
        random_state=SEED, verbose=False, early_stopping_rounds=80,
    )
    cat.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True, verbose=False)
    oof['cat'][va_idx] = cat.predict_proba(Xva)[:, 1]
    test_pred['cat'] += cat.predict_proba(Xt_np)[:, 1] / N_FOLDS

    print(f'  XGB AUC: {roc_auc_score(yva, oof["xgb"][va_idx]):.5f}'
          f'  LGB AUC: {roc_auc_score(yva, oof["lgb"][va_idx]):.5f}'
          f'  CAT AUC: {roc_auc_score(yva, oof["cat"][va_idx]):.5f}')

print('\n=== OOF AUC ===')
auc_xgb = roc_auc_score(y, oof['xgb'])
auc_lgb = roc_auc_score(y, oof['lgb'])
auc_cat = roc_auc_score(y, oof['cat'])
print(f'XGB : {auc_xgb:.5f}')
print(f'LGB : {auc_lgb:.5f}')
print(f'CAT : {auc_cat:.5f}')

# --- Search best blend weights on OOF ---
best_auc, best_w = 0, None
for wx in np.arange(0, 1.01, 0.05):
    for wl in np.arange(0, 1.01 - wx, 0.05):
        wc = 1 - wx - wl
        if wc < 0:
            continue
        blend = wx * oof['xgb'] + wl * oof['lgb'] + wc * oof['cat']
        a = roc_auc_score(y, blend)
        if a > best_auc:
            best_auc, best_w = a, (wx, wl, wc)

wx, wl, wc = best_w
print(f'\nBest blend weights -> XGB={wx:.2f} LGB={wl:.2f} CAT={wc:.2f}')
print(f'Best blended OOF AUC: {best_auc:.5f}')

# Simple average baseline for comparison
mean_auc = roc_auc_score(y, (oof['xgb'] + oof['lgb'] + oof['cat']) / 3)
print(f'Mean blend OOF AUC : {mean_auc:.5f}')

# Final test prediction
final_test = wx * test_pred['xgb'] + wl * test_pred['lgb'] + wc * test_pred['cat']

submission = pd.DataFrame({'id': test_ids, 'Exited': final_test})
submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
print(f'\nsubmission.csv shape: {submission.shape}')
print(submission.head())

# Save OOF for the notebook
np.savez(os.path.join(DATA_DIR, '_oof_predictions.npz'),
         oof_xgb=oof['xgb'], oof_lgb=oof['lgb'], oof_cat=oof['cat'],
         test_xgb=test_pred['xgb'], test_lgb=test_pred['lgb'], test_cat=test_pred['cat'],
         y=y, test_ids=test_ids,
         weights=np.array([wx, wl, wc]))
print('Saved OOF predictions to _oof_predictions.npz')
