"""
Bank Churn v3 - multi-seed + stacking + more features.
Target: beat 0.9383 on Kaggle leaderboard.
"""
import os, warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stop
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
DATA_DIR = '/home/user/Churn'
SEEDS = [42, 7, 2024]
N_FOLDS = 5

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Basic clean
num_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
cat_cols = ['Geography','Gender','Surname']
for df in (train, test):
    for c in num_cols:
        if df[c].isna().any(): df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any(): df[c] = df[c].fillna(df[c].mode().iloc[0])

train = train.dropna(subset=['Exited']).reset_index(drop=True)
train['Exited'] = train['Exited'].astype(int)

# ----- Feature engineering (más fuerte) -----
def engineer(df):
    df = df.copy()
    df['IsZeroBalance']      = (df['Balance'] == 0).astype(int)
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScoreAge']     = df['CreditScore'] / (df['Age'] + 1)
    df['TenureByAge']        = df['Tenure'] / (df['Age'] + 1)
    df['ProductsPerTenure']  = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['BalancePerProduct']  = df['Balance'] / (df['NumOfProducts'] + 1)
    df['SalaryPerProduct']   = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
    df['AgeBalance']         = df['Age'] * df['Balance']
    df['AgeCreditScore']     = df['Age'] * df['CreditScore']
    df['AgeProducts']        = df['Age'] * df['NumOfProducts']
    df['BalanceActive']      = df['Balance'] * df['IsActiveMember']
    df['CreditActive']       = df['CreditScore'] * df['IsActiveMember']
    df['ProductsActive']     = df['NumOfProducts'] * df['IsActiveMember']
    df['AgeGroup']           = pd.cut(df['Age'], bins=[0,30,40,50,60,100], labels=[0,1,2,3,4]).astype(int)
    df['CreditGroup']        = pd.cut(df['CreditScore'], bins=[0,580,670,740,800,1000], labels=[0,1,2,3,4]).astype(int)
    df['BalanceGroup']       = pd.cut(df['Balance'], bins=[-1,1,50000,100000,150000,1e9], labels=[0,1,2,3,4]).astype(int)
    df['HighBalance']        = (df['Balance'] > 100000).astype(int)
    df['IsSenior']           = (df['Age'] >= 60).astype(int)
    df['IsYoung']            = (df['Age'] <= 30).astype(int)
    df['Inactive_NoCard']    = ((df['IsActiveMember']==0) & (df['HasCrCard']==0)).astype(int)
    df['Active_HasCard']     = ((df['IsActiveMember']==1) & (df['HasCrCard']==1)).astype(int)
    df['Geo_Gender']         = df['Geography'].astype(str) + '_' + df['Gender'].astype(str)
    df['Geo_Age']            = df['Geography'].astype(str) + '_' + df['AgeGroup'].astype(str)
    df['LowTenure_HighBalance'] = ((df['Tenure']<=2) & (df['Balance']>100000)).astype(int)
    df['ManyProducts']       = (df['NumOfProducts'] >= 3).astype(int)
    df['SingleProduct']      = (df['NumOfProducts'] == 1).astype(int)
    df['SurnameLen']         = df['Surname'].astype(str).str.len()
    return df

train_fe = engineer(train)
test_fe  = engineer(test)

# Surname frequency encoding (sin target)
all_surnames = pd.concat([train_fe['Surname'], test_fe['Surname']])
freq_map = all_surnames.value_counts(normalize=True).to_dict()
train_fe['SurnameFreq'] = train_fe['Surname'].map(freq_map).fillna(0)
test_fe['SurnameFreq']  = test_fe['Surname'].map(freq_map).fillna(0)

drop_cols = ['id','CustomerId','Surname']
y        = train_fe['Exited'].values
test_ids = test_fe['id'].values
X      = train_fe.drop(columns=drop_cols + ['Exited'])
X_test = test_fe.drop(columns=drop_cols)

# Target encoding K-fold para Surname (limpio, sin leakage)
def kfold_target_encode(train_col, test_col, y, n_splits=5, smooth=20):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    enc_train = np.zeros(len(train_col))
    global_mean = y.mean()
    for tr, va in skf.split(train_col, y):
        df_tr = pd.DataFrame({'c': train_col.iloc[tr].values, 'y': y[tr]})
        stats = df_tr.groupby('c')['y'].agg(['mean','count'])
        stats['enc'] = (stats['mean']*stats['count'] + global_mean*smooth) / (stats['count'] + smooth)
        m = stats['enc'].to_dict()
        enc_train[va] = pd.Series(train_col.iloc[va].values).map(m).fillna(global_mean).values
    # Para test: usar TODO train
    df_all = pd.DataFrame({'c': train_col.values, 'y': y})
    stats = df_all.groupby('c')['y'].agg(['mean','count'])
    stats['enc'] = (stats['mean']*stats['count'] + global_mean*smooth) / (stats['count'] + smooth)
    m = stats['enc'].to_dict()
    enc_test = pd.Series(test_col.values).map(m).fillna(global_mean).values
    return enc_train, enc_test

for c in ['Surname','Geography','Gender','Geo_Gender','Geo_Age']:
    tr_te, te_te = kfold_target_encode(train_fe[c], test_fe[c], y)
    X[f'{c}_te'] = tr_te
    X_test[f'{c}_te'] = te_te

# One-hot
X      = pd.get_dummies(X,      columns=['Geography','Gender','Geo_Gender','Geo_Age'], drop_first=False)
X_test = pd.get_dummies(X_test, columns=['Geography','Gender','Geo_Gender','Geo_Age'], drop_first=False)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
print(f'Features: {X.shape[1]}')

X_np  = X.values.astype(np.float32)
Xt_np = X_test.values.astype(np.float32)

# ----- Multi-seed K-fold para 4 modelos -----
def get_models(seed):
    return {
        'xgb': XGBClassifier(n_estimators=3000, learning_rate=0.025, max_depth=5,
                              min_child_weight=4, subsample=0.85, colsample_bytree=0.8,
                              reg_alpha=0.3, reg_lambda=1.5, gamma=0.1,
                              eval_metric='auc', random_state=seed, n_jobs=-1,
                              early_stopping_rounds=100, tree_method='hist'),
        'lgb': LGBMClassifier(n_estimators=3000, learning_rate=0.025, num_leaves=64,
                               min_child_samples=20, subsample=0.85, colsample_bytree=0.8,
                               reg_alpha=0.3, reg_lambda=1.5,
                               random_state=seed, n_jobs=-1, verbose=-1),
        'cat': CatBoostClassifier(iterations=3000, learning_rate=0.035, depth=6,
                                   l2_leaf_reg=3.0, eval_metric='AUC',
                                   random_state=seed, verbose=False,
                                   early_stopping_rounds=100),
        'hgb': HistGradientBoostingClassifier(max_iter=1500, learning_rate=0.04,
                                               max_depth=7, min_samples_leaf=20,
                                               l2_regularization=0.5,
                                               random_state=seed,
                                               early_stopping=True, validation_fraction=0.15,
                                               n_iter_no_change=50),
    }

model_names = ['xgb','lgb','cat','hgb']
oof  = {m: np.zeros(len(X)) for m in model_names}
test_pred = {m: np.zeros(len(X_test)) for m in model_names}

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    print(f'\n##### SEED {seed} #####')
    for fold, (tr, va) in enumerate(skf.split(X_np, y)):
        Xtr, Xva, ytr, yva = X_np[tr], X_np[va], y[tr], y[va]
        models = get_models(seed)
        fold_aucs = {}

        m = models['xgb']
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof['xgb'][va] += m.predict_proba(Xva)[:,1] / len(SEEDS)
        test_pred['xgb'] += m.predict_proba(Xt_np)[:,1] / (len(SEEDS)*N_FOLDS)
        fold_aucs['xgb'] = roc_auc_score(yva, m.predict_proba(Xva)[:,1])

        m = models['lgb']
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
              callbacks=[lgb_early_stop(100, verbose=False)])
        oof['lgb'][va] += m.predict_proba(Xva)[:,1] / len(SEEDS)
        test_pred['lgb'] += m.predict_proba(Xt_np)[:,1] / (len(SEEDS)*N_FOLDS)
        fold_aucs['lgb'] = roc_auc_score(yva, m.predict_proba(Xva)[:,1])

        m = models['cat']
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True, verbose=False)
        oof['cat'][va] += m.predict_proba(Xva)[:,1] / len(SEEDS)
        test_pred['cat'] += m.predict_proba(Xt_np)[:,1] / (len(SEEDS)*N_FOLDS)
        fold_aucs['cat'] = roc_auc_score(yva, m.predict_proba(Xva)[:,1])

        m = models['hgb']
        m.fit(Xtr, ytr)
        oof['hgb'][va] += m.predict_proba(Xva)[:,1] / len(SEEDS)
        test_pred['hgb'] += m.predict_proba(Xt_np)[:,1] / (len(SEEDS)*N_FOLDS)
        fold_aucs['hgb'] = roc_auc_score(yva, m.predict_proba(Xva)[:,1])

        print(f'  Fold {fold+1}: ' + '  '.join(f'{k}={v:.5f}' for k,v in fold_aucs.items()))

print('\n=== OOF AUC por modelo (promediado entre seeds) ===')
for k in model_names:
    print(f'  {k}: {roc_auc_score(y, oof[k]):.5f}')

# ----- Búsqueda de blend ponderado -----
from itertools import product
best_auc, best_w = 0, None
step = 0.05
for wx in np.arange(0,1.001,step):
    for wl in np.arange(0,1.001-wx,step):
        for wc in np.arange(0,1.001-wx-wl,step):
            wh = 1 - wx - wl - wc
            if wh < -1e-9: continue
            wh = max(wh, 0)
            blend = wx*oof['xgb'] + wl*oof['lgb'] + wc*oof['cat'] + wh*oof['hgb']
            a = roc_auc_score(y, blend)
            if a > best_auc:
                best_auc, best_w = a, (wx, wl, wc, wh)
wx, wl, wc, wh = best_w
print(f'\nPesos blend -> XGB={wx:.2f} LGB={wl:.2f} CAT={wc:.2f} HGB={wh:.2f}')
print(f'AUC blend OOF: {best_auc:.5f}')

# ----- Stacking con LogisticRegression -----
oof_stack = np.column_stack([oof[k] for k in model_names])
test_stack = np.column_stack([test_pred[k] for k in model_names])

# Generar OOF del meta-modelo correctamente
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
meta_oof = np.zeros(len(X))
test_meta = np.zeros(len(X_test))
scaler = StandardScaler()
oof_stack_sc = scaler.fit_transform(oof_stack)
test_stack_sc = scaler.transform(test_stack)
for tr, va in skf.split(oof_stack_sc, y):
    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(oof_stack_sc[tr], y[tr])
    meta_oof[va] = meta.predict_proba(oof_stack_sc[va])[:,1]
    test_meta += meta.predict_proba(test_stack_sc)[:,1] / N_FOLDS
print(f'AUC stacking (LogReg meta): {roc_auc_score(y, meta_oof):.5f}')

# ----- Blend final: combinación de blend ponderado + stacking -----
blend_oof = wx*oof['xgb'] + wl*oof['lgb'] + wc*oof['cat'] + wh*oof['hgb']
blend_test = wx*test_pred['xgb'] + wl*test_pred['lgb'] + wc*test_pred['cat'] + wh*test_pred['hgb']

best_auc2, best_alpha = 0, 0
for a in np.arange(0, 1.001, 0.05):
    pred = a*blend_oof + (1-a)*meta_oof
    auc = roc_auc_score(y, pred)
    if auc > best_auc2:
        best_auc2, best_alpha = auc, a
print(f'\nMezcla final -> alpha(blend)={best_alpha:.2f} (1-alpha)(stack)={1-best_alpha:.2f}')
print(f'AUC final OOF: {best_auc2:.5f}')

final_test = best_alpha*blend_test + (1-best_alpha)*test_meta

sub = pd.DataFrame({'id': test_ids, 'Exited': final_test})
sub.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
print(f'\nsubmission.csv shape: {sub.shape}')
print(sub.head())
