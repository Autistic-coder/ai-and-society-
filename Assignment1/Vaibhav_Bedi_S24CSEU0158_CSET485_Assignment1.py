# CSET485 Assignment 1 — Vaibhav Bedi, S24CSEU0158
# Run: python Vaibhav_Bedi_S24CSEU0158_CSET485_Assignment1.py


# %%
# # CSET485 Assignment 1 — Adult income model audit
# **Vaibhav Bedi · S24CSEU0158 · seed 158**
# 
# Run all cells from top to bottom. The notebook downloads the original UCI files if `data/` is absent and exports every result to `results/`. The supplied ZIP includes the source data for offline reproduction.
# 
# The outcome is `income > $50K`, not recidivism, danger, repayment, or merit. Models A and B use the same 70/30 stratified split. The separate descriptive subgroup analysis uses the full cleaned cohort as requested and does not train or tune the test-set predictor.
# 
# ## Environment
# Use Python 3.12 and install `requirements.txt`. This notebook uses only the standard Python scientific stack. The PDF explains the modeling assumptions and the limits of causal and fairness interpretations.
# 
# ## Methods chosen before reporting
# Exact duplicate removal; training-only category pooling (<100 observations); missing categories retained; log1p of final weight and capital gain/loss; standardized numeric predictors; one-hot categorical predictors; OLS and L2 logistic regression (C=1). The ordinal education-num representation retains the education information without a duplicate column. Subgroup confidence intervals use separate unpenalized explanatory logistic models with HC1 robust covariance; six slope comparisons receive Holm correction. The predictive model is never reselected using the test sweep.

# %%
# ## Imports and reproducibility

# %%
from pathlib import Path
import json, hashlib, urllib.request, platform, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SEED = 158
ROOT = Path.cwd()
DATA = ROOT / 'data'; DATA.mkdir(exist_ok=True)
OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})



# %%
# ## Obtain and clean the original UCI files

# %%
BASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
COLS = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
frames=[]; hashes={}
for filename in ['adult.data','adult.test']:
    path=DATA/filename
    if not path.exists(): path.write_bytes(urllib.request.urlopen(BASE+filename,timeout=60).read())
    hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    frame=pd.read_csv(path,names=COLS,skipinitialspace=True,na_values='?',comment='|')
    frame['income']=frame['income'].str.strip().str.rstrip('.')
    frame['source_file']=filename
    frame['source_row']=np.arange(1,len(frame)+1)
    frames.append(frame)
raw=pd.concat(frames,ignore_index=True)
raw['record_id']=np.arange(1,len(raw)+1)
missing=raw[COLS].isna().sum().to_dict()
duplicate_count=int(raw.duplicated(COLS).sum())
# Exact duplicates are removed before splitting; source identifiers are never features.
df=raw.drop_duplicates(COLS).copy().set_index('record_id')
y=(df['income']=='>50K').astype(int)
train_ids,test_ids=train_test_split(df.index,test_size=.30,random_state=SEED,stratify=y)
train=df.loc[train_ids].copy(); test=df.loc[test_ids].copy()
ytr=y.loc[train_ids]; yte=y.loc[test_ids]
assert set(train_ids).isdisjoint(test_ids)
# education is a deterministic duplicate of education-num; retain numeric representation.
assert df.groupby('education')['education-num'].nunique().max()==1
num_cols=['age','education-num','hours-per-week','fnlwgt','capital-gain','capital-loss']
cat_cols=['workclass','marital-status','occupation','relationship','race','sex','native-country']



# %%
# ## Preprocessing fitted on training data only

# %%
medians=train[num_cols].median()
def numeric(frame):
    v=frame[num_cols].fillna(medians).astype(float).copy()
    for c in ['fnlwgt','capital-gain','capital-loss']:v[c]=np.log1p(v[c])
    return v
scaler=StandardScaler().fit(numeric(train))
retained={c:set(train[c].fillna('Missing').value_counts().loc[lambda s:s>=100].index) for c in cat_cols}
def categorical(frame):
    v=frame[cat_cols].fillna('Missing').copy()
    for c in cat_cols:v[c]=v[c].where(v[c].isin(retained[c]),'OtherRare')
    return v
enc=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False).fit(categorical(train))
names=['z_'+c for c in num_cols]+list(enc.get_feature_names_out(cat_cols))
def design(frame):
    a=np.column_stack([scaler.transform(numeric(frame)),enc.transform(categorical(frame))])
    return sm.add_constant(pd.DataFrame(a,index=frame.index,columns=names),has_constant='add')
Xtr=design(train);Xte=design(test)
assert np.linalg.matrix_rank(Xtr.to_numpy())==Xtr.shape[1]
# Outliers retained. log1p reduces numerical skew, then z scores use training means/SDs.
# fnlwgt is used as a predictor, not a sample weight. These are sample-level results.
preprocessing={'raw_n':len(raw),'deduplicated_n':len(df),'duplicates_removed':duplicate_count,'train_n':len(train),'test_n':len(test),'missing_raw':missing,'positive_train':int(ytr.sum()),'positive_test':int(yte.sum()),'predictor_columns':len(names),'hashes':hashes,'category_reference':dict(zip(cat_cols,[str(x[0]) for x in enc.categories_])),'retained_categories':{k:sorted(v) for k,v in retained.items()},'numeric_means':dict(zip(num_cols,scaler.mean_)),'numeric_sd':dict(zip(num_cols,scaler.scale_))}
print('PREPROCESSING',preprocessing)


# %%
# ## Part 1 Model fitting

# %%
ols=sm.OLS(ytr,Xtr).fit()
# L2 logistic stabilizes sparse/near-collinear categorical levels. Intercept is unpenalized.
clf=LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000,tol=1e-9,random_state=SEED)
clf.fit(Xtr.drop(columns='const'),ytr)
assert clf.n_iter_[0]<clf.max_iter
beta=pd.Series(np.r_[clf.intercept_,clf.coef_[0]],index=Xtr.columns)
pa=ols.predict(Xte);pb=pd.Series(clf.predict_proba(Xte.drop(columns='const'))[:,1],index=test.index)
median=float(np.median(pa));ca=(pa>=median).astype(int);cb=(pb>=.5).astype(int)
agreement=confusion_matrix(ca,cb,labels=[0,1]);mismatch=int((ca!=cb).sum())
part1={'median_ols':median,'agreement':agreement.tolist(),'mismatch_n':mismatch,'mismatch_pct':100*mismatch/len(test),'ols_positive_n':int(ca.sum()),'logit_positive_n':int(cb.sum()),'ols_accuracy':accuracy_score(yte,ca),'logit_accuracy':accuracy_score(yte,cb),'logit_auc':roc_auc_score(yte,pb),'logit_brier':brier_score_loss(yte,pb),'ols_outside_0_1_n':int(((pa<0)|(pa>1)).sum()),'ols_min':float(pa.min()),'ols_max':float(pa.max()),'ols_confusion_truth':confusion_matrix(yte,ca).tolist(),'logit_confusion_truth':confusion_matrix(yte,cb).tolist(),'logit_converged':bool(clf.n_iter_[0]<clf.max_iter)}
print('PART1',part1)
coefs=pd.DataFrame({'coefficient':beta});coefs.to_csv(OUT/'baseline_coefficients.csv')
pred=test.copy();pred['actual']=yte;pred['ols_score']=pa;pred['logit_probability']=pb;pred['ols_class']=ca;pred['logit_class']=cb
pred.to_csv(OUT/'test_predictions.csv')


# %%
# ## Part 2.1 Exploratory subgroup analysis on full cleaned cohort

# %%
# Explanatory fits never change the predictive model or its held-out predictions.
Z=pd.DataFrame({'age':df.age,'education_num':df['education-num'],'hours_per_week':df['hours-per-week'],'log_capital_gain':np.log1p(df['capital-gain']),'log_capital_loss':np.log1p(df['capital-loss']),'married':(df['marital-status']=='Married-civ-spouse').astype(int),'male':(df.sex=='Male').astype(int)},index=df.index,dtype=float)
pooled=sm.GLM(y,sm.add_constant(Z),family=sm.families.Binomial()).fit(cov_type='HC1')
subfits={}
for sex in ['Female','Male']:
    mask=df.sex.eq(sex)
    subfits[sex]=sm.GLM(y.loc[mask],sm.add_constant(Z.loc[mask].drop(columns='male')),family=sm.families.Binomial()).fit(cov_type='HC1')
rows=[]
for key in Z.columns.drop('male'):
    a,b=subfits['Female'],subfits['Male'];delta=float(a.params[key]-b.params[key]);se=float(np.sqrt(a.bse[key]**2+b.bse[key]**2));pv=2*norm.sf(abs(delta/se))
    row={'predictor':key,'pooled':pooled.params[key],'female':a.params[key],'male':b.params[key],'difference_F_minus_M':delta,'difference_ci_low':delta-1.96*se,'difference_ci_high':delta+1.96*se,'p_difference':pv}
    for label,fit in [('pooled',pooled),('female',a),('male',b)]:row[label+'_ci_low'],row[label+'_ci_high']=fit.conf_int().loc[key]
    rows.append(row)
subgroup=pd.DataFrame(rows);subgroup['p_holm']=multipletests(subgroup.p_difference,method='holm')[1]
subgroup.to_csv(OUT/'subgroup_effects.csv',index=False);print('SUBGROUP\n',subgroup.to_string(index=False))


# %%
# ## Part 2.2 Omitted variable sensitivity

# %%
Xred=Xtr.drop(columns='z_education-num')
reduced=LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000,tol=1e-9,random_state=SEED).fit(Xred.drop(columns='const'),ytr)
assert reduced.n_iter_[0]<reduced.max_iter
reduced_beta=pd.Series(np.r_[reduced.intercept_,reduced.coef_[0]],index=Xred.columns)
shared=beta.index.intersection(reduced_beta.index).drop('const')
omit=pd.DataFrame({'full':beta[shared],'reduced':reduced_beta[shared]})
omit['delta']=omit.reduced-omit.full;omit['pct_change_signed']=100*omit.delta/omit.full
omit['absolute_pct_change']=omit.pct_change_signed.abs();omit['absolute_shift']=omit.delta.abs();omit.to_csv(OUT/'omitted_variable_all_coefficients.csv')
key_predictors=['z_age','z_hours-per-week','z_capital-gain','z_capital-loss']
omit_key=omit.loc[key_predictors];omit_key.to_csv(OUT/'omitted_variable_four_predictors.csv')
print('OMITTED FOUR\n',omit_key.to_string());print('LARGEST ABS\n',omit.sort_values('absolute_shift',ascending=False).head(3).to_string());print('LARGEST PCT\n',omit.sort_values('absolute_pct_change',ascending=False).head(3).to_string())


# %%
# ## Part 2.3 High confidence errors and seeded sample

# %%
wrong=((pb>=.75)&(yte==0))|((pb<=.25)&(yte==1))
wrong_all=pred.loc[wrong].copy();adv=wrong_all.sample(n=min(50,len(wrong_all)),random_state=SEED).copy()
wrong_all.to_csv(OUT/'all_high_confidence_errors.csv');adv.to_csv(OUT/'adversarial_50.csv')
def describe_group(frame):
    return {'n':len(frame),'age_mean':float(frame.age.mean()),'age_median':float(frame.age.median()),'age_min':int(frame.age.min()),'age_max':int(frame.age.max()),'female_pct':100*frame.sex.eq('Female').mean(),'male_pct':100*frame.sex.eq('Male').mean(),'white_pct':100*frame.race.eq('White').mean(),'married_pct':100*frame['marital-status'].eq('Married-civ-spouse').mean(),'zero_gain_pct':100*frame['capital-gain'].eq(0).mean(),'zero_loss_pct':100*frame['capital-loss'].eq(0).mean(),'education_ge13_pct':100*frame['education-num'].ge(13).mean(),'hours_ge40_pct':100*frame['hours-per-week'].ge(40).mean(),'positive_actual_pct':100*frame.actual.mean()}
demo=pd.DataFrame({'test':describe_group(pred),'all_errors':describe_group(wrong_all),'sample50':describe_group(adv)});demo.to_csv(OUT/'adversarial_demographics.csv')
contrib=Xte.loc[adv.index].mul(beta,axis=1);contrib.to_csv(OUT/'adversarial_logodds_contributions.csv')
assert np.allclose(contrib.sum(axis=1),np.log(adv.logit_probability/(1-adv.logit_probability)))
adv_stats={'total_n':int(wrong.sum()),'pct_test':100*wrong.mean(),'confident_fp':int(((pb>=.75)&(yte==0)).sum()),'confident_fn':int(((pb<=.25)&(yte==1)).sum()),'sample_fp':int(((adv.logit_probability>=.75)&(adv.actual==0)).sum()),'sample_fn':int(((adv.logit_probability<=.25)&(adv.actual==1)).sum()),'sample_n':len(adv)}
print('ADVERSARIAL',adv_stats);print(demo.to_string())


# %%
# ## Part 2.4 Fairness sweep A Female B Male

# %%
# Income prediction is not a risk or creditworthiness label.
def div(a,b):return float(a/b) if np.isfinite(b) and b!=0 else np.nan
def rates(truth,proba,t):
    labels=(proba>=t).astype(int);tn,fp,fn,tp=confusion_matrix(truth,labels,labels=[0,1]).ravel()
    return {'n':len(truth),'positive_n':int(labels.sum()),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),'selection_rate':float(labels.mean()),'tpr':div(tp,tp+fn),'fpr':div(fp,fp+tn),'ppv':div(tp,tp+fp),'accuracy':float((labels==truth).mean()),'base_rate':float(truth.mean())}
thresholds=np.round(np.arange(.05,1,.1),2);fair=[]
for t in thresholds:
    female=test.sex.eq('Female');male=test.sex.eq('Male');a=rates(yte[female],pb[female],t);b=rates(yte[male],pb[male],t)
    tr,fr=div(a['tpr'],b['tpr']),div(a['fpr'],b['fpr'])
    row={'threshold':float(t),'DP':div(a['selection_rate'],b['selection_rate']),'EO':min(tr,fr) if np.isfinite([tr,fr]).all() else np.nan,'PP':div(a['ppv'],b['ppv']),'TPR_ratio':tr,'FPR_ratio':fr,'accuracy':accuracy_score(yte,pb>=t)}
    row.update({'F_'+k:v for k,v in a.items()});row.update({'M_'+k:v for k,v in b.items()});fair.append(row)
fair=pd.DataFrame(fair);fair.to_csv(OUT/'fairness_thresholds.csv',index=False)
print('FAIRNESS\n',fair[['threshold','DP','EO','PP','TPR_ratio','FPR_ratio','accuracy','F_tpr','M_tpr','F_fpr','M_fpr','F_ppv','M_ppv']].to_string(index=False))
fig,ax=plt.subplots(figsize=(7,5));ax.plot(fair.DP,fair.EO,'o-',color='#28647d',lw=1.5)
for _,r in fair.iterrows():ax.annotate(f"{r.threshold:.2f}",(r.DP,r.EO),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.axvline(1,color='#777',ls='--',lw=.8);ax.axhline(1,color='#777',ls='--',lw=.8)
ax.set(xlim=(0,1.5),ylim=(0,1.5),xlabel='Demographic parity ratio  Female / Male',ylabel='Equalized odds index specified in assignment',title='Threshold sweep on the held-out Adult test set');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(OUT/'fairness_tradeoff.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for col in ['DP','EO','PP']:ax.plot(fair.threshold,fair[col],marker='o',label=col)
ax.axhline(1,color='#777',ls='--',lw=.8);ax.set(xlabel='Decision threshold',ylabel='Female / Male ratio',title='Three fairness measures across thresholds');ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(OUT/'fairness_by_threshold.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,3));ax.imshow(agreement,cmap='Blues')
for (i,j),v in np.ndenumerate(agreement):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>agreement.max()/2 else 'black',fontsize=14)
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Logistic 0','Logistic 1'],yticklabels=['OLS 0','OLS 1'],title='Agreement between the two decision rules');fig.tight_layout();fig.savefig(OUT/'agreement_matrix.png',dpi=200);plt.close(fig)


# %%
# ## Export evidence and verify consistency

# %%
versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'statsmodels':sm.__version__,'matplotlib':matplotlib.__version__}
results={'seed':SEED,'preprocessing':preprocessing,'part1':part1,'adversarial':adv_stats,'versions':versions}
(OUT/'summary.json').write_text(json.dumps(results,indent=2));(OUT/'train_record_ids.json').write_text(json.dumps([int(i) for i in train_ids]));(OUT/'test_record_ids.json').write_text(json.dumps([int(i) for i in test_ids]))
assert agreement.sum()==len(test)
assert mismatch==agreement[0,1]+agreement[1,0]
assert len(adv)==min(50,int(wrong.sum()))
assert set(adv.index).issubset(set(test.index))
assert len(fair)==10 and np.isfinite(beta).all()
print('Completed: all consistency checks passed')


# %%
# ## Additional report evidence and case examples

# %%
hi=fair.set_index('threshold').loc[.85];lo=fair.set_index('threshold').loc[.75]
z_gain=100*(lo.DP-hi.DP)/hi.DP;w_loss=100*(hi.EO-lo.EO)/hi.EO
case_ids=[534,40829]
cases=pred.loc[case_ids,['age','sex','education-num','hours-per-week','actual','logit_probability']].copy()
assert (cases.logit_probability.ge(.75)&cases.logit_probability.lt(.85)).all()
cases['class_at_085']=(cases.logit_probability>=.85).astype(int);cases['class_at_075']=(cases.logit_probability>=.75).astype(int)
cases.to_csv(OUT/'fairness_case_examples.csv')
row1532=adv.loc[1532];c1532=contrib.loc[1532]
evidence={'tradeoff':{'from_threshold':.85,'to_threshold':.75,'dp_gain_pct':float(z_gain),'eo_loss_pct':float(w_loss),'cost_W_over_Z':float(w_loss/z_gain)},'subgroup_n':df.sex.value_counts().to_dict(),'education_odds_ratios':{k:float(np.exp(v.params['education_num'])) for k,v in subfits.items()},'case1532':{'probability':float(row1532.logit_probability),'log_odds':float(c1532.sum()),'education_contribution':float(c1532['z_education-num']),'occupation_contribution':float(c1532['occupation_Machine-op-inspct'])},'missing_any_raw':int(raw[COLS].isna().any(axis=1).sum()),'ols_selection_pct':float(100*ca.mean()),'logistic_selection_pct':float(100*cb.mean()),'zero_classifier_accuracy':float(1-yte.mean())}
(OUT/'report_evidence.json').write_text(json.dumps(evidence,indent=2))
print('ADDITIONAL REPORT EVIDENCE',evidence)
print(cases.to_string())
# Recreate the required 0-1.5 chart with separated labels and leader lines.
fig,ax=plt.subplots(figsize=(7.2,5.3));ax.plot(fair.DP,fair.EO,'o-',color='#245b73',lw=1.4)
label_order=[.05,.85,.15,.75,.25,.35,.45,.65,.55,.95]
for i,t in enumerate(label_order):
    r=fair.set_index('threshold').loc[t]
    ax.annotate(f't = {t:.2f}',(r.DP,r.EO),xytext=(.72,1.36-i*.13),textcoords='data',fontsize=9,arrowprops={'arrowstyle':'-','lw':.65,'color':'#888'})
ax.axvline(1,color='#aaa',ls='--',lw=.8);ax.axhline(1,color='#aaa',ls='--',lw=.8)
ax.set(xlim=(0,1.5),ylim=(0,1.5),xlabel='Demographic parity ratio  Female / Male',ylabel='Equalized odds index specified in assignment',title='Fairness trade-off across ten thresholds');ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(OUT/'fairness_tradeoff.png',dpi=200);plt.close(fig)
print('Final evidence and charts saved')


# %%
# ## Inspect the exported evidence
# All displayed tables are computed above. The requested equalized-odds index is the smaller of two ratios; inspect both component ratios as well. Undefined denominators return NaN, never an invented zero. The two fee-assessment examples in the report are hypothetical uses of real test rows, not observed decisions.

# %% Display results in a standard terminal
for table in [subgroup.round(6), omit_key.round(6), demo.round(3), fair.round(6), cases]:
    print(table.to_string())
    print()
print(f'Charts and result files saved to: {OUT.resolve()}')
