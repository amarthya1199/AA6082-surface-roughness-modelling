"""Re-analysis of the AA6082-T651 Box-Behnken end-milling data (run 7 at 1500 rpm).

Fits the full and reduced quadratic RSM models (Type III ANOVA, lack of fit, PRESS),
evaluates RSM and 3-h-1 ANN models by leave-one-out cross-validation, runs nested LOOCV
for the ANN settings, and locates the minimum-roughness setting with its prediction interval.
"""
import json, sys, time
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import least_squares, minimize

# usage: python src/analysis.py data/aa6082_bbd_surface_roughness.csv results
DATA, OUT = sys.argv[1], sys.argv[2]
df = pd.read_csv(DATA)[["run","N","f","d","Ra"]]
df["x1"]=(df.N-3000)/1500; df["x2"]=(df.f-350)/150; df["x3"]=(df.d-0.4)/0.2
y = df.Ra.values; SST = ((y-y.mean())**2).sum()
res = {}

FULL = "Ra ~ x1+x2+x3+I(x1**2)+I(x2**2)+I(x3**2)+x1:x2+x1:x3+x2:x3"
RED  = "Ra ~ x1+x2+x3+I(x1**2)+I(x3**2)+x1:x3"
LIN  = "Ra ~ x1+x2+x3"

def anova(model):
    a = sm.stats.anova_lm(model, typ=3)
    return a
full = smf.ols(FULL, df).fit(); red = smf.ols(RED, df).fit()
a = anova(full)
# pure error from centre replicates
cen = df[(df.x1==0)&(df.x2==0)&(df.x3==0)].Ra.values
ss_pe = ((cen-cen.mean())**2).sum(); dof_pe = len(cen)-1
ss_e = full.ssr; dof_e = int(full.df_resid)
ss_lof = ss_e-ss_pe; dof_lof = dof_e-dof_pe
from scipy.stats import f as fdist
F_lof = (ss_lof/dof_lof)/(ss_pe/dof_pe); p_lof = 1-fdist.cdf(F_lof,dof_lof,dof_pe)
press = lambda m: (m.resid/(1-m.get_influence().hat_matrix_diag)).pow(2).sum()
res["rsm_full"] = dict(params=full.params.round(5).to_dict(), bse=full.bse.round(5).to_dict(),
    R2=full.rsquared, R2adj=full.rsquared_adj, R2pred=1-press(full)/SST,
    SSE=ss_e, MSE=ss_e/dof_e, SS_pe=ss_pe, MS_pe=ss_pe/dof_pe, SS_lof=ss_lof, F_lof=F_lof, p_lof=p_lof,
    SST=SST, model_F=full.fvalue, model_p=full.f_pvalue,
    anova=a.round(5).to_dict(orient="index"))
res["rsm_red"] = dict(params=red.params.round(5).to_dict(), pvalues=red.pvalues.round(5).to_dict(),
    R2=red.rsquared, R2adj=red.rsquared_adj, R2pred=1-press(red)/SST, MSE=red.mse_resid, F=red.fvalue, p=red.f_pvalue)
# actual-unit equation for reduced model
act = smf.ols("Ra ~ N+f+d+I(N**2)+I(d**2)+N:d", df).fit()
res["rsm_red_actual"] = {k: float(v) for k,v in act.params.items()}
actf = smf.ols("Ra ~ N+f+d+I(N**2)+I(f**2)+I(d**2)+N:f+N:d+f:d", df).fit()
res["rsm_full_actual"] = {k: float(v) for k,v in actf.params.items()}

# feed per revolution check: f/N in mm/rev
df["fr"] = df.f/df.N
m_fr = smf.ols("Ra ~ fr + x3 + I(x3**2)", df).fit()
m_fr2 = smf.ols("Ra ~ fr + I(fr**2) + x3 + I(x3**2) + fr:x3", df).fit()
res["feed_per_rev"] = dict(range=[df.fr.min(), df.fr.max()], R2_lin=m_fr.rsquared, R2adj_lin=m_fr.rsquared_adj,
    R2pred_lin=1-press(m_fr)/SST, params=m_fr.params.to_dict(), pvalues=m_fr.pvalues.to_dict(),
    R2_quad=m_fr2.rsquared, R2adj_quad=m_fr2.rsquared_adj, R2pred_quad=1-press(m_fr2)/SST,
    corr_Ra_fr=float(np.corrcoef(df.fr, df.Ra)[0,1]), corr_Ra_N=float(np.corrcoef(df.N, df.Ra)[0,1]))
# does N carry information beyond f/N?
m_frN = smf.ols("Ra ~ fr + x1 + x3 + I(x3**2)", df).fit()
res["feed_per_rev"]["with_N"] = dict(R2=m_frN.rsquared, p_x1=float(m_frN.pvalues["x1"]), coef_x1=float(m_frN.params["x1"]))

# optimum (min Ra) within coded cube, both models; prediction interval
def pred(model, x):
    d0 = pd.DataFrame(dict(x1=[x[0]],x2=[x[1]],x3=[x[2]]))
    return model.get_prediction(d0).summary_frame(alpha=0.05).iloc[0]
opt = {}
for name, m in [("full",full),("red",red)]:
    best=None
    for s in np.random.default_rng(0).uniform(-1,1,(40,3)):
        r = minimize(lambda x: float(m.predict(pd.DataFrame(dict(x1=[x[0]],x2=[x[1]],x3=[x[2]]))).iloc[0]), s, bounds=[(-1,1)]*3)
        if best is None or r.fun<best.fun: best=r
    x=best.x; p=pred(m,x)
    opt[name]=dict(coded=x.round(3).tolist(), N=3000+1500*x[0], f=350+150*x[1], d=0.4+0.2*x[2],
        Ra=float(p["mean"]), ci=[float(p.mean_ci_lower),float(p.mean_ci_upper)], pi=[float(p.obs_ci_lower),float(p.obs_ci_upper)],
        se_mean=float(p.mean_se))
    # compare with best measured run and with a tested point
for name,m in [("full",full),("red",red)]:
    p=pred(m,[1,-1,0]); opt[name]["at_run2"]=dict(Ra=float(p["mean"]),pi=[float(p.obs_ci_lower),float(p.obs_ci_upper)],se_mean=float(p.mean_se))
    p=pred(m,[0,0,0]); opt[name]["se_centre"]=float(p.mean_se)
    p=pred(m,[1,-1,-1]); opt[name]["corner_se"]=float(p.mean_se)
res["optimum"]=opt

# ---------- LOOCV ----------
X = df[["x1","x2","x3"]].values
def metrics(yt, yp):
    e = yp-yt
    return dict(RMSE=float(np.sqrt((e**2).mean())), MAE=float(np.abs(e).mean()),
                MAPE=float((np.abs(e)/yt).mean()*100), maxAE=float(np.abs(e).max()),
                Q2=float(1-(e**2).sum()/SST), r=float(np.corrcoef(yt,yp)[0,1]))
def loo_ols(formula):
    p=np.zeros(15)
    for i in range(15):
        m=smf.ols(formula, df.drop(i)).fit(); p[i]=m.predict(df.iloc[[i]]).iloc[0]
    return p
loo = {}
preds = {}
for name,fm in [("linear",LIN),("rsm_full",FULL),("rsm_red",RED)]:
    p=loo_ols(fm); preds[name]=p; loo[name]=metrics(y,p)
insample = {"rsm_full":metrics(y,full.fittedvalues.values), "rsm_red":metrics(y,red.fittedvalues.values),
            "linear":metrics(y,smf.ols(LIN,df).fit().fittedvalues.values)}

# ANN 3-h-1: tanh hidden, linear output, weight decay lambda, Levenberg-Marquardt (MINPACK)
def unpack(w,h):
    W1=w[:3*h].reshape(h,3); b1=w[3*h:4*h]; W2=w[4*h:5*h]; b2=w[5*h]
    return W1,b1,W2,b2
def fwd(w,h,X):
    W1,b1,W2,b2=unpack(w,h); return np.tanh(X@W1.T+b1)@W2+b2
def train(Xt,yt,h,lam,seed):
    mu,sd=yt.mean(),yt.std(); z=(yt-mu)/sd
    rng=np.random.default_rng(seed); w0=rng.normal(0,0.5,5*h+1)
    n=len(Xt); sl=np.sqrt(lam)
    reg_rows=np.zeros((4*h,5*h+1))
    reg_rows[:3*h,:3*h]=np.eye(3*h)*sl; reg_rows[3*h:,4*h:5*h]=np.eye(h)*sl
    def resid(w):
        W1,b1,W2,b2=unpack(w,h)
        return np.concatenate([fwd(w,h,Xt)-z, sl*np.concatenate([W1.ravel(),W2])])
    def jac(w):
        W1,b1,W2,b2=unpack(w,h)
        T=np.tanh(Xt@W1.T+b1); G=(1-T**2)*W2            # n x h
        J=np.zeros((n,5*h+1))
        J[:,:3*h]=(G[:,:,None]*Xt[:,None,:]).reshape(n,3*h)
        J[:,3*h:4*h]=G; J[:,4*h:5*h]=T; J[:,5*h]=1
        return np.vstack([J,reg_rows])
    r=least_squares(resid,w0,jac=jac,method="lm",max_nfev=3000)
    return lambda Xn: fwd(r.x,h,Xn)*sd+mu
H=[1,2,3,4,6]; LAM=[1e-8,1e-3,1e-2,1e-1,1.0]; SEEDS=range(10)
def loo_ann(h,lam,seeds,idx=None):
    idx = list(range(15)) if idx is None else idx
    P=np.zeros((len(seeds),len(idx)))
    for k,i in enumerate(idx):
        tr=[j for j in idx if j!=i]
        for s_i,s in enumerate(seeds):
            P[s_i,k]=train(X[tr],y[tr],h,lam,s)(X[[i]])[0]
    return P
t0=time.time()
grid={}
for h in H:
    for lam in LAM:
        P=loo_ann(h,lam,SEEDS)
        per=[metrics(y,P[s]) for s in range(len(SEEDS))]
        ens=metrics(y,P.mean(0))
        grid[f"h{h}_l{lam:g}"]=dict(h=h,lam=lam,
            RMSE_mean=float(np.mean([m["RMSE"] for m in per])), RMSE_sd=float(np.std([m["RMSE"] for m in per])),
            MAPE_mean=float(np.mean([m["MAPE"] for m in per])), Q2_mean=float(np.mean([m["Q2"] for m in per])),
            RMSE_min=float(np.min([m["RMSE"] for m in per])), RMSE_max=float(np.max([m["RMSE"] for m in per])),
            ens=ens)
        if h==6 and lam==1e-8: preds["ann_361"]=P.mean(0); P361=P
print("grid done",time.time()-t0,file=sys.stderr)
# thesis-style in-sample evaluation of 3-6-1 (no regularisation), all 15 points
ins=[metrics(y,train(X,y,6,1e-8,s)(X)) for s in SEEDS]
insample["ann_361"]=dict(RMSE_mean=float(np.mean([m["RMSE"] for m in ins])), RMSE_min=float(np.min([m["RMSE"] for m in ins])),
    MAPE_mean=float(np.mean([m["MAPE"] for m in ins])), MAPE_min=float(np.min([m["MAPE"] for m in ins])),
    r_mean=float(np.mean([m["r"] for m in ins])))
# nested LOOCV: inner LOO over grid (3 seeds) picks (h,lam); outer prediction is 10-seed ensemble
nested=np.zeros(15); chosen=[]
for i in range(15):
    tr=[j for j in range(15) if j!=i]; best=None
    for h in H:
        for lam in LAM:
            P=loo_ann(h,lam,range(3),idx=tr)
            r=np.sqrt(((P-y[tr])**2).mean())
            if best is None or r<best[0]: best=(r,h,lam)
    chosen.append([best[1],best[2]]); print("outer",i,best,time.time()-t0,file=sys.stderr,flush=True)
    nested[i]=np.mean([train(X[tr],y[tr],best[1],best[2],s)(X[[i]])[0] for s in SEEDS])
print("nested done",time.time()-t0,file=sys.stderr)
preds["ann_nested"]=nested
loo["ann_nested"]=metrics(y,nested); loo["ann_nested"]["chosen"]=chosen
loo["ann_361_per_seed"]=grid["h6_l1e-08"]
loo["ann_361_ensemble"]=grid["h6_l1e-08"]["ens"]
best_key=min(grid,key=lambda k:grid[k]["RMSE_mean"])
loo["ann_best_of_grid"]=dict(key=best_key,**grid[best_key])
res.update(loo=loo, insample=insample, ann_grid=grid,
    replicate_sd=float(np.sqrt(ss_pe/dof_pe)), centre=cen.tolist())
pd.DataFrame(dict(run=df.run,N=df.N,f=df.f,d=df.d,Ra=y,rsm_fit=red.fittedvalues.values,
    rsm_full_fit=full.fittedvalues.values, **{f"loo_{k}":v for k,v in preds.items()})).to_csv(f"{OUT}/predictions.csv",index=False)
json.dump(res,open(f"{OUT}/results.json","w"),indent=1,default=float)
print(full.summary())
print(a)
