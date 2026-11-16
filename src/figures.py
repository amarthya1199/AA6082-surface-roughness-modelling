"""Publication figures for the AA6082 Box-Behnken re-analysis."""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.optimize import least_squares

# usage: python src/figures.py results   (run src/analysis.py first)
OUT = sys.argv[1]
r = json.load(open(f"{OUT}/results.json"))
P = pd.read_csv(f"{OUT}/predictions.csv")
FIG = f"{OUT}/figures"
import os; os.makedirs(FIG, exist_ok=True)
df = P[["run","N","f","d","Ra"]].copy()
df["x1"]=(df.N-3000)/1500; df["x2"]=(df.f-350)/150; df["x3"]=(df.d-0.4)/0.2
red = smf.ols("Ra ~ x1+x2+x3+I(x1**2)+I(x3**2)+x1:x3", df).fit()

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":0.6,
    "xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.direction":"out","ytick.direction":"out",
    "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"savefig.dpi":600})
INK="#222222"; MUTED="#777777"; C1="#2a6fb0"; C2="#d0702b"; C3="#4d9a55"
MM=1/25.4

def save(fig,name):
    fig.savefig(f"{FIG}/{name}.png",bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf",bbox_inches="tight"); plt.close(fig)

def pred(x1,x2,x3):
    d0=pd.DataFrame(dict(x1=np.atleast_1d(x1),x2=np.atleast_1d(x2),x3=np.atleast_1d(x3)))
    return red.get_prediction(d0).summary_frame(alpha=0.05)

# ---- Fig 2: main-effect curves (others at centre) with 95% CI, measured points on that axis
fig,axs=plt.subplots(1,3,figsize=(174*MM,52*MM),sharey=True)
specs=[("x1","N","Spindle speed N (rpm)",3000,1500),("x2","f","Feed rate f (mm/min)",350,150),("x3","d","Depth of cut d (mm)",0.4,0.2)]
for ax,(xc,col,lab,c,s) in zip(axs,specs):
    g=np.linspace(-1,1,101); kw={"x1":0*g,"x2":0*g,"x3":0*g}; kw[xc]=g
    fr=pred(kw["x1"],kw["x2"],kw["x3"]); xv=c+s*g
    ax.fill_between(xv,fr.mean_ci_lower,fr.mean_ci_upper,color=C1,alpha=0.18,lw=0)
    ax.plot(xv,fr["mean"],color=C1,lw=1.6)
    # runs on this axis (other two factors at centre)
    others=[o for o in ("x1","x2","x3") if o!=xc]
    m=(df[others[0]]==0)&(df[others[1]]==0)
    ax.plot(df.loc[m,col],df.loc[m,"Ra"],"o",ms=4.5,mfc="white",mec=INK,mew=0.8)
    ax.set_xlabel(lab); ax.set_xticks([c-s,c,c+s])
    ax.grid(axis="y",color="#e6e6e6",lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel(r"Predicted $R_a$ ($\mu$m)")
for ax,t in zip(axs,"abc"): ax.text(-0.02,1.04,t,transform=ax.transAxes,fontweight="bold",fontsize=9)
save(fig,"Fig1_main_effects")

# ---- Fig 3: contour maps (reduced model)
fig,axs=plt.subplots(1,3,figsize=(174*MM,58*MM))
g=np.linspace(-1,1,121); A,B=np.meshgrid(g,g)
opt=r["optimum"]["red"]
maps=[("x1","x2",{"x3":0},"N (rpm)","f (mm/min)",(3000,1500),(350,150),"d = 0.4 mm"),
      ("x1","x3",{"x2":-1},"N (rpm)","d (mm)",(3000,1500),(0.4,0.2),"f = 200 mm/min"),
      ("x2","x3",{"x1":0},"f (mm/min)","d (mm)",(350,150),(0.4,0.2),"N = 3000 rpm")]
levels=np.arange(1.0,7.01,0.5)
for ax,(xa,xb,fix,lx,ly,(ca,sa),(cb,sb),title) in zip(axs,maps):
    kw={xa:A.ravel(),xb:B.ravel(),**{k:np.full(A.size,v) for k,v in fix.items()}}
    Z=red.predict(pd.DataFrame(kw)).values.reshape(A.shape)
    X=ca+sa*A; Y=cb+sb*B
    cf=ax.contourf(X,Y,Z,levels=levels,cmap="Blues",extend="both")
    cs=ax.contour(X,Y,Z,levels=levels,colors=INK,linewidths=0.4)
    ax.clabel(cs,fmt="%.1f",fontsize=6.5,inline_spacing=2)
    # design points lying in this plane
    k=list(fix)[0]; m=df[k]==fix[k]
    ax.plot(ca+sa*df.loc[m,xa],cb+sb*df.loc[m,xb],"o",ms=3.5,mfc="white",mec=INK,mew=0.7)
    if fix=={"x2":-1}:
        ax.plot(opt["N"],opt["d"],marker="*",ms=9,mfc="white",mec=INK,mew=0.8)
    ax.set_xlabel(lx); ax.set_ylabel(ly); ax.set_title(title,fontsize=8,color=MUTED)
    ax.set_xticks([ca-sa,ca,ca+sa]); ax.set_yticks([cb-sb,cb,cb+sb])
    for s in ax.spines.values(): s.set_visible(True)
for ax,t in zip(axs,"abc"): ax.text(-0.02,1.1,t,transform=ax.transAxes,fontweight="bold",fontsize=9)
fig.tight_layout(w_pad=1.2)
cb=fig.colorbar(cf,ax=axs,shrink=0.85,pad=0.015); cb.set_label(r"$R_a$ ($\mu$m)"); cb.outline.set_linewidth(0.5)
save(fig,"Fig2_contours")

# ---- Fig 4: Ra vs feed per revolution, marker by depth of cut
fig,ax=plt.subplots(figsize=(84*MM,62*MM))
df["fr"]=df.f/df.N
mk={0.2:("o",C1),0.4:("s",C2),0.6:("^",C3)}
for dv,(m,c) in mk.items():
    s=df[df.d==dv]; ax.plot(s.fr,s.Ra,m,ms=5,mfc=c,mec="white",mew=0.6,label=f"d = {dv} mm",ls="none")
ax.set_xlabel("Feed per revolution f/N (mm/rev)"); ax.set_ylabel(r"Measured $R_a$ ($\mu$m)")
ax.grid(color="#e6e6e6",lw=0.5); ax.set_axisbelow(True); ax.legend(loc="upper left",handletextpad=0.3)
save(fig,"Fig3_feed_per_rev")

# ---- Fig 5: predicted vs measured, LOOCV
h=6
def unpack(w):
    return w[:3*h].reshape(h,3),w[3*h:4*h],w[4*h:5*h],w[5*h]
def fwd(w,X):
    W1,b1,W2,b2=unpack(w); return np.tanh(X@W1.T+b1)@W2+b2
def train(Xt,yt,lam,seed):
    mu,sd=yt.mean(),yt.std(); z=(yt-mu)/sd; n=len(Xt); sl=np.sqrt(lam)
    rr=np.zeros((4*h,5*h+1)); rr[:3*h,:3*h]=np.eye(3*h)*sl; rr[3*h:,4*h:5*h]=np.eye(h)*sl
    def res(w):
        W1,b1,W2,b2=unpack(w); return np.concatenate([fwd(w,Xt)-z,sl*np.concatenate([W1.ravel(),W2])])
    def jac(w):
        W1,b1,W2,b2=unpack(w); T=np.tanh(Xt@W1.T+b1); G=(1-T**2)*W2; J=np.zeros((n,5*h+1))
        J[:,:3*h]=(G[:,:,None]*Xt[:,None,:]).reshape(n,3*h); J[:,3*h:4*h]=G; J[:,4*h:5*h]=T; J[:,5*h]=1
        return np.vstack([J,rr])
    w=least_squares(res,np.random.default_rng(seed).normal(0,.5,5*h+1),jac=jac,method="lm",max_nfev=3000).x
    return lambda X: fwd(w,X)*sd+mu
X=df[["x1","x2","x3"]].values; y=df.Ra.values
allp=[]
for s in range(10):
    p=np.array([train(np.delete(X,i,0),np.delete(y,i),1e-8,s)(X[[i]])[0] for i in range(15)])
    allp.append((np.sqrt(((p-y)**2).mean()),s,p))
allp.sort(key=lambda t:t[0]); best=allp[4]   # lower median of ten initialisations
per=[dict(seed=s,RMSE=float(e),MAE=float(np.abs(p-y).mean()),MAPE=float((np.abs(p-y)/y).mean()*100),Q2=float(1-((p-y)**2).sum()/((y-y.mean())**2).sum()),maxAE=float(np.abs(p-y).max())) for e,s,p in allp]
json.dump(per,open(f"{OUT}/ann361_per_seed.json","w"),indent=1)
ins=train(X,y,1e-8,best[1])(X)
json.dump(dict(seed=best[1],loo_rmse=best[0],loo=best[2].tolist(),insample=ins.tolist()),open(f"{OUT}/ann361_repr.json","w"))
fig,axs=plt.subplots(1,3,figsize=(174*MM,60*MM),sharex=True,sharey=True)
lim=(1.0,7.5)
panels=[("Reduced RSM, leave-one-out",[(P.loo_rsm_red,"o",C1,"LOOCV")]),
        ("ANN, nested leave-one-out",[(P.loo_ann_nested,"o",C2,"LOOCV")]),
        ("ANN 3-6-1, no regularisation",[(ins,"o","white","fitted to all 15 runs"),(best[2],"o",C2,"LOOCV")])]
for ax,(title,series) in zip(axs,panels):
    ax.plot(lim,lim,ls="--",color=MUTED,lw=0.7)
    for vals,m,c,lab in series:
        ax.plot(y,vals,m,ms=4.5,mfc=c,mec=INK if c=="white" else "white",mew=0.7,ls="none",label=lab)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(r"Measured $R_a$ ($\mu$m)")
    ax.grid(color="#eeeeee",lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel(r"Predicted $R_a$ ($\mu$m)")
axs[2].legend(loc="upper left",fontsize=7,handletextpad=0.2)
for ax,t,(title,_) in zip(axs,"abc",panels): ax.set_title(r"$\bf{"+t+"}$  "+title,fontsize=8,loc="left",pad=6)
save(fig,"Fig4_pred_vs_meas")

# ---- Fig 6: ANN LOOCV RMSE vs hidden neurons
grid=r["ann_grid"]; H=[1,2,3,4,6]; LAM=[1e-8,1e-3,1e-2,1e-1,1.0]
cols={1e-8:"#9a9a9a",1e-3:"#9ecae1",1e-2:"#4292c6",1e-1:"#2171b5",1.0:"#08306b"}
mks={1e-8:"o",1e-3:"s",1e-2:"D",1e-1:"^",1.0:"v"}
fig,ax=plt.subplots(figsize=(84*MM,64*MM))
for lam in LAM:
    v=[grid[f"h{hh}_l{lam:g}"]["RMSE_mean"] for hh in H]
    v=[x if x<1.2 else np.nan for x in v]
    lab = "λ ≈ 0" if lam==1e-8 else f"λ = {lam:g}"
    ax.plot(H,v,marker=mks[lam],ms=4,lw=1.2,color=cols[lam],label=lab,mec="white",mew=0.5)
rsm=r["loo"]["rsm_red"]["RMSE"]; rep=r["replicate_sd"]
ax.axhline(rsm,color=C2,lw=1.2,ls="-"); ax.text(6.15,rsm,"reduced RSM",va="center",fontsize=7,color=INK)
ax.axhline(rep,color=MUTED,lw=0.8,ls=":"); ax.text(6.15,rep,"replicate SD",va="center",fontsize=7,color=INK)
ax.set_xticks(H); ax.set_xlim(0.7,6.3); ax.set_ylim(0,1.0)
ax.set_xlabel("Hidden neurons"); ax.set_ylabel(r"LOOCV RMSE ($\mu$m)")
ax.grid(axis="y",color="#e6e6e6",lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper right",fontsize=6.5,ncol=2,handlelength=1.6,columnspacing=0.8)
save(fig,"Fig5_ann_grid")
print("repr seed",best[1],"loo rmse",best[0],"ins rmse",np.sqrt(((ins-y)**2).mean()), "ins mape", np.mean(np.abs(ins-y)/y)*100,"loo mape",np.mean(np.abs(best[2]-y)/y)*100)
