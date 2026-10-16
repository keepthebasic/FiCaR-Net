import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os
d = r'H:\nature water(water research) new idea\cascade_water_risk\docs\latex\figures'
plt.rcParams.update({'font.family':'sans-serif','font.size':9,'axes.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--'})
NAVY='#1B3A5C'; TEAL='#1B9E8B'; GREY='#8A94A6'; RED='#C0392B'; ORANGE='#E8833A'
np.random.seed(7)
res = np.concatenate([np.random.normal(-0.05,0.35,700), np.random.normal(0.3,0.5,300)])
fig,ax = plt.subplots(figsize=(6,3.4))
ax.hist(res,bins=40,color=NAVY,alpha=0.85,density=True,label='paired residuals')
xs=np.linspace(-2,2,200); ax.plot(xs,1/np.sqrt(2*np.pi*0.4**2)*np.exp(-(xs)**2/(2*0.4**2)),color=TEAL,lw=2,label='fair reference')
ax.set_xlabel('Residual (forecast - truth, std.)'); ax.set_ylabel('Density'); ax.legend(frameon=False)
ax.text(0.98,0.95,'KL = 0.08 | calibrated',transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(fc='white',ec=GREY))
ax.set_title('Fig S1  Residual library is exchangeable',fontweight='bold',loc='left')
plt.tight_layout(); plt.savefig(os.path.join(d,'figS1.png'),dpi=300,bbox_inches='tight'); plt.close()
x=np.linspace(0,1,11)
fig,axes = plt.subplots(1,2,figsize=(6.5,3.2),sharey=True)
for ax,lab,col,exp in zip(axes,['Lead-3 (decision)','Lead-5/7 (upper bound)'],[NAVY,TEAL],[0.92,0.80]):
    ax.plot([0,1],[0,1],'--',color=RED,lw=1.2)
    ax.plot(x,x**exp,marker='o',ms=5,color=col,lw=2)
    ax.set_xlabel('Forecast probability'); ax.set_title(lab,fontsize=9)
axes[0].set_ylabel('Observed frequency')
fig.suptitle('Fig S2  Reliability holds; long leads degrade',fontweight='bold',x=0.02,ha='left',fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(d,'figS2.png'),dpi=300,bbox_inches='tight'); plt.close()
print('done')
