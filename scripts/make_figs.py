import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os
d = r'H:\nature water(water research) new idea\cascade_water_risk\docs\latex\figures'
os.makedirs(d, exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':9,'axes.linewidth':0.8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'--'})
NAVY='#1B3A5C'; TEAL='#1B9E8B'; ORANGE='#E8833A'; GREY='#8A94A6'; RED='#C0392B'
def save(n):
    plt.tight_layout(); plt.savefig(os.path.join(d,n),dpi=300,bbox_inches='tight'); plt.close()
leads=np.array([1,3,5,7]); dm=np.array([0.044,0.063,0.093,0.079]); nse=np.array([0.34,0.30,0.24,0.20])
fig,ax1=plt.subplots(figsize=(6,3.4))
ax1.plot(leads,dm,marker='o',ms=7,color=NAVY,lw=2,label='delta miss'); ax1.fill_between(leads,dm,alpha=0.08,color=NAVY)
ax1.set_xlabel('Forecast lead (days)'); ax1.set_ylabel('Miss-rate increase',color=NAVY); ax1.set_xticks(leads)
ax2=ax1.twinx(); ax2.bar(leads,nse,width=0.5,alpha=0.25,color=TEAL,label='flow NSE'); ax2.set_ylabel('Flow NSE',color=TEAL)
ax1.set_title('Fig 1  Forecast errors amplify into exceedance misses',fontweight='bold',loc='left')
save('fig1.png')
fig,ax=plt.subplots(figsize=(6,3.4))
x=np.arange(2); w=0.34
ax.bar(x-w/2,[0.263,0.022],w,color=[TEAL,NAVY],label='exceed rate')
ax.bar(x+w/2,[0.55,0.10],w,color=[ORANGE,GREY],label='miss rate')
ax.set_xticks(x); ax.set_xticklabels(['Agricultural season\n(n=263)','High-flow compound\n(n=7)'])
ax.set_ylabel('Rate'); ax.legend(frameon=False,ncol=2)
ax.set_title('Fig 2  Seasonal cascade vs event-density regimes',fontweight='bold',loc='left')
save('fig2.png')
fig,ax=plt.subplots(figsize=(6,3.4))
cats=['Dry holdout','Wet holdout','Cross-source','GRDC intercont.']; vals=[0.13,0.215,0.325,0.027]
ax.barh(cats,vals,color=[NAVY,TEAL,ORANGE,RED],height=0.55)
for i,v in enumerate(vals): ax.text(v+0.008,i,f'{v:.3f}',va='center',fontsize=8)
ax.set_xlabel('NSE (zero-shot)'); ax.set_title('Fig 3  Static backbone fails to transfer',fontweight='bold',loc='left')
save('fig3.png')
fig,ax=plt.subplots(figsize=(6,3.4))
ax.bar(['Oracle','FiCaR','Baseline'],[0.276,0.324,0.335],color=[TEAL,NAVY,GREY],width=0.55)
ax.set_ylabel('Expected cost (lead-3)'); ax.set_title('Fig 4  Cost wall + 2.4x flow rule (CI 1.58-4.99)',fontweight='bold',loc='left')
save('fig4.png')
fig,ax=plt.subplots(figsize=(6,3.4))
cr=[1.00,1.42,2.11,1.79]; cols=[GREY if v<=1 else NAVY for v in cr]
ax.bar(['L1','L3','L5','L7'],cr,color=cols,width=0.55); ax.axhline(1,color=RED,ls='--',lw=1.2)
for i,v in enumerate(cr): ax.text(i,v+0.05,f'{v:.2f}',ha='center',fontsize=8)
ax.set_ylabel('Retention CR(L)'); ax.set_title('Fig 5  Mid-range amplification exceeds baseline',fontweight='bold',loc='left')
save('fig5.png')
print('done')
