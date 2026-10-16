import matplotlib; matplotlib.use('SVG')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
d = r'H:\nature water(water research) new idea\cascade_water_risk\docs\latex\figures'
NAVY='#1B3A5C'; TEAL='#1B9E8B'; ORANGE='#E8833A'; GREY='#E8ECF1'; RED='#C0392B'
fig,ax = plt.subplots(figsize=(7.5,3.2))
ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis('off')
plt.rcParams['font.family']='sans-serif'
def box(x,y,w,h,label,sub,fc):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=fc,ec=NAVY,lw=1.2)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2+0.15,label,ha='center',va='center',fontsize=9,fontweight='bold',color=NAVY if fc!=NAVY else 'white')
    ax.text(x+w/2,y+h/2-0.28,sub,ha='center',va='center',fontsize=7,color=NAVY if fc!=NAVY else 'white')
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1.4,color=NAVY))
box(0.2,1.5,1.8,1.2,'Forecast','ERA5 / GraphCast\nlead 1-7','white')
box(2.6,1.5,2.0,1.2,'Hydro backbone','LSTM  lookback 30\n17 static attrs',GREY)
box(5.2,1.5,1.6,1.2,'Flow','NSE 0.34\nmm / day',TEAL)
box(7.4,1.5,2.2,1.2,'Water-quality head','14-d flow + season + ag\nmiss-rate / 2.4x rule',ORANGE)
arrow(2.0,2.1,2.6,2.1); arrow(4.6,2.1,5.2,2.1); arrow(6.8,2.1,7.4,2.1)
ax.text(5,0.7,'Paired design: same head | truth forcing vs forecast forcing   |   CR(L) retention   |   zero-shot transfer',ha='center',fontsize=7.5,color=NAVY,
 bbox=dict(fc='white',ec=NAVY,boxstyle='round,pad=0.3'))
ax.set_title('FiCaR-Net  forecast to runoff to nitrate-exceedance cascade',fontweight='bold',loc='left',fontsize=10,color=NAVY)
plt.tight_layout()
plt.savefig(os.path.join(d,'fig_model.svg'),bbox_inches='tight')
plt.savefig(os.path.join(d,'fig_model.pdf'),bbox_inches='tight')
print('done')
