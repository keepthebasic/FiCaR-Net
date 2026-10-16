import matplotlib; matplotlib.use('SVG')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
d = r'H:\nature water(water research) new idea\cascade_water_risk\docs\latex\figures'
NAVY='#1B3A5C'; TEAL='#1B9E8B'; ORANGE='#E8833A'; GREY='#E8ECF1'; LT='#F4F7FA'
fig,ax = plt.subplots(figsize=(8,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,5.2); ax.axis('off')
def box(x,y,w,h,t,s,fc='white',tc=NAVY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=fc,ec=NAVY,lw=1.1))
    ax.text(x+w/2,y+h/2+0.14,t,ha='center',va='center',fontsize=8,fontweight='bold',color=tc)
    ax.text(x+w/2,y+h/2-0.26,s,ha='center',va='center',fontsize=6.5,color=tc)
def ar(x1,y1,x2,y2,c=NAVY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,lw=1.2,color=c))
box(0.2,3.4,1.6,1.0,'Meteo forcing','P/T/ERA5-L\nT-30..T+7'); box(0.2,2.0,1.6,1.0,'Static attrs','17 attrs\nstandardised')
box(2.4,2.6,1.8,1.2,'LSTM encoder','2-layer x128\nlog-MSE','white'); box(2.4,1.2,1.8,0.9,'Residual lib.','paired T/F\nfair baseline',GREY)
box(4.8,2.6,1.7,1.2,'Forecast head','lead 1-7 flow\nNSE 0.34',TEAL,'white'); box(4.8,1.2,1.7,0.9,'CR module','CR(L) retention','white')
box(7.1,2.6,2.5,1.2,'FiCaR head','14-d flow+season+ag\nmiss-rate + cost 5x',ORANGE,'white')
ar(1.8,3.9,2.4,3.4); ar(1.8,2.5,2.4,3.0); ar(4.2,3.2,4.8,3.2); ar(4.2,1.6,4.8,1.6); ar(6.5,3.2,7.1,3.2)
ar(5.6,2.6,5.6,2.1)
ax.text(8.35,1.7,'2.4x rule',ha='center',fontsize=7,color=NAVY,bbox=dict(fc='white',ec=NAVY,boxstyle='round,pad=0.25')); ar(9.0,2.6,8.7,2.0)
ax.set_title('FiCaR-Net architecture  hydro backbone + forecast head + cascade-risk head',fontweight='bold',loc='left',fontsize=10,color=NAVY)
plt.tight_layout()
for ext in ['svg','pdf']:
    plt.savefig(os.path.join(d,f'ficarnet.{ext}'),bbox_inches='tight')
print('done')
