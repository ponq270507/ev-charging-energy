"""Export four standalone vector figures directly from source data.

Usage: python redraw_figures.py --project-root ADY_Group25 --output-dir figures
Dependencies: numpy, pandas, reportlab. No page from the report is imported.
"""
from pathlib import Path
import argparse,json,math
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib import colors

W=347.0
BLUE=colors.HexColor('#27648c'); GREEN=colors.HexColor('#24745b')
GREY=colors.HexColor('#747b82'); ORANGE=colors.HexColor('#b75b16')
INK=colors.HexColor('#253340')

def text(c,x,y,s,size=7.5,align='left',color=INK,bold=False):
    c.setFillColor(color); c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
    {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,y,str(s))

def line(c,x1,y1,x2,y2,color=GREY,lw=.6,dash=None):
    c.setStrokeColor(color);c.setLineWidth(lw);c.setDash(dash or [])
    c.line(x1,y1,x2,y2);c.setDash([])

def arrow(c,x1,y1,x2,y2,color=GREY):
    line(c,x1,y1,x2,y2,color)
    a=math.atan2(y2-y1,x2-x1); r=3
    p=c.beginPath();p.moveTo(x2,y2)
    p.lineTo(x2-r*math.cos(a-.45),y2-r*math.sin(a-.45))
    p.lineTo(x2-r*math.cos(a+.45),y2-r*math.sin(a+.45));p.close()
    c.setFillColor(color);c.drawPath(p,stroke=0,fill=1)

def box(c,x,y,w,h,lines,fill):
    c.setFillColor(colors.HexColor(fill));c.setStrokeColor(GREY);c.setLineWidth(.55)
    c.roundRect(x,y,w,h,1.5,stroke=1,fill=1)
    for i,s in enumerate(lines):text(c,x+w/2,y+h/2+(len(lines)-1)*4.1-i*8.2,s,7.2,'center')

def begin(out,n,h):
    c=canvas.Canvas(str(out/f'figure{n}.pdf'),pagesize=(W,h),pageCompression=1)
    c.setTitle(f'Group 25 Figure {n}');c.setAuthor('Group 25')
    return c

def pipeline(out):
    c=begin(out,1,198)
    data='#e8f1f7';model='#faeddf';ev='#e5f2eb'
    box(c,5,167,91,24,['Raw Korean data','72,856; 13 fields'],data)
    box(c,119,167,94,24,['Six quality rules'],data)
    box(c,248,167,94,24,['Quarantined','3,286 sessions'],data)
    arrow(c,98,179,115,179);arrow(c,216,179,244,179)
    box(c,119,132,94,24,['Retained','69,570 sessions'],data)
    arrow(c,166,165,166,159)
    line(c,166,132,166,123);line(c,67,123,253,123)
    arrow(c,67,123,67,115);arrow(c,253,123,253,115)
    box(c,5,85,125,28,['Nine SQL queries','Full dataset; pandas checks'],data)
    box(c,157,85,185,28,['137 predictors before encoding','Eligible completed histories'],data)
    arrow(c,250,83,250,77)
    box(c,157,49,185,27,['Seven families + two baselines','Original CV / common-input nested CV'],model)
    arrow(c,250,47,250,40)
    box(c,157,12,185,27,['Original blend: later-date test','Nested CV: development windows only'],ev)
    box(c,5,12,125,27,['Identity / history ablation','Fixed LightGBM; same folds'],ev)
    line(c,155,62,67,62);arrow(c,67,62,67,41)
    for x,label,fill in [(62,'Data',data),(112,'Model comparison',model),(219,'Evaluation',ev)]:
        c.setFillColor(colors.HexColor(fill));c.setStrokeColor(GREY);c.rect(x,1.5,12,5,fill=1,stroke=1)
        text(c,x+17,1.5,label,7)
    c.save()

def timeline(out):
    c=begin(out,2,136)
    x=lambda t:10+t*32.2
    t=6.8
    line(c,x(t),43,x(t),115,ORANGE,.8,[3,2])
    text(c,x(t),122,'New arrival t',8,'center',ORANGE)
    rows=[(99,.8,3.7,4.35,'A: available',GREEN),(73,1.8,6.35,7,'B: buffer pending',GREY),(47,3,8.25,8.9,'C: still connected',GREY)]
    for y,start,end,avail,label,col in rows:
        line(c,x(start),y,x(end),y,col,6)
        line(c,x(end),y,x(avail),y,col,.7,[1,1.7]);line(c,x(avail),y-6,x(avail),y+6,col,.8)
        text(c,x(start),y+7,label,7.5,color=col)
        # All availability labels sit above the bars, away from the tick and time axis.
        if label.startswith('B'):
            tx=x(avail)+40;ty=y+14
            line(c,x(avail)+2,y+5,tx-3,ty-2,col,.5)
        else:tx=x(avail)+6;ty=y+11
        text(c,tx,ty,'End + 60 s',7.2,color=col)
    arrow(c,x(.5),30,x(10),30,INK)
    text(c,x(10),17,'Time',7.5,'right')
    text(c,x(.5),14,'Known at t: account, charger, arrival time',7.5)
    text(c,x(.5),2,"Unknown at t: this session's duration and energy",7.5)
    c.save()

def axes(c,x,y,w,h,xlim,ylim,xticks,yticks,xlabel,ylabel,title):
    fx=lambda v:x+(v-xlim[0])/(xlim[1]-xlim[0])*w
    fy=lambda v:y+(v-ylim[0])/(ylim[1]-ylim[0])*h
    line(c,x,y,x+w,y,INK,.55);line(c,x,y,x,y+h,INK,.55)
    for v in xticks:
        line(c,fx(v),y,fx(v),y-2.5,INK,.55);text(c,fx(v),y-12,str(v),7.5,'center')
    for v in yticks:
        line(c,x-2.5,fy(v),x,fy(v),INK,.55);text(c,x-5,fy(v)-2.5,str(v),7.5,'right')
    text(c,x+w/2,y-25,xlabel,8,'center')
    c.saveState();c.translate(x-30,y+h/2);c.rotate(90);text(c,0,0,ylabel,8,'center');c.restoreState()
    text(c,x+w/2,y+h+12,title,8,'center')
    return fx,fy

def hourly(root,out):
    h=pd.read_csv(root/'analysis_outputs/revision/hour_mean_intervals.csv')
    assert h.loc[h.arrival_hour.eq(23),'sessions'].item()==1669
    assert h.sessions.sum()==69570
    c=begin(out,3,169)
    # Wider right-side spacing prevents any annotation from crossing the panel boundary.
    fx,fy=axes(c,36,38,117,109,(-.6,23.6),(0,7200),[0,6,12,18,23],[0,2000,4000,6000],'Arrival hour','Sessions','(a) Arrival frequency')
    c.setFillColor(BLUE)
    for r in h.itertuples():c.rect(fx(r.arrival_hour-.41),fy(0),fx(r.arrival_hour+.41)-fx(r.arrival_hour-.41),fy(r.sessions)-fy(0),fill=1,stroke=0)
    for hour in [18,23]:
        r=h[h.arrival_hour.eq(hour)].iloc[0]
        # Exact hour location; lift the last label above its taller neighbour.
        # The leader ends at the 23:00 bar, so it cannot be mistaken for 22:00.
        if hour==23:
            label_level=max(r.sessions,h.loc[h.arrival_hour.eq(22),'sessions'].item())+420
            line(c,fx(hour),fy(r.sessions)+2,fx(hour),fy(label_level)-3,INK,.45)
            ty=fy(label_level)
        else:ty=fy(r.sessions)+5
        text(c,fx(hour),ty,f'{int(r.sessions):,}',7.5,'center',bold=True)
    gx,gy=axes(c,222,38,117,109,(-.6,23.6),(0,35),[0,6,12,18,23],[0,10,20,30],'Arrival hour','Mean energy (kWh)','(b) Eventual energy')
    poly=c.beginPath()
    for i,r in enumerate(h.itertuples()):
        (poly.moveTo if i==0 else poly.lineTo)(gx(r.arrival_hour),gy(r.lower_95_kWh))
    for r in reversed(list(h.itertuples())):poly.lineTo(gx(r.arrival_hour),gy(r.upper_95_kWh))
    poly.close();c.setFillColor(colors.HexColor('#d5e3ed'));c.drawPath(poly,fill=1,stroke=0)
    rows=list(h.itertuples())
    for a,b in zip(rows,rows[1:]):line(c,gx(a.arrival_hour),gy(a.mean_kWh),gx(b.arrival_hour),gy(b.mean_kWh),BLUE,.9)
    for r in rows:
        if r.sessions<500:
            x=gx(r.arrival_hour);y=gy(r.mean_kWh);p=c.beginPath();p.moveTo(x,y+2.6);p.lineTo(x-2.3,y-1.8);p.lineTo(x+2.3,y-1.8);p.close()
            c.setFillColor(colors.white);c.setStrokeColor(BLUE);c.setLineWidth(.6);c.drawPath(p,fill=1,stroke=1)
    c.save()

PALETTE=np.array([[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]])/255
def palette(v):
    t=np.clip(v,0,1)*(len(PALETTE)-1);i=min(int(t),len(PALETTE)-2);rgb=PALETTE[i]*(1-(t-i))+PALETTE[i+1]*(t-i)
    return colors.Color(*rgb)

def hexcounts(xx,yy,xlim,ylim,nx=35):
    ny=int(nx/np.sqrt(3));sx=(xlim[1]-xlim[0])/nx;sy=(ylim[1]-ylim[0])/ny
    ix=(xx-xlim[0])/sx;iy=(yy-ylim[0])/sy
    a=np.rint(ix).astype(int);b=np.rint(iy).astype(int);u=np.floor(ix).astype(int);v=np.floor(iy).astype(int)
    first=(ix-a)**2+3*(iy-b)**2 < (ix-u-.5)**2+3*(iy-v-.5)**2
    bins={}
    for k in range(len(xx)):
        key=(int(a[k]),int(b[k]),0) if first[k] else (int(u[k]),int(v[k]),1)
        bins[key]=bins.get(key,0)+1
    return [(xlim[0]+(i+.5*f)*sx,ylim[0]+(j+.5*f)*sy,n) for (i,j,f),n in bins.items()],sx,sy

def colorbar(c,x,y,w,maximum):
    vmax=max(1,math.log10(maximum))
    for i in range(150):
        c.setFillColor(palette(i/149));c.rect(x+w*i/150,y,w/150+.05,5,fill=1,stroke=0)
    c.setStrokeColor(INK);c.setLineWidth(.35);c.rect(x,y,w,5,fill=0,stroke=1)
    for v in [1,10,100,1000]:
        if v>maximum:continue
        xx=x+math.log10(v)/vmax*w;line(c,xx,y,xx,y-2,INK,.4);text(c,xx,y-11,f'{v:,}',7,'center')
    text(c,x+w/2,y-23,'Sessions per hexagon',7.5,'center')

def errors(root,out):
    data=pd.read_csv(root/'analysis_outputs/models/personalized_test_predictions.csv')
    model=json.loads((root/'analysis_outputs/manifest.json').read_text())['selected_model']
    actual=data.Demand.to_numpy();pred=data[model].to_numpy();res=actual-pred
    assert len(data)==10999
    c=begin(out,4,202)
    for panel,(xx,yy,xlim,ylim,xlabel,ylabel,title) in enumerate([
        (actual,pred,(0,100),(0,100),'Actual energy (kWh)','Predicted energy (kWh)','(a) Prediction density'),
        (pred,res,(0,100),(-100,100),'Predicted energy (kWh)','Actual - predicted (kWh)','(b) Residual pattern')]):
        x=37 if panel==0 else 222;y=74;w=116;h=107
        fx,fy=axes(c,x,y,w,h,xlim,ylim,[0,50,100],[0,20,40,60,80,100] if panel==0 else [-100,-50,0,50,100],xlabel,ylabel,title)
        bins,sx,sy=hexcounts(xx,yy,xlim,ylim);maximum=max(v[2] for v in bins)
        c.saveState();clip=c.beginPath();clip.rect(x,y,w,h);c.clipPath(clip,stroke=0)
        vertices=[(.5,-.5),(.5,.5),(0,1),(-.5,.5),(-.5,-.5),(0,-1)]
        for cx,cy,count in bins:
            path=c.beginPath()
            for j,(vx,vy) in enumerate(vertices):(path.moveTo if j==0 else path.lineTo)(fx(cx+vx*sx),fy(cy+vy*sy/3))
            path.close();c.setFillColor(palette(math.log10(count)/max(1,math.log10(maximum))));c.drawPath(path,fill=1,stroke=0)
        if panel==0:line(c,fx(0),fy(0),fx(100),fy(100),ORANGE,.7,[3,2])
        else:
            line(c,fx(0),fy(0),fx(100),fy(0),ORANGE,.7,[3,2])
            line(c,fx(0),fy(0),fx(100),fy(-100),GREY,.6,[1,2])
            group=np.floor(pred/5).astype(int);med=[]
            for k in sorted(set(group)):
                if (group==k).sum()>=20:med.append((fx(5*k+2.5),fy(np.median(res[group==k]))))
            for col,lw in [(colors.black,2.6),(colors.white,1.3)]:
                for a,b in zip(med,med[1:]):line(c,*a,*b,col,lw)
        c.restoreState()
        if panel==1:
            line(c,x+52,y+h-9,x+65,y+h-9,colors.black,2.6);line(c,x+52,y+h-9,x+65,y+h-9,colors.white,1.3)
            text(c,x+69,y+h-11.5,'Bin median',7)
        colorbar(c,x,29,w,maximum)
    c.save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,default=Path('figures'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    pipeline(args.output_dir);timeline(args.output_dir);hourly(args.project_root,args.output_dir);errors(args.project_root,args.output_dir)
    print('Exported four standalone vector PDF figures; no report PDF imported.')
