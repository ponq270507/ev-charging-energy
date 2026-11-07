from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from redraw_vector_figures import pipeline,timeline,errors
from redraw_hour_matplotlib import main as hourly
FIG=ROOT/'report_source/figures'; FIG.mkdir(exist_ok=True)
pipeline(FIG); timeline(FIG); hourly(ROOT,FIG); errors(ROOT,FIG)
from PIL import Image
import pypdfium2 as pdfium
for n,name in [(1,'pipeline'),(2,'timeline'),(3,'hour'),(4,'errors')]:
    pdf=FIG/f'figure{n}.pdf'
    shutil.copy2(pdf,FIG/(name+'.pdf'))
    doc=pdfium.PdfDocument(str(pdf)); page=doc[0]
    bitmap=page.render(scale=2.5); image=bitmap.to_pil()
    image.save(FIG/(name+'.png'))
    image.save(FIG/f'figure{n}.png')
    page.close(); doc.close()
print('Four vector figures exported directly from code/data; no report PDF is imported.')
