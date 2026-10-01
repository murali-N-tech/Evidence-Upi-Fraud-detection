import re, subprocess, copy
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

REFS = {
 "rastogi": 'S. Rastogi, C. Panse, A. Sharma, and V. M. Bhimavarapu, "Unified Payment Interface (UPI): A digital innovation and its impact on financial inclusion and economic development," *Universal J. Accounting Finance*, vol. 9, no. 3, pp. 518–530, 2021, doi: 10.13189/ujaf.2021.090326.',
 "bolton": 'R. J. Bolton and D. J. Hand, "Statistical fraud detection: A review," *Statist. Sci.*, vol. 17, no. 3, pp. 235–255, 2002, doi: 10.1214/ss/1042727940.',
 "dalpozzolo": 'A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: A realistic modeling and a novel learning strategy," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 29, no. 8, pp. 3784–3797, Aug. 2018, doi: 10.1109/TNNLS.2017.2736643.',
 "braithwaite": 'J. Braithwaite, "\'Authorized push payment\' bank fraud: What does an effective regulatory response look like?," *J. Financial Regulation*, vol. 10, no. 2, pp. 174–193, 2024, doi: 10.1093/jfr/fjae006.',
 "spade": 'J. Jiang, Y. Chen, B. He, M. Chen, and J. Chen, "Spade+: A generic real-time fraud detection framework on dynamic graphs," *IEEE Trans. Knowl. Data Eng.*, vol. 36, no. 11, pp. 7058–7073, Nov. 2024, doi: 10.1109/TKDE.2024.3394155.',
 "fraudar": 'B. Hooi, H. A. Song, A. Beutel, N. Shah, K. Shin, and C. Faloutsos, "FRAUDAR: Bounding graph fraud in the face of camouflage," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 895–904, doi: 10.1145/2939672.2939747.',
 "pgltse": 'J. Ma, S. Xiang, Q. Li, L. Yuan, D. Cheng, and C. Jiang, "Parallel graph learning with temporal stamp encoding for fraudulent transactions detections," *IEEE Trans. Big Data*, vol. 11, no. 4, pp. 1945–1958, 2025, doi: 10.1109/TBDATA.2024.3499338.',
 "secureupi": 'R. Rani, A. Alam, and A. Javed, "Secure UPI: Machine learning-driven fraud detection system for UPI transactions," in *Proc. 2nd Int. Conf. Disruptive Technol. (ICDT)*, 2024, pp. 924–928, doi: 10.1109/ICDT61202.2024.10489682.',
 "caregnn": 'Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, and P. S. Yu, "Enhancing graph neural network-based fraud detectors against camouflaged fraudsters," in *Proc. 29th ACM Int. Conf. Inf. Knowl. Manage. (CIKM)*, 2020, pp. 315–324, doi: 10.1145/3340531.3411903.',
 "weber": 'M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. E. Leiserson, "Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics," arXiv:1908.02591, 2019.',
 "apgnn": 'X. Wang, H. Yu, J. Guo, P. Li, and X. Luo, "Towards fraud detection via fine-grained classification of user behavior," *IEEE Trans. Big Data*, vol. 11, no. 4, pp. 1994–2007, 2025, doi: 10.1109/TBDATA.2024.3517313.',
 "saito": 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.',
 "davis": 'J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. 23rd Int. Conf. Mach. Learn. (ICML)*, 2006, pp. 233–240, doi: 10.1145/1143844.1143874.',
 "niculescu": 'A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in *Proc. 22nd Int. Conf. Mach. Learn. (ICML)*, 2005, pp. 625–632, doi: 10.1145/1102351.1102430.',
 "zadrozny": 'B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate multiclass probability estimates," in *Proc. 8th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2002, pp. 694–699, doi: 10.1145/775047.775151.',
 "brier": 'G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Rev.*, vol. 78, no. 1, pp. 1–3, 1950.',
 "friedman": 'J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.',
 "paysim": 'E. A. Lopez-Rojas, A. Elmir, and S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection," in *Proc. Eur. Model. Simul. Symp. (EMSS)*, 2016, pp. 249–255.',
 "smote": 'N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002, doi: 10.1613/jair.953.',
 "lightgbm": 'G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017, pp. 3146–3154.',
 "shap": 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017, pp. 4765–4774.',
 "sklearn": 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
 "breiman": 'L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.',
}

import json as _json
src = "".join(open(f).read() for f in ["p1.md","p2.md","p3.md","p4a.md","p4b.md","p5.md"])
src = src.replace("[[ABSTRACT]]", open("abstract.md").read().strip())
for k in ("TAB3", "TAB4", "TAB5", "TAB6"):
    src = src.replace("[[%s]]" % k, open("tabs/%s.md" % k.lower()).read().strip())
_facts = _json.load(open("tabs/facts.json"))
def _fact(m):
    v = str(_facts[m.group(1)])
    return v.replace("-", "\u2212") if v.startswith("-") else v
src = re.sub(r"\[\[F:(\w+)\]\]", _fact, src)
assert "[[F:" not in src and "[[TAB" not in src, "unfilled placeholder"
# typographic minus in table cells and signed numbers
src = re.sub(r"(?<=[\s|(])-(?=\d)", "\u2212", src)
# strip math from figure captions
def capfix(m):
    c = m.group(1).replace("$t$","t")
    assert "$" not in c, c
    return "![" + c + "]("
src = re.sub(r"!\[(.*?)\]\(", capfix, src)
# citations by first appearance
order = []
for k in re.findall(r"\{\{(\w+)\}\}", src):
    if k not in order: order.append(k)
missing = [k for k in order if k not in REFS]; assert not missing, missing
unused = [k for k in REFS if k not in order]; print("not cited (dropped):", unused)
num = {k: i+1 for i, k in enumerate(order)}
src = re.sub(r"\{\{(\w+)\}\}", lambda m: "[%d]" % num[m.group(1)], src)
refs = "\n".join('::: {custom-style="Reference"}\n[%d]\t%s\n:::\n' % (num[k], REFS[k]) for k in order)
src += "\n" + refs
open("paper.md", "w").write(src)
print("refs:", len(order))
subprocess.run(["pandoc", "paper.md", "-o", "raw.docx", "--reference-doc=ref.docx", "--resource-path=."], check=True)

d = Document("raw.docx")
W = lambda tag: qn("w:" + tag)

def set_font(style, size=None, bold=None, italic=None, smallcaps=None, name="Times New Roman"):
    f = style.font; f.name = name; f.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(W("rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ["ascii", "hAnsi", "cs", "eastAsia"]: rf.set(W(a), name)
    for a in list(rf.attrib):
        if "Theme" in a: del rf.attrib[a]
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if smallcaps is not None: f.small_caps = smallcaps

S = {s.name: s for s in d.styles}
def pf(name): return S[name].paragraph_format
for s in d.styles:
    try: set_font(s)
    except Exception: pass

for n in ["Normal", "Body Text", "First Paragraph", "Compact"]:
    if n in S:
        set_font(S[n], 10)
        p = pf(n); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.space_before = Pt(0); p.space_after = Pt(0); p.line_spacing = 1.0
        p.first_line_indent = Inches(0.14)
pf("Compact").first_line_indent = Inches(0); pf("Compact").alignment = WD_ALIGN_PARAGRAPH.LEFT
set_font(S["Title"], 22); pf("Title").alignment = WD_ALIGN_PARAGRAPH.CENTER
pf("Title").space_after = Pt(8); pf("Title").space_before = Pt(0)
set_font(S["Heading 1"], 10, bold=False, italic=False, smallcaps=True)
p = pf("Heading 1"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.space_before = Pt(8); p.space_after = Pt(4); p.keep_with_next = True
set_font(S["Heading 2"], 10, bold=False, italic=True)
p = pf("Heading 2"); p.alignment = WD_ALIGN_PARAGRAPH.LEFT; p.space_before = Pt(5); p.space_after = Pt(2); p.keep_with_next = True
for n in ["AuthorLine", "Affiliation", "AbstractPara", "IndexTerms", "Marker", "Reference", "TableNote", "Image Caption", "Table Caption", "Captioned Figure", "Figure"]:
    if n not in S: continue
set_font(S["AuthorLine"], 11); p = pf("AuthorLine"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.space_after = Pt(2); p.first_line_indent = Inches(0)
set_font(S["Affiliation"], 9, italic=True); p = pf("Affiliation"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.space_after = Pt(10); p.first_line_indent = Inches(0)
for n in ["AbstractPara", "IndexTerms"]:
    set_font(S[n], 9, bold=True); p = pf(n); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.first_line_indent = Inches(0.14); p.space_after = Pt(4)
pf("IndexTerms").space_after = Pt(10)
set_font(S["Marker"], 1); p = pf("Marker"); p.space_after = Pt(0); p.space_before = Pt(0); p.line_spacing = Pt(1)
set_font(S["Reference"], 8); p = pf("Reference"); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
p.left_indent = Inches(0.3); p.first_line_indent = Inches(-0.3); p.space_after = Pt(1)
set_font(S["TableNote"], 7.5); p = pf("TableNote"); p.alignment = WD_ALIGN_PARAGRAPH.LEFT; p.first_line_indent = Inches(0); p.space_after = Pt(6)
set_font(S["Image Caption"], 8, italic=False); p = pf("Image Caption"); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.space_before = Pt(2); p.space_after = Pt(8); p.first_line_indent = Inches(0)
set_font(S["Table Caption"], 8, italic=False, smallcaps=True); p = pf("Table Caption"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.space_before = Pt(8); p.space_after = Pt(3); p.first_line_indent = Inches(0); p.keep_with_next = True
for n in ["Captioned Figure", "Figure"]:
    if n in S:
        p = pf(n); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.first_line_indent = Inches(0); p.space_before = Pt(6); p.space_after = Pt(0); p.keep_with_next = True

body = d.element.body

def make_sectpr(cols, first=False):
    sp = OxmlElement("w:sectPr")
    if not first:
        t = OxmlElement("w:type"); t.set(W("val"), "continuous"); sp.append(t)
    pg = OxmlElement("w:pgSz"); pg.set(W("w"), "11906"); pg.set(W("h"), "16838"); sp.append(pg)
    mar = OxmlElement("w:pgMar")
    for k, v in dict(top="1077", right="907", bottom="1247", left="907", header="567", footer="567", gutter="0").items(): mar.set(W(k), v)
    sp.append(mar)
    c = OxmlElement("w:cols"); c.set(W("num"), str(cols)); c.set(W("space"), "284"); sp.append(c)
    return sp

# markers -> section breaks
state_cols = 1  # title block is one column
first = True
for par in list(d.paragraphs):
    txt = par.text.strip()
    if txt in ("[[TWO-COL]]", "[[WIDE-START]]", "[[WIDE-END]]"):
        for r in par.runs: r.text = ""
        # the section that ENDS here has the current column count
        sp = make_sectpr(state_cols, first=first); first = False
        par._p.get_or_add_pPr().append(sp)
        state_cols = 2 if txt in ("[[TWO-COL]]", "[[WIDE-END]]") else 1
# final body sectPr: two columns + footer page number
final = body.find(W("sectPr"))
for ch in list(final): final.remove(ch)
nf = make_sectpr(2)
for ch in list(nf): final.append(ch)

# page number footer
sec = d.sections[-1]
for s_ in d.sections:
    s_.footer.is_linked_to_previous = True
ft = d.sections[0].footer
fp = ft.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if tag:
        fc = OxmlElement("w:fldChar"); fc.set(W("fldCharType"), tag); run._r.append(fc)
    else:
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)
run.font.size = Pt(8); run.font.name = "Times New Roman"

# table captions: "TABLE I. Title" -> "TABLE I" <br> "Title"
for par in d.paragraphs:
    if par.style.name == "Table Caption":
        t = par.text
        m = re.match(r"(TABLE [IVX]+)\.\s*(.*)", t)
        if m:
            for r in par.runs[1:]: r._r.getparent().remove(r._r)
            r0 = par.runs[0]; r0.text = m.group(1)
            r0.add_break(WD_BREAK.LINE); r1 = par.add_run(m.group(2)); r1.font.small_caps = True

# figure captions: bold nothing, keep "Fig. n." normal
# tables: IEEE rules, 8pt, compact
def border(tag, sz="6"):
    b = OxmlElement("w:" + tag); b.set(W("val"), "single"); b.set(W("sz"), sz); b.set(W("space"), "0"); b.set(W("color"), "000000"); return b
for tbl in d.tables:
    tpr = tbl._tbl.tblPr
    for old in tpr.findall(W("tblBorders")): tpr.remove(old)
    bs = OxmlElement("w:tblBorders")
    bs.append(border("top", "8")); bs.append(border("bottom", "8"))
    for tag in ("left", "right", "insideV", "insideH"):
        e = OxmlElement("w:" + tag); e.set(W("val"), "nil"); bs.append(e)
    tpr.append(bs)
    for old in tpr.findall(W("tblStyle")): tpr.remove(old)
    w = tpr.find(W("tblW"))
    if w is None: w = OxmlElement("w:tblW"); tpr.append(w)
    w.set(W("type"), "pct"); w.set(W("w"), "5000")
    tpr_j = tpr.find(W("jc"))
    if tpr_j is None: tpr_j = OxmlElement("w:jc"); tpr.append(tpr_j)
    tpr_j.set(W("val"), "center")
    cm = OxmlElement("w:tblCellMar")
    for side, v in (("left", "50"), ("right", "50"), ("top", "10"), ("bottom", "10")):
        e = OxmlElement("w:" + side); e.set(W("w"), v); e.set(W("type"), "dxa"); cm.append(e)
    tpr.append(cm)
    for ri, row in enumerate(tbl.rows):
        for cell in row.cells:
            if ri == 0:
                tcpr = cell._tc.get_or_add_tcPr()
                tb = OxmlElement("w:tcBorders"); tb.append(border("bottom", "4")); tcpr.append(tb)
            for par in cell.paragraphs:
                par.paragraph_format.first_line_indent = Inches(0)
                par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                par.paragraph_format.space_after = Pt(0)
                if ri < len(tbl.rows) - 1: par.paragraph_format.keep_with_next = True
                for r in par.runs:
                    r.font.size = Pt(7.5); r.font.name = "Times New Roman"
                    if ri == 0: r.font.bold = True

WIDTHS = [
 [0.26,0.27,0.37,0.10],
 [0.30,0.70],
 [0.13,0.14,0.12,0.14,0.12,0.22,0.13],
 [0.205,0.03,0.095,0.09,0.06,0.055,0.05,0.05,0.05,0.06,0.06,0.06,0.06],
 [0.20,0.105,0.105,0.084,0.084,0.084,0.084,0.084,0.084,0.086],
 [0.16,0.12,0.12,0.12,0.12,0.12,0.12,0.12],
]
WIDE = {2,3,4,5}
for ti, tbl in enumerate(d.tables):
    fr = WIDTHS[ti]; assert len(fr) == len(tbl.columns), (ti, len(tbl.columns))
    total = 10092 if ti in WIDE else 4904
    tpr = tbl._tbl.tblPr
    w = tpr.find(W("tblW")); w.set(W("type"), "dxa"); w.set(W("w"), str(total))
    lay = OxmlElement("w:tblLayout"); lay.set(W("type"), "fixed"); tpr.append(lay)
    grid = tbl._tbl.find(W("tblGrid"))
    for gc, f in zip(grid.findall(W("gridCol")), fr): gc.set(W("w"), str(int(total*f)))
    for row in tbl.rows:
        for cell, f in zip(row.cells, fr):
            tcpr = cell._tc.get_or_add_tcPr()
            tw = tcpr.find(W("tcW"))
            if tw is None: tw = OxmlElement("w:tcW"); tcpr.insert(0, tw)
            tw.set(W("type"), "dxa"); tw.set(W("w"), str(int(total*f)))

d.core_properties.title = "Evidence-Aware Pre-Payment Risk Assessment for UPI Transactions Using Behavioural, Contextual, and Payer–Payee Graph Evidence"
d.core_properties.author = "C. Murali Naga Raju et al."
d.save("paper.docx")
print("ok")
