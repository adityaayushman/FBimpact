IEEE manuscript: "Anticipating Falls from Skeletal Pose"
=========================================================

Contents: 12 pages in IEEE two-column format; 7 figures (processing chain, network architecture,
evaluation workflow, and four results graphs); 11 tables; 4 algorithms; 8 equations; 19 references.

Files
  fbimpact_ieee.tex            the manuscript, IEEEtran class (this is the file to submit / edit)
  figures/*.pdf, *.png         the figures (PDF for LaTeX, PNG for the preview)
  fbimpact_ieee.pdf            an IEEE-style preview of the same content (rendered from fbimpact_ieee.html)
  fbimpact_ieee_overleaf.zip   .tex + figures, ready to upload to Overleaf
  fbimpact_ieee.docx           Word version: IEEE page geometry, two columns, Times New Roman 10 pt, with the same
                               text, tables, algorithms, equations and figures (equations are typed text, not
                               Equation Editor objects)

Compile
  Overleaf: New project > Upload project > fbimpact_ieee_overleaf.zip; main file fbimpact_ieee.tex;
  compiler pdfLaTeX; compile twice so the cross-references resolve.
  The .tex was checked statically (python paper/check_tex.py) but NOT compiled on the machine that
  produced it, because no LaTeX installation was available. Compile it once and fix anything it reports.
  It uses the standard packages cite, amsmath, amssymb, graphicx, array, algorithm, algorithmic, textcomp, hyperref.

Before submitting
  - the author line contains only the name; most journals also want affiliation and e-mail, which are
    usually entered in the submission system (add them to \author{} if the journal asks for them in the file)
  - the Acknowledgment holds the generative-AI disclosure that IEEE requires; keep it, and make sure it
    states exactly what was done
  - confirm the pose estimator named in Section III-C (Ultralytics YOLO11n-pose is the extraction default)
  - confirm the volume number of reference [7] (Kwolek and Kepski): two sources disagree (117 or 118)
  - choose IEEE index terms from the IEEE Taxonomy if the target journal asks for them

Regenerate
  python paper/make_figures.py     figures from the run data
  python paper/build_paper.py      .tex and .html
  python paper/check_tex.py        static checks on the .tex
  python paper/build_docx.py       the Word version
  python scripts/verify_results.py recompute every reported number from the run folders
  python scripts/urfall_paired.py  the UR Fall paired bootstrap intervals (Table VI)
