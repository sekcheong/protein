TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
JGRAPH = jgraph

.PHONY: all view

all: report.pdf

view:
	open report.pdf

report.pdf: report.tex hiddenunits.eps learningrate.eps momentum.eps weightdecay.eps
	$(TEX) report.tex

hiddenunits.eps: hiddenunits.jgr
	$(JGRAPH) hiddenunits.jgr > hiddenunits.eps 

learningrate.eps: learningrate.jgr
	$(JGRAPH) learningrate.jgr > learningrate.eps 

momentum.eps: momentum.jgr
	$(JGRAPH) momentum.jgr > momentum.eps 

weightdecay.eps: weightdecay.jgr
	$(JGRAPH) weightdecay.jgr > weightdecay.eps 

clean: 
	rm *.log *.aux *.out *.eps *-eps-converted*.pdf
