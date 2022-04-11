.PHONY: pdf watch clean

OUTDIR = "build"
# COMMON_OPTIONS = -pdflatex="lualatex -interaction nonstopmode" -pdf -outdir=$(OUTDIR)
COMMON_OPTIONS = --output-directory=$(OUTDIR)
FILES = presentation.tex
OUTFILES = $(foreach file, $(FILES), $(OUTDIR)/$(file:.tex=.pdf))

pdf:
	pdflatex $(COMMON_OPTIONS) $(FILES)

publish: pdf
	cp $(OUTFILES) .

clean:
	rm -rf build
