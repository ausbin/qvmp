.PHONY: main clean FORCE

main: poster.pdf

poster.pdf: FORCE
	latexmk -outdir=build -pdflatex='lualatex -interaction nonstopmode' -pdf poster.tex

clean:
	rm -rf build
