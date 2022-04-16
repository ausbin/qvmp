.PHONY: pdf publish clean

pdf:
	latexmk -pdf -outdir=build thesis.tex

publish: pdf
	cp build/thesis.pdf .

clean:
	rm -rf build
