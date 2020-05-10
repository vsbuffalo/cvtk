diff.pdf: diff.tex manuscript.tex
	latexmk -xelatex -pv -f diff.tex

diff.tex:
	rm -f old.tex
	git show master:manuscript/manuscript.tex > old.tex
	latexdiff --graphics-markup=off old.tex manuscript.tex > diff.tex



