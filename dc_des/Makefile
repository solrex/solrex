ARTICLE=dc_des_lr
ARTICLE_SRC=$(ARTICLE).tex des.c

SRC=des.c
BIN=des
CC=gcc

all: article $(BIN)

article: $(ARTICLE).pdf

$(ARTICLE).pdf: $(ARTICLE_SRC)
	latex $(ARTICLE).tex
	latex $(ARTICLE).tex
	dvipdfmx $(ARTICLE).dvi

$(BIN): $(SRC)
	$(CC) -Wpointer-sign -o $@ $<

clean:
	rm -f *.aux *.toc *.lon *.lor *.lof *.ilg *.idx *.ind *.out *.log *.exa
	rm -f *.nav *.snm *.bbl *.blg

distclean: clean
	rm -f *.dvi *~ *.exe
