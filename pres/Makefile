DOCNAME = steam_ana-epland

TEXENG = lualatex
TEXFLAGS = --file-line-error

# BIBENG = bibtex
BIBENG =
BIBFLAGS =

.PHONY: all clean cleanpdf cleanall info open help

all:
	$(info $(TEXENG) $(TEXFLAGS) $(DOCNAME))
	$(TEXENG) $(TEXFLAGS) $(DOCNAME)
	# $(BIBENG) $(BIBFLAGS) $(DOCNAME)
	# $(TEXENG) $(TEXFLAGS) $(DOCNAME)
	$(TEXENG) $(TEXFLAGS) $(DOCNAME)

clean:
	-@rm -f comment.cut
	-@rm -f *.log
	-@rm -f *.aux
	-@rm -f *.vrb
	-@rm -f $(DOCNAME).bbl
	-@rm -f $(DOCNAME).toc
	-@rm -f $(DOCNAME).out
	-@rm -f $(DOCNAME).blg
	-@rm -f $(DOCNAME).nav
	-@rm -f $(DOCNAME).snm

cleanpdf:
	-@rm -f $(DOCNAME).pdf

cleanall: clean cleanpdf

info:
	$(info DOCNAME = $(DOCNAME))
	$(info TEXENG = $(TEXENG))
	$(info TEXFLAGS = $(TEXFLAGS))
	$(info BIBENG = $(BIBENG))
	$(info BIBFLAGS = $(BIBFLAGS))

open:
	-@evince $(DOCNAME).pdf </dev/null &>/dev/null &

help:
	@echo ""
	@echo "make          to make output PDF"
	@echo "make clean    to clean auxiliary files (not output PDF)"
	@echo "make cleanpdf to clean output PDF file"
	@echo "make cleanall to clean all files"
	@echo "make open     to open output PDF"
	@echo "make info     to view Makefile settings"
	@echo ""

