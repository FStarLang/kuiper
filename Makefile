include .common.mk

.PHONY: all
all:
	+$(MAKE) -f verify.mk all

.PHONY: prepare
prepare:
	+$(MAKE) -f verify.mk prepare

.PHONY: test
test:
	+$(MAKE) -f verify.mk test

.PHONY: echo-fstar
echo-fstar:
	+$(MAKE) -f verify.mk $@
.PHONY: echo-krml
echo-krml:
	+$(MAKE) -f verify.mk $@

.PHONY: ci
ci:
	+$(MAKE) -f verify.mk all test

.PHONY: depgraph
depgraph:
	+$(MAKE) -f verify.mk depgraph

.SUFFIXES:

.PHONY: watch
watch:
	while true; do \
		$(MAKE) ;\
		inotifywait -qre close_write .; \
	done

clean:
	rm -rf obj/
	rm -f .*.touch

clean-full: clean
	git submodule foreach git clean -dXf

dist: all
	rm -f dist/*
	cp obj/*.cu dist

.PHONY: lint
lint:
	./FStar/.scripts/remove_all_unused_opens.sh extraction
	./FStar/.scripts/remove_all_unused_opens.sh src
	( cd src && git sed 's/ *$$//' )
	( cd extraction && git sed 's/ *$$//' )
	( cd src && ../scripts/find-pulse-noix.sh )
	( cd src && ../scripts/check-attrs.sh )
	indent -linux test/*.cu && rm -f test/*.cu~

.PHONY: list-admits
list-admits:
	-git grep -w 'assume_\|assume\|admit' src

.PHONY: wc
wc:
	echo F*:
	find src/ \( -name '*.fst' -o -name '*.fsti' \) -exec cat {} \+ | grep '[^ ]' | wc -l
	echo CUDA:
	find dist/ -name '*.cu' -exec cat {} \+ | grep '[^ ]' | wc -l
