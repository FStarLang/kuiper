include .common.mk

.SUFFIXES:

default: all

.PHONY: watch
watch:
	while true; do \
		$(MAKE) ;\
		inotifywait -qre close_write .; \
	done

clean:
	rm -f .plugin.touch
	rm -rf obj/

clean-modules:
	git submodule foreach git clean -dXf

clean-full: clean clean-modules
	rm -f .*.touch

dist: extraction-targets
	rm -f dist/*
	cp obj/*.cu obj/*.h dist

.PHONY: lint-c
lint-c:
	indent -linux -nut -i4 test/*.cu test/*.c.inc && rm -f test/*.cu~ test/*.c.inc~

.PHONY: lint-fstar
lint-fstar:
	./FStar/.scripts/remove_all_unused_opens.sh extraction
	./FStar/.scripts/remove_all_unused_opens.sh src
	( cd src && git sed 's/[[:space:]]*$$//' )
	( cd extraction && git sed 's/[[:space:]]*$$//' )
	( cd src && ../scripts/find-pulse-noix.sh )
	( cd src && ../scripts/check-attrs.sh )

.PHONY: lint
lint: lint-c lint-fstar

.PHONY: list-admits
list-admits:
	-git grep -w 'assume_\|assume\|admit\|magic' src

.PHONY: wc
wc:
	echo F*:
	find src/ \( -name '*.fst' -o -name '*.fsti' \) -exec cat {} \+ | grep '[^ ]' | wc -l
	echo CUDA:
	find dist/ -name '*.cu' -exec cat {} \+ | grep '[^ ]' | wc -l

.PHONY: ci
ci:
	+$(MAKE) -f verify.mk all test

# Everything else goes to verify.mk
%:
	+$(MAKE) -f verify.mk $@
