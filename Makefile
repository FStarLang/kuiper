include .common.mk

.PHONY: all
all:
	+$(MAKE) -f verify.mk all

.PHONY: minimal
minimal:
	+$(MAKE) -f verify.mk minimal

.PHONY: prepare
prepare:
	+$(MAKE) -f verify.mk prepare

.PHONY: test
test:
	+$(MAKE) -f verify.mk test

.PHONY: accept
accept:
	+$(MAKE) -f verify.mk accept

.PHONY: extract-all
extract-all:
	+$(MAKE) -f verify.mk extract-all

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
	rm -f .plugin.touch
	rm -rf obj/

clean-modules:
	git submodule foreach git clean -dXf

clean-full: clean clean-modules
	rm -f .*.touch

dist: extract-all
	rm -f dist/*
	cp obj/*.cu obj/*.h dist

.PHONY: lint-c
lint-c:
	indent -linux -nut -i4 test/*.cu test/*.c.inc && rm -f test/*.cu~ test/*.c.inc~

.PHONY: lint-fstar
lint-fstar:
	./FStar/.scripts/remove_all_unused_opens.sh extraction
	./FStar/.scripts/remove_all_unused_opens.sh src
	( cd src && ../scripts/git-sed 's/[[:space:]]*$$//' )
	( cd extraction && ../scripts/git-sed 's/[[:space:]]*$$//' )
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

.PHONY: bench-package
bench-package: kuiper-bench.tgz

kuiper-bench.tgz: all
	rm -rf kuiper-bench
	mkdir -p kuiper-bench
	mkdir -p kuiper-bench/obj
	cp -r obj/*.cu obj/*.h kuiper-bench/obj
	cp -r include kuiper-bench/include
	cp -r test kuiper-bench/test
	cp -r scripts kuiper-bench/scripts
	cp -r configure kuiper-bench/configure
	cp -r bench-package.mk kuiper-bench/Makefile
	cp -r nvcc.mk kuiper-bench/nvcc.mk
	cp -r .common.mk kuiper-bench/.common.mk
	cp -r .configure.mk kuiper-bench/.configure.mk
	cp -r bench kuiper-bench/bench
	rm -f kuiper-bench/bench/*.o   # clean built files
	rm -f kuiper-bench/bench/bench # clean built files
	tar czf kuiper-bench.tgz ./kuiper-bench

.PHONY: test-bench-package
test-bench-package: bench-package
	rm -rf _tmp
	mkdir _tmp
	cd _tmp && tar xzf ../kuiper-bench.tgz
	$(MAKE) -C _tmp
	rm -rf _tmp
