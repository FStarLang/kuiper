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
