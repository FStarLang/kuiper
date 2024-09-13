include .common.mk

.PHONY: all
all: verify-all

.PHONY: verify-all
verify-all:
	+$(MAKE) -f verify.mk all

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

.SUFFIXES:

.PHONY: watch
watch:
	while true; do \
		$(MAKE) ;\
		inotifywait -qre close_write .; \
	done
