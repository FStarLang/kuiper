include .common.mk

.PHONY: all
all:
	+$(MAKE) -f verify.mk all

.PHONY: test
test:
	+$(MAKE) -f verify.mk test

.PHONY: reboot
reboot:
	FSTAR_HOME=$(CURDIR)/FStar && cd FStar/ && $(MAKE) 1 && $(MAKE) bootstrap ADMIT=1
	FSTAR_HOME=$(CURDIR)/FStar && cd pulse/ && $(MAKE) boot-checker OTHERFLAGS='--admit_smt_queries true'


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
