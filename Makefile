include .common.mk

PULSE_REPO := https://github.com/FStarLang/pulse
PULSE_HASH := $(shell cat .pulse.hash)

.PHONY: all
all: verify-all

.PHONY: verify-all
verify-all: pulse
	+$(MAKE) -f verify.mk all

.PHONY: test
test: pulse
	+$(MAKE) -f verify.mk test

.PHONY: echo-fstar
echo-fstar: pulse
	+$(MAKE) -f verify.mk $@
.PHONY: echo-krml
echo-krml: pulse
	+$(MAKE) -f verify.mk $@

pulse:
	$(error pulse directory not found: Run `make update-pulse` to fetch and compile Pulse)

.PHONY: update-pulse
update-pulse:
	./scripts/update-pulse.sh "${PULSE_REPO}" "${PULSE_HASH}"
	@# All we do is build the ocaml plugin. We check the library
	@# files incrementally, on demand.
	@echo "::group:build plugin"
	+$(MAKE) -C pulse/src/ build-ocaml
	@echo "::endgroup"

.PHONY: save-pulse
save-pulse:
	git -C pulse rev-parse HEAD >.pulse.hash

.PHONY: pull-pulse
pull-pulse: pulse
	git -C pulse pull
	+$(MAKE) save-pulse

.PHONY: ci
ci:
	+$(MAKE) update-pulse
	+$(MAKE) -f verify.mk all test

.SUFFIXES:
