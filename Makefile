include .common.mk

PULSE_REPO := https://github.com/FStarLang/pulse
PULSE_HASH := $(shell cat .pulse.hash)

.PHONY: all
all: verify-all

.PHONY: verify-all
verify-all: pulse
	$(MAKE) -f Makefile.verify $@

.PHONY: echo-fstar
echo-fstar: pulse
	$(MAKE) -f Makefile.verify $@

pulse:
	$(error pulse directory not found: Run `make update-pulse` to fetch and compile Pulse)

.PHONY: update-pulse
update-pulse:
	./scripts/update-pulse.sh "${PULSE_REPO}" "${PULSE_HASH}"
	@# All we do is build the ocaml plugin. We check the library
	@# files incrementally, on demand.
	$(MAKE) -C pulse/src/ build-ocaml

.PHONY: save-pulse
save-pulse:
	git -C pulse rev-parse HEAD >.pulse.hash

.PHONY: pull-pulse
pull-pulse: pulse
	git -C pulse pull
	$(MAKE) save-pulse

.PHONY: ci
ci:
	$(MAKE) update-pulse
	$(MAKE) all
