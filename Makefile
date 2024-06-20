PULSE_REPO := https://github.com/FStarLang/pulse
PULSE_HASH := $(shell cat .pulse.hash)

ROOTS := $(shell find src/ -name '*.fst' -o -name '*.fsti')

CACHEDIR := .cache
OUTDIR   := .out

FSTAR_FLAGS += --cache_checked_modules
FSTAR_FLAGS += --cache_dir $(CACHEDIR)
FSTAR_FLAGS += --odir $(OUTDIR)
FSTAR_FLAGS += $(OTHERFLAGS)

FSTAR := fstar.exe					\
	--include pulse/lib/pulse/			\
	--include pulse/lib/pulse/core/			\
	--include pulse/lib/pulse/lib/			\
	--include pulse/lib/pulse/lib/class/		\
	--include src/lib/				\
	--include src/examples/				\
	--load_cmxs pulse				\
	--warn_error -249-321				\
	$(FSTAR_FLAGS)

# This sandwich is needed so all is the first rule (and not
# something in the include), and verify-all can refer to ALL_CHECKED_FILES,
# which is empty before including .depend. Sigh.
# We also need to allow calling make update-pulse without requiring
# .depend,
all: verify-all
ifeq ($(MAKECMDGOALS),update-pulse)
else
include .depend
endif
verify-all: $(ALL_CHECKED_FILES)

# Dependencies come from .depend. We still need this rule.
%.checked:
	$(FSTAR) $<
	@touch -c $@

.PHONY: echo-fstar
echo-fstar:
	@echo $(FSTAR)

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
pull-pulse:
	git -C pulse pull
	$(MAKE) save-pulse

.depend: $(ROOTS) pulse
	$(FSTAR) --dep full $(ROOTS) --output_deps_to $@
