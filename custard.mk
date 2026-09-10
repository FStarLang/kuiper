# Extraction: F* straight to CUDA via Custard.
#
# Must be included after .common.mk, .configure.output and nvcc.mk, since it
# uses $(FSTAR), $(PLUGIN), $(OUTDIR) and $(EXTRACT).
#
# Custard is a backend of F* itself, so there is no intermediate IR file and no
# separate compiler run: one fstar.exe invocation per entry module produces the
# .cu and its .h together.  The extraction rules that give Kuiper's primitives
# their CUDA spelling (KPR_KCALL, KPR_SHMEM_AT, wmma::, the memcpy and
# allocation intrinsics, ...) live in extraction/KuiperCustard.fst.

CUSTARD_FLAGS += --codegen Custard
CUSTARD_FLAGS += --custard_backend C
CUSTARD_FLAGS += --custard_monomorphize_types true
CUSTARD_FLAGS += --custard_norm_budget 200000000
# Kuiper assumes FStar.SizeT.fits_u32 (Kuiper.SizeT.fsti, SizeTFitsU32), which
# is what licenses narrowing every index to 32 bits.  This is a performance
# choice, not a correctness-neutral one: it is what karamel did, and without it
# register pressure across the extracted kernels rises by about 4%.
CUSTARD_FLAGS += --custard_sizet_width 32
CUSTARD_FLAGS += --load_cmxs $(PLUGIN)
# Every dependency is already checked by the verify step; extraction must not
# re-check anything, or a single module's extraction pulls in the whole tree.
CUSTARD_FLAGS += --already_cached '*'
CUSTARD_FLAGS += $(CUSTARD_OTHERFLAGS)

# One rule per module rather than a pattern rule: Custard extracts a whole
# module in a single invocation named by --custard_entry_module, and both the
# module name and the source path come from the same entry of $(EXTRACT).  A
# pattern rule would have to search for the source, which is what the karamel
# pipeline used .depend for, and .depend has no Custard targets.
#
# Output lands in $(OUTDIR)/pre/; verify.mk formats it into $(OUTDIR)/.
define custard_rule
$(OUTDIR)/pre/$(subst .,_,$(basename $(notdir $1))).cu \
$(OUTDIR)/pre/$(subst .,_,$(basename $(notdir $1))).h &: $1 .plugin.touch | .fstar.touch
	$$(call msg,"CUSTARD")
	@mkdir -p $(OUTDIR)/pre
	$$(Q)$$(FSTAR) $$(CUSTARD_FLAGS) --odir $(OUTDIR)/pre			\
		--custard_entry_module $(basename $(notdir $1))			\
		-o $(OUTDIR)/pre/$(subst .,_,$(basename $(notdir $1))).cu	\
		$1
endef
$(foreach f,$(EXTRACT),$(eval $(call custard_rule,$f)))

# A missing extraction rule degrades silently: the primitive comes out as an
# admit, which compiles and links and then aborts at run time (or, worse, runs
# a host-side loop over device pointers).  Fail the build instead.
.PHONY: check-no-admits
check-no-admits: extract-all
	$(Q)if grep -l Prims_admit $(EXTRACTED_CU) $(EXTRACTED_H); then		\
		echo "*** extraction produced admit stubs: a primitive is missing an extraction rule"; \
		exit 1;								\
	fi
