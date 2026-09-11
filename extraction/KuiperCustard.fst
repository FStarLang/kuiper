module KuiperCustard

(* A Custard rule for Kuiper's kernel launcher.

   [Kuiper.Kernel.Base.launch_kernel_full] takes a [kernel_desc], which is
   compile-time input to code generation: it stores a list of
   [Kuiper.SHMem.shmem_desc], whose constructor stores a [Type0] that a later
   field's type mentions, so it has no C layout (error 368).  The rule reads
   the descriptor and emits a launch, after which the descriptor is dead. *)

open FStarC
open FStarC.Effect
open FStarC.List
open FStarC.Class.Show
open FStarC.Const
open FStarC.Custard.Syntax

module B      = FStarC.Custard.Builtins
module Ident  = FStarC.Ident
module BU     = FStarC.Util

(* Every kernel is lifted, and [lift_named] is verbatim, so two kernels in
   one unit are one symbol (error 378).  A rule cannot see the definition it
   is expanding inside, so the only thing here that distinguishes two lifts
   is the order they happen in. *)
let kernel_seq : ref int = mk_ref 0

let fresh_kernel_name () : ML string =
  let n = !kernel_seq in
  kernel_seq := n + 1;
  "kuiper_kernel_" ^ show n

let tag (e:expr) : string =
  match e.e with
  | EConst _   -> "a constant"       | EVar _     -> "a local variable"
  | EQual _    -> "a top-level ref"  | ELet _     -> "a let"
  | EApp _     -> "an application"   | EFun _     -> "a lambda"
  | EMatch _   -> "a match"          | EIf _      -> "an if"
  | ESeq _     -> "a sequence"       | ECtor _    -> "a constructor application"
  | ETuple _   -> "a tuple"          | ERecord _  -> "a record literal"
  | EProj _    -> "a projection"     | EDiscrim _ -> "a discriminator"
  | ECoerce _  -> "a coercion"       | ECast _    -> "a cast"
  | EAny       -> "an arbitrary value" | EAbort _ -> "an abort"
  | EOp _      -> "a primitive operation" | EWhile _ -> "a while"
  | ERaise _   -> "a raise"          | ETry _     -> "a try"

let die (#a:Type) (want:string) (e:expr) : ML a =
  failwith ("KuiperCustard: expected " ^ want ^ ", got " ^ tag e)

let field_at (i:int) (fname:string) (e:expr) : ML expr =
  match e.e with
  | ECtor (_, args) ->
    if i < List.length args then List.nth args i else die "a wider constructor" e
  | ERecord (_, fs) ->
    (match BU.try_find (fun (f, _) -> f = fname) fs with
     | Some (_, v) -> v
     | None -> die ("a record with a field " ^ fname) e)
  | _ -> die "a constructor application or record literal" e

let rec elements (e:expr) : ML (list expr) =
  match e.e with
  | ECtor (n, args) ->
    if n.id = "Nil" then []
    else if n.id = "Cons" then
      (match List.rev args with
       | tl :: hd :: _ -> hd :: elements tl
       | _ -> die "a two-argument Cons" e)
    else die "Prims.Nil or Prims.Cons" e
  | _ -> die "a list literal" e

let die_ty (#a:Type) (want:string) (t:cty) : ML a =
  failwith ("KuiperCustard: expected " ^ want ^ ", got " ^ show t)

let shmem_at_lid = "Kuiper.Example.ARPort.kpr_shmem_at"

(* One shared-memory request: [SHArray (Mksized (size, zero)) len].  The
   element *type* was erased from [SHArray], but the [sized] instance carries
   a zero of that type, so it comes back off [zero.ty]. *)
let parse_shmem (e:expr) : ML (cty & expr & expr) =
  let sized = field_at 0 "sized" e in
  let len   = field_at 1 "len"   e in
  let sz    = field_at 0 "size"  sized in
  let zero  = field_at 1 "zero"  sized in
  (zero.ty, sz, len)

let usize : cty = TInt (Unsigned, Sizet)
let uop (o:op) (a b : expr) : expr =
  mk (EOp ({ po_op = o; po_ty = Some (PInt (Unsigned, Sizet)) }, [a; b]))
     usize E_Pure
let uconst (n:int) : expr =
  mk (EConst (CInt (n, Dec, Some (Unsigned, Sizet)))) usize E_Pure

(* [KPR_SHMEM_AT(off)] is the base of the block's dynamic shared memory plus
   [off] bytes; the cast to the element type is the [ECoerce]. *)
let shmem_at (off:expr) (elt:cty) : expr =
  let f = mk (EQual ({ ns = ["Kuiper"; "Example"; "ARPort"];
                       id = "kpr_shmem_at"; spec = None }, []))
             (TArrow (usize, E_Impure, TBuf (TInt (Unsigned, Int8)))) E_Pure in
  let call = mk (EApp (f, [off])) (TBuf (TInt (Unsigned, Int8))) E_Impure in
  mk (ECoerce (call, TBuf elt)) (TBuf elt) E_Impure

(* [c_shmems ds] is a type-level recursion over the descriptor list: one nested
   pair per request, ending in [unit].  The witness is a value of that shape,
   one [KPR_SHMEM_AT] per request at consecutive byte offsets.

   [exp_ty] is the binder's *own* type, as Custard resolved [c_shmems ds].
   Reuse it rather than rebuilding it: each nested [tuple2] instance is
   declared only because the source mentions it, so a freshly built [TTuple]
   would match no declaration (error 368). *)
let rec build_shmems (off:expr) (exp_ty:cty) (ds : list (cty & expr & expr))
      : ML expr =
  match ds with
  | [] -> mk (EConst CUnit) exp_ty E_Pure
  | (elt, sz, len) :: ds' ->
    let here = shmem_at off elt in
    let off' = uop Add off (uop Mult sz len) in
    let rest_ty =
      match exp_ty with
      | TApp (_, [_; r]) -> r
      | TTuple [_; r] -> r
      | _ -> die_ty "a two-argument tuple type" exp_ty in
    let rest = build_shmems off' rest_ty ds' in
    let mktup2 = { ns = ["FStar"; "Pervasives"; "Native"];
                   id = "Mktuple2"; spec = None } in
    mk (ECtor (mktup2, [here; rest])) exp_ty E_Impure

let rec pvars (p:pat) : ML (list string) =
  match p with
  | PVar x -> [x]
  | PCtor (_, ps) | PTuple ps | POr ps -> List.collect pvars ps
  | PRecord (_, fs) -> List.collect (fun (_, q) -> pvars q) fs
  | _ -> []

let rec fvs (bound : list string) (e : expr) : ML (list (string & cty)) =
  match e.e with
  | EVar x -> if List.mem x bound then [] else [(x, e.ty)]
  | EApp (h, args) -> fvs bound h @ List.collect (fvs bound) args
  | EOp (_, args) | ECtor (_, args) | ETuple args -> List.collect (fvs bound) args
  | EFun (bs, b) -> fvs (List.map (fun (b:binder) -> b.b_name) bs @ bound) b
  | ELet (x, _, d, b) -> fvs bound d @ fvs (x :: bound) b
  | EIf (c, a, b) -> fvs bound c @ fvs bound a @ fvs bound b
  | ESeq (a, b) | EWhile (a, b) -> fvs bound a @ fvs bound b
  | ECoerce (x, _) | ECast (x, _) | EProj (x, _, _)
  | EDiscrim (x, _) | ERaise x -> fvs bound x
  | ERecord (_, fs) -> List.collect (fun (_, v) -> fvs bound v) fs
  | EMatch (sc, brs) ->
    fvs bound sc @
    List.collect (fun ((p, g, b) : branch) ->
      let bound = pvars p @ bound in
      (match g with Some ge -> fvs bound ge | None -> []) @ fvs bound b) brs
  | _ -> []

let port_ns = ["Kuiper"; "Example"; "ARPort"]
let kcall_lid    = "Kuiper.Example.ARPort.kcall"
let blockidx_lid = "Kuiper.Example.ARPort.kpr_blockidx"
let threadidx_lid = "Kuiper.Example.ARPort.kpr_threadidx"

(* [blockIdx.x] and [threadIdx.x] are read from CUDA inside the kernel, so the
   body's index binders are not launch parameters: bind them at the top of the
   lifted body instead. *)
let cuda_index (id:string) (t:cty) : expr =
  mk (EQual ({ ns = port_ns; id = id; spec = None }, [])) t E_Pure

(* [launch_kernel_full] arrives as [desc; stream; _; _; _], the descriptor
   fully reduced to [Mkkernel_desc(nblk, nthr, shmems, f)]. *)
let launch (tys : list cty) (args : list expr) : ML expr =
  match args with
  | d :: stream :: _ ->
    let nblk   = field_at 0 "nblk"   d in
    let nthr   = field_at 1 "nthr"   d in
    let shmems = field_at 2 "shmems" d in
    let body   = field_at 3 "f"      d in
    let nshmem = List.length (elements shmems) in
    let bs, inner =
      match body.e with
      | EFun (bs, inner) -> bs, inner
      | _ -> die "the kernel body as a lambda" body in
    (* The body's first binder is the shared-memory handle.  It comes from the
       [shmem_desc] existential, so Custard has lost its representation (TAny,
       error 368); give it the base pointer's C type. *)
    let bs = List.map (fun (b:binder) ->
               match b.b_ty with
               | TAny -> { b with b_ty = TBuf (TInt (Unsigned, Int8)) }
               | _ -> b) bs in
    let bound = List.map (fun (b:binder) -> b.b_name) bs in
    let caps = BU.remove_dups (fun (a, _) (b, _) -> a = b) (fvs bound inner) in
    let cap_bs = List.map (fun (v, t) -> { b_name = v; b_ty = t }) caps in
    (* The body's own binders are [_sh], [bid], [tid] and a unit.  None is a
       launch parameter: shared memory is a pointer the launch sizes, and the
       indices come from CUDA.  Bind the two indices at the top of the body and
       drop all four from the signature. *)
    (* Bind the two indices unconditionally.  Whether the body mentions one is
       not worth a free-variable scan: an unused local costs nothing, and
       missing one is an unbound variable at the C backend. *)
    let bind_idx (i:int) (id:string) (body:expr) : ML expr =
      if List.length bs <= i then body
      else
        let b : binder = List.nth bs i in
        mk (ELet (b.b_name, b.b_ty, cuda_index id b.b_ty, body))
           body.ty body.eff in
    (* The shared-memory handle is the body's first binder -- but only when
       there is one: with no requests its type is [c_shmems [] = unit], which
       Custard erases, and the indices move down one slot. *)
    (* The kernel body is [fun <sh> bid tid () -> ...].  The shared-memory
       handle binder is present whenever [c_shmems] is not erased -- which
       depends on [c_shmems [] = int] vs [= unit], not on whether this kernel
       requests anything -- so take the offset from the binder list rather than
       from the request count. *)
    let base = if List.length bs >= 4 then 1 else 0 in
    let inner = bind_idx (base + 1) "kpr_threadidx" inner in
    let inner = bind_idx base       "kpr_blockidx"  inner in
    (* Shared memory: the body destructures its handle as nested pairs, so the
       witness is a value of that shape, one [KPR_SHMEM_AT] per request at
       consecutive byte offsets.  [smem] is the tot_bytes the launch must reserve. *)
    let descs = List.map parse_shmem (elements shmems) in
    let inner =
      if nshmem = 0 then inner
      else
        let b : binder = List.nth bs 0 in
        let sh_val = build_shmems (uconst 0) b.b_ty descs in
        mk (ELet (b.b_name, b.b_ty, sh_val, inner)) inner.ty inner.eff in
    (* Bytes the launch must reserve: sum of [size * len] over the requests. *)
    let smem =
      List.fold_left (fun acc (_, sz, len) -> uop Add acc (uop Mult sz len))
                     (uconst 0) descs in
    let closed_bs = cap_bs in
    (* Custard refuses a binder-less [EFun] (49.4), and a capture-free kernel
       is [__global__ void k(void)], so keep a [unit] parameter. *)
    let closed_bs =
      if List.isEmpty closed_bs then [{ b_name = "uu___"; b_ty = TUnit }]
      else closed_bs in
    (* Custard collapses a binder-less [EFun] to a constant, which loses both
       the [Prologue] and the function-ness, so a capture-free kernel keeps a
       [unit] parameter: [__global__ void k(void)]. *)

    let closed_ty =
      List.fold_right (fun (b:binder) acc -> TArrow (b.b_ty, E_Impure, acc))
                      closed_bs TUnit in
    let closed = mk (EFun (closed_bs, inner)) closed_ty E_Impure in
    let kernel = B.lift_named (fresh_kernel_name ())
                   [Comment "hoisted by the Custard Kuiper rule";
                    Prologue "__global__";
                    (* Section 51.3.  Anything the kernel body reaches is
                       device code; a helper shared with host code needs
                       both qualifiers or nvcc rejects one of the two calls. *)
                    ClosurePrologue ("__device__", "__device__ __host__");
                    Private] closed in
    let cap_args = List.map (fun (v, t) -> mk (EVar v) t E_Pure) caps in
    (* [smem] is the tot_bytes byte size computed with the witness above. *)
    let kty = TArrow (closed_ty, E_Impure,
               (TArrow (nblk.ty, E_Impure, TArrow (nthr.ty, E_Impure,
                  TArrow (smem.ty, E_Impure,
                    TArrow (stream.ty, E_Impure, TUnit)))))) in
    let sy = mk (EQual ({ ns = ["Kuiper"; "Kernel"; "Base"];
                          id = "sync_stream"; spec = None }, []))
                (TArrow (stream.ty, E_Impure, TUnit)) E_Impure in
    (* One [external] per capture count: a C variadic macro, a fixed F* arity. *)
    let kid = if List.isEmpty caps then "kcall"
              else "kcall" ^ show (List.length caps) in
    let kty = List.fold_right (fun (_, t) acc -> TArrow (t, E_Impure, acc))
                              caps kty in
    let kc = mk (EQual ({ ns = port_ns; id = kid; spec = None },
                        List.map snd caps)) kty E_Impure in

    let _ = sy in
    let call = mk (EApp (kc, [kernel; nblk; nthr; smem; stream] @ cap_args))
                  TUnit E_Impure in
    (* Section 79.  The Krml plugin put two more statements in front of the
       launch (ExtractKuiper.fst:450-482): a check that the request fits the
       device, and the opt-in that raises the 48KiB dynamic shared memory cap.
       Without the second, a kernel over the cap does not launch at all.  Both
       are skipped when the kernel asks for no shared memory, which is the same
       test [shmem_is_nonzero] made. *)
    let asks_for_shmem =
      match smem.e with
      | EConst (CInt (0, _, _)) -> false
      | _ -> true in
    if not asks_for_shmem then call
    else
      let arport (id:string) (t:cty) : expr =
        mk (EQual ({ ns = port_ns; id = id; spec = None }, [])) t E_Pure in
      let fits = arport "kpr_shmem_fits" (TArrow (smem.ty, E_Impure, TUnit)) in
      let setm = arport "kpr_set_max_dyn_shmem"
                   (TArrow (closed_ty, E_Impure,
                     TArrow (smem.ty, E_Impure, TUnit))) in
      let seq (a:expr) (b:expr) : expr = mk (ESeq (a, b)) b.ty b.eff in
      seq (mk (EApp (fits, [smem])) TUnit E_Impure)
          (seq (mk (EApp (setm, [kernel; smem])) TUnit E_Impure) call)
  | _ -> failwith "KuiperCustard: launch_kernel_full with too few arguments"

(* [Kuiper.Array.Core.slice_read] and [slice_write] are [@@noextract_to "krml"]
   and the Krml plugin supplies their meaning (ExtractKuiper.fst:985-988).  The
   Custard equivalent: a buffer operation, not a call.  This also keeps them
   out of the kernel, where a plain C function would be [__host__]. *)
let elem_ty (t:cty) : ML cty =
  match t with
  | TBuf e -> e
  | _ -> die "a buffer" (mk EAny t E_Pure)

let slice_read (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [r; idx] ->
    mk (EOp ({ po_op = BufRead; po_ty = None }, [r; idx]))
       (elem_ty r.ty) E_Impure
  | _ -> failwith "KuiperCustard: slice_read arity"

let slice_write (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [r; idx; v] ->
    mk (EOp ({ po_op = BufWrite; po_ty = None }, [r; idx; v])) TUnit E_Impure
  | _ -> failwith "KuiperCustard: slice_write arity"

(* [Kuiper.Array.Core.gpu_array_alloc] allocates device memory.  Without a
   rule its Pulse body is extracted, and that body allocates a *stack* array
   and returns it -- which compiles, dangles at run time, and made nvcc ICE on
   the element initialiser.  The Krml plugin intercepted the call the same way
   (ExtractKuiper.fst:977-980).

   The element type was erased from the call, so it comes back off the [sized]
   instance's [default], whose type is the element type; the instance's [size]
   is the element width in bytes. *)
let gpu_array_alloc (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [sized; len] ->
    let sz   = field_at 0 "size" sized in
    let dflt = field_at 1 "zero" sized in
    let elt  = dflt.ty in
    let f = mk (EQual ({ ns = ["Kuiper"; "Example"; "ARPort"];
                         id = "kpr_gpu_alloc"; spec = None }, []))
               (TArrow (usize, E_Impure,
                 TArrow (usize, E_Impure, TBuf (TInt (Unsigned, Int8)))))
               E_Pure in
    let call = mk (EApp (f, [sz; len])) (TBuf (TInt (Unsigned, Int8))) E_Impure in
    mk (ECoerce (call, TBuf elt)) (TBuf elt) E_Impure
  | _ -> failwith "KuiperCustard: gpu_array_alloc arity"

(* [Kuiper.SizeT.sizet_to_u32] is a cast in C; its F* body mentions the ghost
   [FStar.SizeT.v] and so cannot be extracted (ExtractKuiper.fst:1104). *)
let sizet_to_u32 (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [x] -> mk (ECast (x, TInt (Unsigned, Int32))) (TInt (Unsigned, Int32)) E_Pure
  | _ -> failwith "KuiperCustard: sizet_to_u32 arity"

(* [Kuiper.SizeT.sizet_and] is a bitwise and at size_t width; its F* body goes
   through [FStar.UInt.logand] and a bit vector (ExtractKuiper.fst:1103). *)
let sizet_ty : cty = TInt (Unsigned, Sizet)

let sizet_and (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [x; y] ->
    mk (EOp ({ po_op = BAnd; po_ty = Some (PInt (Unsigned, Sizet)) }, [x; y]))
       sizet_ty E_Pure
  | _ -> failwith "KuiperCustard: sizet_and arity"


(* [Kuiper.Array.Vectorized.array_vec_cpy] is [noextract] and the Krml plugin
   supplies its meaning (ExtractKuiper.fst:1046): a call to [vec_memcpy] on the
   two offset addresses, with the [sized]/[has_vec_cpy] instances dropped.
   Without this the typeclass binders reach an external and give error 376. *)
let vec_cpy (tys : list cty) (args : list expr) : ML expr =
  match args with
  | _sized :: _hvc :: dst :: dst_off :: src :: src_off :: _rest ->
    let sub (b:expr) (o:expr) =
      mk (EOp ({ po_op = BufSub; po_ty = None }, [b; o])) b.ty E_Pure in
    let vm = mk (EQual ({ ns = port_ns; id = "vec_memcpy"; spec = None },
                        [elem_ty dst.ty; elem_ty src.ty]))
                (TArrow (dst.ty, E_Impure, TArrow (src.ty, E_Impure, TUnit)))
                E_Impure in
    mk (EApp (vm, [sub dst dst_off; sub src src_off])) TUnit E_Impure
  | _ -> failwith ("KuiperCustard: array_vec_cpy arity " ^ show (List.length args))

(* PROBE: does a rule bypass error 376 on an external? *)
(* Emit a call to an unknown C identifier: an [EQual] whose namespace is
   empty prints as a bare name, which is how the wmma:: vocabulary is
   reached without an F* declaration per spelling. *)
let cname (id:string) (t:cty) : expr =
  mk (EQual ({ ns = port_ns; id = id; spec = None }, [])) t E_Pure

(* Section 69.  [wmma_fragment] carries a template target with six
   placeholders, so the Custard type of a fragment is the C++ type: the kind
   and the layout arrive as nullary external types, the m/n/k as [TConst].
   Before section 69 all three of aFrag, bFrag and accumFrag collapsed onto
   one [TApp fragment [et]] and had to be spelled [auto]. *)
let tc_ty (id:string) : cty =
  TApp ({ ns = ["Kuiper"; "TensorCore"; "Base"]; id = id; spec = None }, [])

let ext (id:string) (t:cty) (targs:list cty) : expr =
  mk (EQual ({ ns = port_ns; id = id; spec = None }, targs)) t E_Pure

let rec arrows (args:list cty) (res:cty) : cty =
  match args with
  | [] -> res
  | a :: rest -> TArrow (a, E_Impure, arrows rest res)

(* ---- The host-side runtime (section 79) ---------------------------------
   The Krml plugin mapped six host-side primitives onto the CUDA runtime
   (ExtractKuiper.fst:964-1030) and this plugin did not, so they kept their
   [admit]/Pulse bodies.  Three different wrong things came out:

     - [Kuiper.Ref.memcpy_*] and [Kuiper.Assert.dguard] are [admit]s, so
       they became [abort()] -- loud, but every kernel driver dies at the
       first host/device transfer.
     - [gpu_array_free] became [(void)r;] -- a silent leak.
     - the [Kuiper.Array.Core] memcpys have real Pulse bodies, and those
       bodies walk the *device* pointer from the host one element at a time
       ([dst_garr[dst_off] = x]).  That compiles, and it is undefined
       behaviour: the one failure here that is silent at both compile and
       (potentially) run time.

   The bytes argument is [size * cnt], with [size] read off the [sized]
   instance exactly as [gpu_array_alloc] above reads it. *)

(* [array]/[vec]/[ptr] are [TBuf] and [ref]/[box] are [TRef], but C spells
   both [t*] and the buffer operations apply to either (Syntax.fsti:24-38).
   The [Kuiper.Ref] memcpys take one of each. *)
let pointee (t:cty) : ML cty =
  match t with
  | TBuf e -> e
  | TRef e -> e
  | _ -> die "a buffer or a reference" (mk EAny t E_Pure)

let sub_at (b:expr) (o:expr) : ML expr =
  mk (EOp ({ po_op = BufSub; po_ty = None }, [b; o])) b.ty E_Pure

let bytes_of (sized:expr) (cnt:expr) : ML expr =
  let sz = field_at 0 "size" sized in
  mk (EOp ({ po_op = Mult; po_ty = Some (PInt (Unsigned, Sizet)) }, [sz; cnt]))
     sizet_ty E_Pure

(* [dst] and [src] keep their own element types: the copy is by bytes, and
   the two sides need not agree (a [gpu_ref a] and a [ref a] do, an offset
   array copy does, but nothing here depends on it). *)
let memcpy_call (macro:string) (dst src bytes : expr) : ML expr =
  let f = ext macro
            (arrows [dst.ty; src.ty; sizet_ty] TUnit)
            [pointee dst.ty; pointee src.ty] in
  mk (EApp (f, [dst; src; bytes])) TUnit E_Impure

(* [Kuiper.Ref.memcpy_*]: one element, so
   the count is 1 and the byte size is the [sized] width alone. *)
let ref_memcpy (macro:string) : list cty -> list expr -> ML expr =
  fun _tys args ->
  match args with
  | [sized; dst; src] -> memcpy_call macro dst src (field_at 0 "size" sized)
  | _ -> failwith ("KuiperCustard: " ^ macro ^ " arity "
                   ^ show (List.length args))

(* [Kuiper.Array.Core.memcpy_*], unprimed: [sized; dst; src; cnt]. *)
let arr_memcpy (macro:string) : list cty -> list expr -> ML expr =
  fun _tys args ->
  match args with
  | [sized; dst; src; cnt] -> memcpy_call macro dst src (bytes_of sized cnt)
  | _ -> failwith ("KuiperCustard: " ^ macro ^ " arity "
                   ^ show (List.length args))

(* [Kuiper.Array.Core.memcpy_*']: the offsets are in *elements*, so they
   go through [BufSub] and not into the byte count (ExtractKuiper.fst:999). *)
let arr_memcpy' (macro:string) : list cty -> list expr -> ML expr =
  fun _tys args ->
  match args with
  | [sized; dst; dst_off; src; src_off; cnt] ->
    memcpy_call macro (sub_at dst dst_off) (sub_at src src_off)
                (bytes_of sized cnt)
  | _ -> failwith ("KuiperCustard: " ^ macro ^ "' arity "
                   ^ show (List.length args))

(* [Kuiper.Array.Core.gpu_array_free] is device memory: [cudaFree], not the
   [BufFree] operation, which is the host [free] (ExtractKuiper.fst:982). *)
let gpu_array_free (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [r] ->
    let f = ext "kpr_gpu_free" (arrows [r.ty] TUnit) [pointee r.ty] in
    mk (EApp (f, [r])) TUnit E_Impure
  | _ -> failwith "KuiperCustard: gpu_array_free arity"

(* [Kuiper.Array.Core.get_ref_of_array_cell] is the address of a cell: the
   same [EBufSub] the Krml plugin emitted (ExtractKuiper.fst:974). *)
let get_ref_of_array_cell (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [a; i] -> sub_at a i
  | _ -> failwith "KuiperCustard: get_ref_of_array_cell arity"

(* [Kuiper.Assert.dguard] is a live runtime check in every build;
   [Kuiper.Assert.dassert] compiles away unless KPR_DEBUG.  Both constant-fold
   on [true], which is what the Krml plugin did (ExtractKuiper.fst:582,593)
   and which keeps the ~1300 discharged guards out of the emitted code. *)
let assert_like (macro:string) : list cty -> list expr -> ML expr =
  fun _tys args ->
  match args with
  | [b] ->
    (match b.e with
     | EConst (CBool true) -> unit_expr
     | _ ->
       let f = ext macro (arrows [b.ty] TUnit) [] in
       mk (EApp (f, [b])) TUnit E_Impure)
  | _ -> failwith ("KuiperCustard: " ^ macro ^ " arity")

(* [Kuiper.Kernel.Base.sync_device] is a host-side [cudaDeviceSynchronize]
   (ExtractKuiper.fst:1081).  Without a rule it extracts to a declaration with
   no definition and the link fails. *)
let sync_device : list cty -> list expr -> ML expr =
  fun _tys _args ->
  let f = ext "kpr_sync_device" (arrows [TUnit] TUnit) [] in
  mk (EApp (f, [unit_expr])) TUnit E_Impure

(* The [16]s of the template argument list.  They reach the rule as the
   translated F* indices, which are literals after normalization. *)
let dim_const (e:expr) : ML cty =
  match e.e with
  | EConst c -> TConst c
  | _ -> die "a constant fragment dimension" e

(* The Custard type of a fragment *is* its C++ type: the kind and the layout
   arrive as nullary external types, the m/n/k as [TConst] value arguments.
   Before section 69 all three of aFrag, bFrag and accumFrag collapsed onto
   one [TApp fragment [et]] and had to be spelled [auto]. *)
let frag_cty (et:cty) (knd:expr) (m n k:expr) (layout:expr) : ML cty =
  let ctor (e:expr) : ML string =
    (match e.e with ECtor (n, _) -> n.id | _ -> die "a fragment index constructor" e) in
  let use_t = (match ctor knd with
               | "FragA" -> "ty_matrix_a" | "FragB" -> "ty_matrix_b"
               | "FragAcc" -> "ty_accumulator"
               | s -> failwith ("KuiperCustard: fragment kind " ^ s)) in
  let lay_t = (match ctor layout with
               | "FragLRM" -> "ty_row_major" | "FragLCM" -> "ty_col_major"
               | "FragLAcc" -> "ty_void"
               | s -> failwith ("KuiperCustard: fragment layout " ^ s)) in
  TApp ({ ns = ["Kuiper"; "TensorCore"; "Base"]; id = "wmma_fragment"; spec = None },
        [tc_ty use_t; dim_const m; dim_const n; dim_const k; et; tc_ty lay_t])

(* A default-constructed fragment.  Before section 69 the type was [auto], so
   the declaration needed an initializer and the initializer needed the
   template argument list spelled out by hand -- hence [KPR_INIT] and the
   [kpr_fragment(...)] token macro.  With the type spelled, an uninitialized
   local is the whole thing. *)
let alloc_fragment (tys : list cty) (args : list expr) : ML expr =
  match tys, args with
  | [et], [knd; m; n; k; layout] ->
    mk EAny (frag_cty et knd m n k layout) E_Pure
  | _ -> failwith "KuiperCustard: __alloc_fragment shape"

(* The 2D kernels hold a row of A fragments and a column of B fragments.
   A stack allocation of a known type: no macro, no compound literal. *)
let alloc_array_fragment (tys : list cty) (args : list expr) : ML expr =
  match tys, args with
  | [et], [knd; m; n; k; layout; size] ->
    let ft = frag_cty et knd m n k layout in
    mk (EOp ({ po_op = BufCreate LStack; po_ty = None },
             [mk EAny ft E_Pure; size]))
       (TBuf ft) E_Impure
  | _ -> failwith "KuiperCustard: __alloc_array_fragment shape"

(* [wmma::fill_fragment(fr, v)].  The accumulator of a 2D kernel is filled
   with a zero rather than loaded from C. *)
let mma_fill (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [_knd; _layout; fr; v] ->
    let f = ext "kpr_fill" (arrows [fr.ty; v.ty] TUnit) [fr.ty; v.ty] in
    mk (EApp (f, [fr; v])) TUnit E_Impure
  | _ -> failwith ("KuiperCustard: mma_fill shape, " ^ show (List.length args) ^ " args")

(* [Mkstrided_row_major (offset, stride)]: the offset bumps the pointer, the
   stride is wmma's leading dimension. *)
let strided_off_stride (e:expr) : ML (expr & expr) =
  (field_at 0 "offset" e, field_at 1 "stride" e)

let bufsub (b:expr) (o:expr) : expr =
  mk (EOp ({ po_op = BufSub; po_ty = None }, [b; o])) b.ty E_Pure

(* An elementwise map that extracts to [fun x -> x] is the identity, so the
   macro's register loop is a no-op and the load is a plain one.  Likewise a
   combine [fun acc old -> acc] discards the resident tile, so the
   read-modify-write store degenerates to a plain store. *)
let returns_binder (n:int) (i:int) (e:expr) : ML bool =
  match e.e with
  | EFun (bs, body) ->
    List.length bs = n &&
    (match body.e with
     | EVar v -> v = (List.nth bs i).b_name
     | _ -> false)
  | _ -> false

(* [wmma::load_matrix_sync] for an accumulator takes the memory layout as a
   fourth argument; for A/B it is fixed by the fragment's own type. *)
let mma_load_accum (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [fr; sl; gm] ->
    let (off, ldm) = strided_off_stride sl in
    let gm = bufsub gm off in
    let ft = fr.ty in
    let f = ext "kpr_load_accum" (arrows [ft; gm.ty; ldm.ty] TUnit) [ft; elem_ty gm.ty] in
    mk (EApp (f, [fr; gm; ldm])) TUnit E_Impure
  | _ -> failwith "KuiperCustard: mma_loadAccum shape"

let mma_load_map (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [fmap; fr; sl; gm] ->
    let (off, ldm) = strided_off_stride sl in
    let gm = bufsub gm off in
    let ft = fr.ty in
    if returns_binder 1 0 fmap then
      let f = ext "kpr_load_ab" (arrows [ft; gm.ty; ldm.ty] TUnit) [ft; elem_ty gm.ty] in
      mk (EApp (f, [fr; gm; ldm])) TUnit E_Impure
    else failwith "KuiperCustard: non-identity mma_load map is not supported yet"
  | _ -> failwith "KuiperCustard: mma_load_map shape"

let mma_store_comb (tys : list cty) (args : list expr) : ML expr =
  match args with
  | [gcomb; fr; sl; gm] ->
    let (off, ldm) = strided_off_stride sl in
    let gm = bufsub gm off in
    let ft = fr.ty in
    (* [g acc old]: returning binder 0 keeps the accumulator and drops the
       resident tile, which is a plain store. *)
    if returns_binder 2 0 gcomb then
      let f = ext "kpr_store" (arrows [gm.ty; ft; ldm.ty] TUnit) [ft; elem_ty gm.ty] in
      mk (EApp (f, [gm; fr; ldm])) TUnit E_Impure
    else failwith "KuiperCustard: non-overwrite mma_store combine is not supported yet"
  | _ -> failwith "KuiperCustard: mma_store_comb shape"

let mma_sync (tys : list cty) (args : list expr) : ML expr =
  match tys, args with
  | et_ab :: _, [_scal_ab; _scal_acc; _la; _lb; fa; fb; fc] ->
    let ft = fc.ty in
    let f = mk (EQual ({ ns = port_ns; id = "kpr_mma_sync"; spec = None }, [ft]))
               (TArrow (ft, E_Impure, TArrow (ft, E_Impure,
                  TArrow (ft, E_Impure, TArrow (ft, E_Impure, TUnit)))))
               E_Impure in
    mk (EApp (f, [fc; fa; fb; fc])) TUnit E_Impure
  | _ -> failwith ("KuiperCustard: mma_sync' shape, " ^ show (List.length args) ^ " args")

let mma_probe (nm:string) (tys : list cty) (args : list expr) : ML expr =
  BU.print_endline ("KCPROBE " ^ nm ^
    " tys=[" ^ String.concat "; " (List.map show tys) ^ "]" ^
    " args=[" ^ String.concat "; " (List.map (fun (e:expr) -> tag e ^ ":" ^ show e.ty ^ "=" ^ show e) args) ^ "]");
  mk (EConst CUnit) TUnit E_Impure

(* ---------------------------------------------------------------- WGMMA --

   Hopper's warpgroup MMA.  Unlike the wmma fragment, whose C++ type is a
   template instantiation that has to be reassembled from the kind, the
   dimensions and the layout, [kpr_wgmma_fragment] is a plain named struct
   (include/kuiper/wgmma.h), so the type is spelled by the [custard_extern]
   on [Kuiper.TensorCore.WGMMA.fragment] and there is nothing to compute. *)
let wgmma_frag_cty : cty =
  TApp ({ ns = ["Kuiper"; "TensorCore"; "WGMMA"]; id = "fragment"; spec = None }, [])

(* An uninitialized local of the fragment type.  The Krml plugin had to emit
   [KPR_INIT(kpr_wgmma_fragment)] because it spelled the type [auto]. *)
let wgmma_alloc_fragment (_tys : list cty) (args : list expr) : ML expr =
  match args with
  | [_unit] -> mk EAny wgmma_frag_cty E_Pure
  | _ -> failwith ("KuiperCustard: wgmma alloc_fragment arity "
                   ^ show (List.length args))

let wgmma_fill (_tys : list cty) (args : list expr) : ML expr =
  match args with
  | [fr; x] ->
    let f = ext "kpr_wgmma_fill" (arrows [fr.ty; x.ty] TUnit) [fr.ty] in
    mk (EApp (f, [fr; x])) TUnit E_Impure
  | _ -> failwith ("KuiperCustard: wgmma fill arity " ^ show (List.length args))

(* [load_accum] and [store] differ only in the direction, and both take the
   accumulator tile as a strided row-major [array2]: the offset bumps the
   pointer and the stride is passed alongside. *)
let wgmma_mem (macro:string) : list cty -> list expr -> ML expr =
  fun _tys args ->
  match args with
  | [fr; sl; c] ->
    let (off, ldm) = strided_off_stride sl in
    let c = bufsub c off in
    let f = ext macro (arrows [fr.ty; c.ty; ldm.ty] TUnit) [fr.ty; elem_ty c.ty] in
    mk (EApp (f, [fr; c; ldm])) TUnit E_Impure
  | _ -> failwith ("KuiperCustard: wgmma " ^ macro ^ " arity "
                   ^ show (List.length args))

let wgmma_mma_sync (_tys : list cty) (args : list expr) : ML expr =
  match args with
  | [a; b; fr] ->
    let f = ext "kpr_wgmma_mma_sync" (arrows [a.ty; b.ty; fr.ty] TUnit)
              [fr.ty; elem_ty a.ty] in
    mk (EApp (f, [a; b; fr])) TUnit E_Impure
  | _ -> failwith ("KuiperCustard: wgmma mma_sync arity "
                   ^ show (List.length args))

let _ =
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_sync'")
                  (B.Rule_prim (7, mma_sync));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_loadA_map")
                  (B.Rule_prim (4, mma_load_map));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_loadA_map_cm")
                  (B.Rule_prim (4, mma_probe "mma_loadA_map_cm"));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_loadB_map")
                  (B.Rule_prim (4, mma_load_map));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_loadB_map_cm")
                  (B.Rule_prim (4, mma_probe "mma_loadB_map_cm"));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_loadAccum")
                  (B.Rule_prim (3, mma_load_accum));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_fill")
                  (B.Rule_prim (4, mma_fill));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.mma_store_comb")
                  (B.Rule_prim (4, mma_store_comb));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.with_fragment")
                  (B.Rule_prim (2, mma_probe "with_fragment"));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.__alloc_fragment")
                  (B.Rule_prim (5, alloc_fragment));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.Base.__alloc_array_fragment")
                  (B.Rule_prim (6, alloc_array_fragment));
  B.register_rule (Ident.lid_of_str "Kuiper.SizeT.sizet_and")
                  (B.Rule_prim (2, sizet_and));
  B.register_rule (Ident.lid_of_str "Kuiper.SizeT.sizet_to_u32")
                  (B.Rule_prim (1, sizet_to_u32));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.gpu_array_alloc")
                  (B.Rule_prim (2, gpu_array_alloc));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.slice_read")
                  (B.Rule_prim (2, slice_read));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.slice_write")
                  (B.Rule_prim (3, slice_write));
  B.register_rule (Ident.lid_of_str "Kuiper.Kernel.Base.launch_kernel_full")
                  (B.Rule_prim (2, launch));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Vectorized.array_vec_cpy")
                  (B.Rule_prim (10, vec_cpy));

(* Section 79: the host-side runtime. *)
  B.register_rule (Ident.lid_of_str "Kuiper.Ref.memcpy_host_to_device")
                  (B.Rule_prim (3, ref_memcpy "kpr_memcpy_h2d"));
  B.register_rule (Ident.lid_of_str "Kuiper.Ref.memcpy_device_to_host")
                  (B.Rule_prim (3, ref_memcpy "kpr_memcpy_d2h"));
  B.register_rule (Ident.lid_of_str "Kuiper.Ref.memcpy_device_to_device")
                  (B.Rule_prim (3, ref_memcpy "kpr_memcpy_d2d"));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.memcpy_host_to_device")
                  (B.Rule_prim (4, arr_memcpy "kpr_memcpy_h2d"));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.memcpy_device_to_host")
                  (B.Rule_prim (4, arr_memcpy "kpr_memcpy_d2h"));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.memcpy_device_to_device")
                  (B.Rule_prim (4, arr_memcpy "kpr_memcpy_d2d"));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.memcpy_host_to_device'")
                  (B.Rule_prim (6, arr_memcpy' "kpr_memcpy_h2d"));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.memcpy_device_to_host'")
                  (B.Rule_prim (6, arr_memcpy' "kpr_memcpy_d2h"));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.gpu_array_free")
                  (B.Rule_prim (1, gpu_array_free));
  (* Section 96: Hopper WGMMA. *)
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.WGMMA.alloc_fragment")
                  (B.Rule_prim (1, wgmma_alloc_fragment));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.WGMMA.fill")
                  (B.Rule_prim (2, wgmma_fill));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.WGMMA.load_accum")
                  (B.Rule_prim (3, wgmma_mem "kpr_wgmma_load_accum"));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.WGMMA.store")
                  (B.Rule_prim (3, wgmma_mem "kpr_wgmma_store"));
  B.register_rule (Ident.lid_of_str "Kuiper.TensorCore.WGMMA.mma_sync")
                  (B.Rule_prim (3, wgmma_mma_sync));
  B.register_rule (Ident.lid_of_str "Kuiper.Array.Core.get_ref_of_array_cell")
                  (B.Rule_prim (2, get_ref_of_array_cell));
  B.register_rule (Ident.lid_of_str "Kuiper.Assert.dguard")
                  (B.Rule_prim (1, assert_like "kpr_guard"));
  B.register_rule (Ident.lid_of_str "Kuiper.Assert.dassert")
                  (B.Rule_prim (1, assert_like "kpr_assert"));
  B.register_rule (Ident.lid_of_str "Kuiper.Kernel.Base.sync_device")
                  (B.Rule_prim (1, sync_device));
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_memcpy_h2d");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_memcpy_d2h");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_memcpy_d2d");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_gpu_free");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_guard");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_assert");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_sync_device");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_shmem_fits");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_set_max_dyn_shmem");

  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.vec_memcpy");
  B.register_root (Ident.lid_of_str "Kuiper.Example.ARPort.kpr_gpu_alloc");
  B.register_root (Ident.lid_of_str shmem_at_lid);
  (* One launcher per capture count; all of them are the same variadic macro. *)
  B.register_root (Ident.lid_of_str kcall_lid);
  List.iter
    (fun n -> B.register_root (Ident.lid_of_str (kcall_lid ^ show n)))
    [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12];
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_mma_sync"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_store"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_load_ab"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_load_accum"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_fill"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_wgmma_fill"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_wgmma_load_accum"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_wgmma_store"));
  B.register_root (Ident.lid_of_str ("Kuiper.Example.ARPort.kpr_wgmma_mma_sync"));
  (* Section 69: the rule names these type declarations directly. *)
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.wmma_fragment"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.WGMMA.fragment"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.ty_matrix_a"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.ty_matrix_b"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.ty_accumulator"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.ty_row_major"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.ty_col_major"));
  B.register_root (Ident.lid_of_str ("Kuiper.TensorCore.Base.ty_void"));
  B.register_root (Ident.lid_of_str blockidx_lid);
  B.register_root (Ident.lid_of_str threadidx_lid)
