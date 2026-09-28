module ExtractKuiper

(* Kuiper's extraction rules for Custard (--codegen Custard
   --custard_backend KrmlC).

   Each rule replaces a use of a Kuiper primitive by the CUDA it stands for.
   Most of the targets are CUDA intrinsics or macros from include/kuiper.h,
   which must not get an [extern] prototype, so they are referenced as bare
   (namespace-less) names: the karamel backend prints those verbatim, and
   karamel accepts unknown names (its warning 2 is disabled).  Custard
   exempts such names from its dangling-reference check and never
   eta-reduces a call to one, since their arity is unknown.  The few names
   that are not valid C identifiers ([wmma::...], [threadIdx.x]) are spelled
   with [__]/[_] and fixed up by scripts/fixup.sed, and fragment types are
   spelled [auto_AMP] and rewritten to [auto&].

   The arity of each [Rule_prim] counts the arguments Custard *retains* after
   erasure; see [FStarC.Custard.Builtins.rule]. *)

open FStarC
open FStarC.Effect
open FStarC.List
open FStarC.Class.Show
open FStarC.Const
open FStarC.Custard.Syntax

module B     = FStarC.Custard.Builtins
module Ident = FStarC.Ident
module BU    = FStarC.Util

(* -------------------------------------------------------------------- *)
(* Helpers                                                              *)
(* -------------------------------------------------------------------- *)

let fail (#a:Type) (msg:string) : ML a =
  failwith ("ExtractKuiper: " ^ msg)

let tag (e:expr) : string =
  match e.e with
  | EConst _   -> "a constant"         | EVar _     -> "a local variable"
  | EQual _    -> "a top-level name"   | ELet _     -> "a let"
  | EApp _     -> "an application"     | EFun _     -> "a lambda"
  | EMatch _   -> "a match"            | EIf _      -> "an if"
  | ESeq _     -> "a sequence"         | ECtor _    -> "a constructor application"
  | ETuple _   -> "a tuple"            | ERecord _  -> "a record literal"
  | EProj _    -> "a projection"       | EDiscrim _ -> "a discriminator"
  | ECoerce _  -> "a coercion"         | ECast _    -> "a cast"
  | EAny       -> "an arbitrary value" | EAbort _   -> "an abort"
  | EOp _      -> "a primitive operation" | EWhile _ -> "a while"
  | ERaise _   -> "a raise"            | ETry _     -> "a try"

let die (#a:Type) (want:string) (e:expr) : ML a =
  fail ("expected " ^ want ^ ", got " ^ tag e ^ ": " ^ show e)

let rec strip (e:expr) : ML expr =
  match e.e with
  | ECoerce (e', _) -> strip e'
  | _ -> e

let usize : cty = TInt (Unsigned, WSizet)
let tbool : cty = TApp ({ ns = ["Prims"]; id = "bool"; spec = None }, [])
let u32   : cty = TInt (Unsigned, W32)
let i32   : cty = TInt (Signed, W32)

let bare_name (id:string) : name = { ns = []; id = id; spec = None }

(* A bare C name, printed verbatim by the karamel backend. *)
let cname (id:string) (t:cty) : expr =
  mk (EQual (bare_name id, [])) t E_Pure

(* A bare C type name. *)
let ctype (id:string) : cty = TApp (bare_name id, [])

let rec arrows (args:list cty) (res:cty) : cty =
  match args with
  | [] -> res
  | a :: rest -> TArrow (a, E_Impure, arrows rest res)

(* A call to a bare C function or macro. *)
let call_eff (eff:eff) (id:string) (args:list expr) (res:cty) : ML expr =
  let args = if Nil? args then [unit_expr] else args in
  let f = cname id (arrows (List.map (fun (a:expr) -> a.ty) args) res) in
  mk (EApp (f, args)) res eff

let call  = call_eff E_Impure
let pcall = call_eff E_Pure

let must (e:expr) : ML expr = call "MUST" [e] TUnit

let seq (a:expr) (b:expr) : ML expr = mk (ESeq (a, b)) b.ty (join_eff a.eff b.eff)

let uconst (n:int) : expr =
  mk (EConst (CInt (n, Dec, Some (Unsigned, WSizet)))) usize E_Pure

let uop (o:op) (a b : expr) : ML expr =
  mk (EOp ({ po_op = o; po_ty = Some (PInt (Unsigned, WSizet)) }, [a; b]))
     (match o with Lt | Lte | Gt | Gte | Eq | Neq -> tbool | _ -> usize) E_Pure

let is_const_true (e:expr) : ML bool =
  match (strip e).e with
  | EConst (CBool true) -> true
  | _ -> false

let is_const_zero (e:expr) : ML bool =
  match (strip e).e with
  | EConst (CInt (0, _, _)) -> true
  | _ -> false

let pointee (t:cty) : ML cty =
  match t with
  | TBuf e | TRef e -> e
  | _ -> fail ("expected a buffer or reference type, got " ^ show t)

let bufsub (b:expr) (o:expr) : ML expr =
  if is_const_zero o then b
  else mk (EOp ({ po_op = BufSub; po_ty = None }, [b; o])) b.ty E_Pure

(* A field of a record literal (or constructor application), looking through
   coercions: Custard wraps typeclass dictionaries and other existentials in
   [(e <: t<any>)]. *)
let field (names:list string) (i:int) (e:expr) : ML expr =
  let e0 = strip e in
  match e0.e with
  | ERecord (_, fs) ->
    (match BU.try_find (fun (f, _) -> List.mem f names) fs with
     | Some (_, v) -> strip v
     | None -> die ("a record with a field " ^ String.concat "/" names) e0)
  | ECtor (_, args) ->
    if i < List.length args then strip (List.nth args i)
    else die "a wider constructor" e0
  | _ -> die ("a record literal (looking for " ^ String.concat "/" names ^ ")") e0

let rec elements (e:expr) : ML (list expr) =
  let e = strip e in
  match e.e with
  | ECtor (n, args) ->
    if n.id = "Nil" then []
    else if n.id = "Cons" then
      (match List.rev args with
       | tl :: hd :: _ -> hd :: elements tl
       | _ -> die "a two-argument Cons" e)
    else die "Prims.Nil or Prims.Cons" e
  | _ -> die "a list literal" e

(* The C byte size of a type, when it is a scalar or a pointer. *)
let sizeof_cty (t:cty) : ML (option int) =
  match t with
  | TInt (_, W8) -> Some 1
  | TInt (_, W16) -> Some 2
  | TInt (_, W32) -> Some 4
  | TInt (_, W64) -> Some 8
  | TInt (_, WSizet) -> Some (if Options.custard_sizet_32 () then 4 else 8)
  | TFloat Float16 | TFloat BFloat16 -> Some 2
  | TFloat Float32 -> Some 4
  | TFloat Float64 -> Some 8
  | TBuf _ | TRef _ -> Some 8
  | TApp ({ ns = []; id = "half" }, [])
  | TApp ({ ns = []; id = "__nv_bfloat16" }, []) -> Some 2
  | _ -> None

(* The element size, from a [Kuiper.Sized.sized] dictionary when it is a
   literal, otherwise from the element type. *)
let elem_size (sized:expr) (elt:cty) : ML expr =
  match (strip sized).e with
  | ERecord _ | ECtor _ -> field ["size"] 0 sized
  | _ ->
    match sizeof_cty elt with
    | Some n -> uconst n
    | None -> die ("a sized dictionary (element type " ^ show elt ^ ")") sized

(* Free variables, with their types, in order of first occurrence. *)
let rec pvars (p:pat) : ML (list string) =
  match p with
  | PVar x -> [x]
  | PCtor (_, ps) | PTuple ps | POr ps -> List.collect pvars ps
  | PRecord (_, fs) -> List.collect (fun (_, q) -> pvars q) fs
  | _ -> []

let rec fvs (bound:list string) (e:expr) : ML (list (string & cty)) =
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
  | EMatch (sc, brs) | ETry (sc, brs) ->
    fvs bound sc @
    List.collect (fun ((p, g, b) : branch) ->
      let bound = pvars p @ bound in
      (match g with Some ge -> fvs bound ge | None -> []) @ fvs bound b) brs
  | EConst _ | EQual _ | EAny | EAbort _ -> []

let free_vars (bound:list string) (e:expr) : ML (list (string & cty)) =
  BU.remove_dups (fun (a, _) (b, _) -> a = b) (fvs bound e)

let occurs (x:string) (e:expr) : ML bool =
  List.existsb (fun (y, _) -> y = x) (fvs [] e)

(* Substitute a closed expression for a variable. *)
let rec subst (x:string) (v:expr) (e:expr) : ML expr =
  let s = subst x v in
  let re (e':expr') : expr = { e with e = e' } in
  match e.e with
  | EVar y -> if y = x then v else e
  | EApp (h, args) -> re (EApp (s h, List.map s args))
  | EOp (o, args) -> re (EOp (o, List.map s args))
  | ECtor (n, args) -> re (ECtor (n, List.map s args))
  | ETuple args -> re (ETuple (List.map s args))
  | EFun (bs, b) ->
    if List.existsb (fun (b:binder) -> b.b_name = x) bs then e
    else re (EFun (bs, s b))
  | ELet (y, t, d, b) -> re (ELet (y, t, s d, (if y = x then b else s b)))
  | EIf (c, a, b) -> re (EIf (s c, s a, s b))
  | ESeq (a, b) -> re (ESeq (s a, s b))
  | EWhile (a, b) -> re (EWhile (s a, s b))
  | ECoerce (y, t) -> re (ECoerce (s y, t))
  | ECast (y, t) -> re (ECast (s y, t))
  | EProj (y, n, f) -> re (EProj (s y, n, f))
  | EDiscrim (y, n) -> re (EDiscrim (s y, n))
  | ERaise y -> re (ERaise (s y))
  | ERecord (n, fs) -> re (ERecord (n, List.map (fun (f, y) -> (f, s y)) fs))
  | EMatch (sc, brs) ->
    re (EMatch (s sc, List.map (fun ((p, g, b) : branch) ->
                  if List.mem x (pvars p) then (p, g, b)
                  else (p, (match g with Some g -> Some (s g) | None -> None), s b)) brs))
  | ETry (sc, brs) ->
    re (ETry (s sc, List.map (fun ((p, g, b) : branch) ->
                  if List.mem x (pvars p) then (p, g, b)
                  else (p, (match g with Some g -> Some (s g) | None -> None), s b)) brs))
  | EConst _ | EQual _ | EAny | EAbort _ -> e

(* The binders and body of a (possibly curried) lambda. *)
let rec lambda_parts (e:expr) : ML (list binder & expr) =
  match e.e with
  | EFun (bs, body) ->
    let bs', body' = lambda_parts body in
    (bs @ bs', body')
  | ECoerce (e', _) -> lambda_parts e'
  | _ -> ([], e)

(* Apply a function to arguments, beta-reducing a literal lambda. *)
let apply (f:expr) (args:list expr) (res:cty) : ML expr =
  let bs, body = lambda_parts f in
  if List.length bs = List.length args then
    List.fold_left2 (fun body (b:binder) a -> subst b.b_name a body) body bs args
  else mk (EApp (f, args)) res E_Impure

(* Is this [fun x1 ... xn -> xi]? *)
let returns_binder (n:int) (i:int) (e:expr) : ML bool =
  let bs, body = lambda_parts e in
  List.length bs = n &&
  (match (strip body).e with
   | EVar v -> v = (List.nth bs i).b_name
   | _ -> false)

let fresh_ctr : ref int = mk_ref 0
let fresh (base:string) : ML string =
  let n = !fresh_ctr in
  fresh_ctr := n + 1;
  base ^ "_kpr" ^ show n

(* -------------------------------------------------------------------- *)
(* Types                                                                *)
(* -------------------------------------------------------------------- *)

(* karamel has no 16-bit float width, so the CUDA types are named. *)
let half_t : cty = ctype "half"
let bf16_t : cty = ctype "__nv_bfloat16"
(* Fragments are spelled [auto&] (scripts/fixup.sed): their C++ type is a
   template instantiation karamel cannot express. *)
let auto_t : cty = ctype "auto_AMP"
let stream_t : cty = ctype "cudaStream_t"

(* The C spelling of a fragment element type, as a macro argument. *)
let c_type_name (t:cty) : ML string =
  match t with
  | TApp ({ ns = []; id = id }, []) -> id
  | TFloat Float32 -> "float"
  | TFloat Float64 -> "double"
  | _ -> fail ("unexpected fragment element type " ^ show t)

(* -------------------------------------------------------------------- *)
(* Kernel launch                                                        *)
(* -------------------------------------------------------------------- *)

(* A launch bound is an upper limit on the block size, not a request for a
   minimum occupancy, so only attach one when extraction knows the actual block
   size: a runtime-sized launch must not inherit a bound from another
   instantiation.  nvcc caps the bound at 1024 threads. *)
let kernel_qualifier (nthr:expr) : ML string =
  match (strip nthr).e with
  | EConst (CInt (n, _, _)) ->
    if 0 < n && n <= 1024
    then "__global__ __launch_bounds__(" ^ show n ^ ")"
    else "__global__"
  | _ -> "__global__"

(* Kernels are named after the definition that launches them. *)
let kernel_counters : ref (list (string & int)) = mk_ref []

let sanitize (s:string) : ML string =
  String.concat ""
    (List.map (fun c ->
       if (BU.is_letter_or_digit c && BU.int_of_char c < 128) || c = '_'
       then String.make 1 c else "_") (String.list_of_string s))

let fresh_kernel_name () : ML (string & string) =
  let base, pretty =
    match B.current_decl () with
    | Some nm ->
      (sanitize (nm.id ^ (match nm.spec with Some s -> "__" ^ s | None -> "")),
       nm.id)
    | None -> ("kernel", "<unknown>") in
  let n = match List.assoc base !kernel_counters with Some n -> n | None -> 0 in
  kernel_counters := (base, n + 1) :: !kernel_counters;
  ("__hoisted_" ^ base ^ "_" ^ show n, pretty)

(* The default limit on dynamic shared memory per block, in bytes.  Raising
   it requires cudaFuncSetAttribute; at or below it, that call does nothing. *)
let default_dynamic_shmem_limit : int = 49152

(* The shared memory arrays requested by a kernel: element size and length. *)
let parse_shmem (d:expr) : ML (expr & expr) =
  let sized = field ["sized"] 0 d in
  let len   = field ["len"] 1 d in
  let dflt  = field ["default"] 1 sized in
  (elem_size sized dflt.ty, len)

let shmem_at (off:expr) (elt:cty) : ML expr =
  let raw = pcall "KPR_SHMEM_AT" [off] (TBuf (TInt (Unsigned, W8))) in
  mk (ECast (raw, TBuf elt)) (TBuf elt) E_Pure

let tuple_components (t:cty) : option (cty & cty) =
  match t with
  | TApp (_, [a; b]) -> Some (a, b)
  | TTuple [a; b] -> Some (a, b)
  | _ -> None

let rec build_shmems (t:cty) (offs:list (expr & cty)) (tailv:expr) : ML expr =
  match offs with
  | [] -> tailv
  | (off, elt) :: offs' ->
    (match tuple_components t with
     | Some (t1, t2) ->
       let mktup2 = { ns = ["FStar"; "Pervasives"; "Native"]; id = "Mktuple2"; spec = None } in
       mk (ECtor (mktup2, [shmem_at off elt; build_shmems t2 offs' tailv])) t E_Pure
     | None -> fail ("expected a pair type for shared memory, got " ^ show t))

(* Rebuild [e] with [f] applied to its immediate subexpressions. *)
let emap (f:expr -> ML expr) (e:expr) : ML expr =
  let re (e':expr') : expr = { e with e = e' } in
  let fo (g:option expr) : ML (option expr) = match g with Some g -> Some (f g) | None -> None in
  let fb (brs:list branch) : ML (list branch) =
    List.map (fun ((p, g, b) : branch) -> (p, fo g, f b)) brs in
  match e.e with
  | EApp (h, args) -> re (EApp (f h, List.map f args))
  | EOp (o, args) -> re (EOp (o, List.map f args))
  | ECtor (n, args) -> re (ECtor (n, List.map f args))
  | ETuple args -> re (ETuple (List.map f args))
  | EFun (bs, b) -> re (EFun (bs, f b))
  | ELet (y, t, d, b) -> re (ELet (y, t, f d, f b))
  | EIf (c, a, b) -> re (EIf (f c, f a, f b))
  | ESeq (a, b) -> re (ESeq (f a, f b))
  | EWhile (a, b) -> re (EWhile (f a, f b))
  | ECoerce (y, t) -> re (ECoerce (f y, t))
  | ECast (y, t) -> re (ECast (f y, t))
  | EProj (y, n, fl) -> re (EProj (f y, n, fl))
  | EDiscrim (y, n) -> re (EDiscrim (f y, n))
  | ERaise y -> re (ERaise (f y))
  | ERecord (n, fs) -> re (ERecord (n, List.map (fun (fl, y) -> (fl, f y)) fs))
  | EMatch (sc, brs) -> re (EMatch (f sc, fb brs))
  | ETry (sc, brs) -> re (ETry (f sc, fb brs))
  | EConst _ | EVar _ | EQual _ | EAny | EAbort _ -> e

(* Resolve every use of the shared memory tuple [sh] -- projections, and
   matches that destructure it, however deeply nested -- to the arrays
   themselves, so that the tuple (and the [int] it ends in) never reaches C.
   [env] maps variables to the path into [sh] they denote. *)
let rw_shmems (sh:string) (offs:list (expr & cty)) (tailv:expr) (e:expr) : ML expr =
  let rec comp (path:list int) (offs:list (expr & cty)) : ML (option expr) =
    match path, offs with
    | [], [] -> Some tailv
    | [0], (off, elt) :: _ -> Some (shmem_at off elt)
    | 1 :: rest, _ :: offs' -> comp rest offs'
    | _ -> None in
  let proj_index (f:string) : option int =
    if f = "_1" || f = "fst" then Some 0
    else if f = "_2" || f = "snd" then Some 1
    else None in
  let rec resolve (env:list (string & list int)) (e:expr) : ML (option (list int)) =
    match e.e with
    | ECoerce (x, _) -> resolve env x
    | EVar y -> List.assoc y env
    | EProj (x, _, f) ->
      (match resolve env x, proj_index f with
       | Some p, Some i -> Some (p @ [i])
       | _ -> None)
    | EApp (h, [x]) ->
      (match (strip h).e with
       | EQual (n, _) ->
         let i = if BU.ends_with n.id "item___1" || n.id = "fst" then Some 0
                 else if BU.ends_with n.id "item___2" || n.id = "snd" then Some 1
                 else None in
         (match resolve env x, i with
          | Some p, Some i -> Some (p @ [i])
          | _ -> None)
       | _ -> None)
    | _ -> None in
  let unbind (env:list (string & list int)) (xs:list string) : ML (list (string & list int)) =
    List.filter (fun (y, _) -> not (List.mem y xs)) env in
  (* Bind the variables of [p], a pattern matched against [path]. *)
  let rec bindpat (env:list (string & list int)) (p:pat) (path:list int)
      (lets:list (string & expr)) : ML (option (list (string & list int) & list (string & expr))) =
    match p with
    | PWild -> Some (env, lets)
    | PVar x ->
      (match comp path offs with
       | Some v -> Some (unbind env [x], lets @ [(x, v)])
       | None -> Some ((x, path) :: unbind env [x], lets))
    | PCtor (_, [p1; p2]) | PTuple [p1; p2] ->
      (match bindpat env p1 (path @ [0]) lets with
       | Some (env, lets) -> bindpat env p2 (path @ [1]) lets
       | None -> None)
    | PRecord (_, fs) ->
      List.fold_left (fun acc (f, fp) ->
        match acc, proj_index f with
        | Some (env, lets), Some i -> bindpat env fp (path @ [i]) lets
        | _ -> None) (Some (env, lets)) fs
    | _ -> None in
  let rec rw (env:list (string & list int)) (e:expr) : ML expr =
    if Nil? env then e else
    match resolve env e with
    | Some path when Some? (comp path offs) -> Some?.v (comp path offs)
    | _ ->
    match e.e with
    | EMatch (sc, [(p, None, body)]) when Some? (resolve env sc) ->
      (match bindpat env p (Some?.v (resolve env sc)) [] with
       | Some (env', lets) ->
         let body = rw env' body in
         List.fold_right (fun (x, v) (b:expr) ->
           if occurs x b then mk (ELet (x, v.ty, v, b)) b.ty b.eff else b) lets body
       | None -> emap (rw env) e)
    | ELet (x, t, d, b) ->
      (match resolve env d with
       | Some path when None? (comp path offs) -> rw ((x, path) :: unbind env [x]) b
       | _ -> { e with e = ELet (x, t, rw env d, rw (unbind env [x]) b) })
    | EFun (bs, b) ->
      { e with e = EFun (bs, rw (unbind env (List.map (fun (b:binder) -> b.b_name) bs)) b) }
    | EMatch (sc, brs) ->
      { e with e = EMatch (rw env sc,
          List.map (fun ((p, g, b) : branch) ->
            let env = unbind env (pvars p) in
            (p, (match g with Some g -> Some (rw env g) | None -> None), rw env b)) brs) }
    | _ -> emap (rw env) e in
  rw [(sh, [])] e

(* The element types of the shared memory arrays, from the type of [sh]. *)
let rec shmem_elts (t:cty) (n:int) : ML (list cty) =
  if n = 0 then []
  else match tuple_components t with
       | Some (TBuf elt, rest) -> elt :: shmem_elts rest (n - 1)
       | _ -> fail ("unexpected shared memory type " ^ show t)

(* Bring a kernel body to the form [fun sh bid tid u -> body]. *)
let kernel_binders (f:expr) : ML (list binder & expr) =
  let bs, body = lambda_parts f in
  if List.length bs >= 4 then begin
    let bs4 = fst (BU.first_N 4 bs) in
    let rest = snd (BU.first_N 4 bs) in
    let body = if Nil? rest then body else mk (EFun (rest, body)) f.ty E_Pure in
    (bs4, body)
  end else begin
    (* eta-expand *)
    let rec arg_tys (t:cty) (n:int) : ML (list cty & cty) =
      if n = 0 then ([], t)
      else match t with
           | TArrow (a, _, b) -> let ts, r = arg_tys b (n - 1) in (a :: ts, r)
           | _ -> fail ("kernel body has a non-function type " ^ show t) in
    let tys, res = arg_tys f.ty 4 in
    let bs = List.map (fun t -> { b_name = fresh "karg"; b_ty = t }) tys in
    let args = List.map (fun (b:binder) -> mk (EVar b.b_name) b.b_ty E_Pure) bs in
    (bs, mk (EApp (f, args)) res E_Impure)
  end

let launch (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [kdesc; stream] ->
    let nblk = field ["nblk"] 0 kdesc in
    let nthr = field ["nthr"] 1 kdesc in
    let shmems = List.map parse_shmem (elements (field ["shmems_desc"] 2 kdesc)) in
    let kf = field ["f"] 3 kdesc in
    let bs, inner = kernel_binders kf in
    let b_sh, b_bid, b_tid, b_unit =
      match bs with [a; b; c; d] -> (a, b, c, d) | _ -> fail "unreachable" in
    let bound = List.map (fun (b:binder) -> b.b_name) bs in

    (* Shared memory: consecutive byte offsets into the dynamic region. *)
    let elts = shmem_elts b_sh.b_ty (List.length shmems) in
    let offs, smem =
      List.fold_left2 (fun (offs, off) (sz, len) elt ->
                         (offs @ [(off, elt)], uop Add off (uop Mult sz len)))
                      ([], uconst 0) shmems elts in
    let smem = if Nil? shmems then uconst 0 else smem in
    let tail_ty =
      let rec go (t:cty) (n:int) : ML cty =
        if n = 0 then t
        else match tuple_components t with
             | Some (_, r) -> go r (n - 1)
             | None -> t in
      go b_sh.b_ty (List.length shmems) in
    let tailv = mk (EConst (CInt (0, Dec, None))) tail_ty E_Pure in
    let inner =
      let inner = rw_shmems b_sh.b_name offs tailv inner in
      if occurs b_sh.b_name inner
      then mk (ELet (b_sh.b_name, b_sh.b_ty, build_shmems b_sh.b_ty offs tailv, inner))
              inner.ty inner.eff
      else inner in

    (* The indices come from CUDA. *)
    let bind (b:binder) (v:expr) (body:expr) : ML expr =
      if occurs b.b_name body then mk (ELet (b.b_name, b.b_ty, v, body)) body.ty body.eff
      else body in
    let inner = bind b_unit unit_expr inner in
    let inner = bind b_tid (cname "threadIdx_x" b_tid.b_ty) inner in
    let inner = bind b_bid (cname "blockIdx_x" b_bid.b_ty) inner in

    (* Close the body over its free variables; they become kernel parameters. *)
    let caps = free_vars bound inner in
    let cap_bs = List.map (fun (v, t) -> { b_name = v; b_ty = t }) caps in
    let kbs = if Nil? cap_bs then [{ b_name = "extra_unit"; b_ty = TUnit }] else cap_bs in
    let kty = arrows (List.map (fun (b:binder) -> b.b_ty) kbs) TUnit in
    let closed = mk (EFun (kbs, inner)) kty E_Impure in
    let kname, parent = fresh_kernel_name () in
    let kernel = B.lift_named kname
                   [Comment ("  hoisted when extracting " ^ parent);
                    Private;
                    Prologue (kernel_qualifier nthr);
                    (* Anything the kernel reaches is device code; a helper
                       shared with host code needs both qualifiers. *)
                    ClosurePrologue ("__device__", "__device__ __host__")]
                   closed in
    let cap_args = List.map (fun (v, t) -> mk (EVar v) t E_Pure) caps in
    let kcall = call "KPR_KCALL" ([kernel; nblk; nthr; smem; stream] @ cap_args) TUnit in
    if Nil? shmems then kcall
    else begin
      (* Check that the request fits the device, and opt in to more than the
         default 48KiB of dynamic shared memory when needed.  karamel folds the
         test away when the size is a constant. *)
      let fits = call "KPR_SHMEM_FITS" [smem] TUnit in
      let attr = cname "cudaFuncAttributeMaxDynamicSharedMemorySize" i32 in
      let setattr = must (call "cudaFuncSetAttribute" [kernel; attr; smem] i32) in
      let cond = uop Gte smem (uconst default_dynamic_shmem_limit) in
      let setup = mk (EIf (cond, setattr, unit_expr)) TUnit E_Impure in
      seq fits (seq setup kcall)
    end
  | _ -> fail ("launch_kernel_full: expected 2 arguments, got " ^ show (List.length args))

(* -------------------------------------------------------------------- *)
(* For loops                                                            *)
(* -------------------------------------------------------------------- *)

(* [for_loop lo hi f]: a counter in a stack cell, which karamel turns into a
   [for] loop. *)
let for_loop (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [lo; hi; f] ->
    let i = fresh "i" in
    let iv = mk (EVar i) (TRef usize) E_Pure in
    let rd () = mk (EOp ({ po_op = BufRead; po_ty = None }, [iv; uconst 0])) usize E_Impure in
    let with_hi (k:expr -> ML expr) : ML expr =
      match (strip hi).e with
      | EVar _ | EConst _ -> k hi
      | _ ->
        let h = fresh "hi" in
        let body = k (mk (EVar h) usize E_Pure) in
        mk (ELet (h, usize, hi, body)) TUnit E_Impure in
    with_hi (fun hi ->
      let cond = uop Lt (rd ()) hi in
      let j = fresh "j" in
      let jv = mk (EVar j) usize E_Pure in
      let step = mk (EOp ({ po_op = BufWrite; po_ty = None },
                          [iv; uconst 0; uop Add (rd ()) (uconst 1)])) TUnit E_Impure in
      let body = mk (ELet (j, usize, rd (), seq (apply f [jv] TUnit) step)) TUnit E_Impure in
      let loop = mk (EWhile (cond, body)) TUnit E_Impure in
      let cell = mk (EOp ({ po_op = BufCreate LStack; po_ty = None }, [lo; uconst 1]))
                    (TRef usize) E_Impure in
      mk (ELet (i, TRef usize, cell, loop)) TUnit E_Impure)
  | _ -> fail "for_loop: expected 3 arguments"

(* -------------------------------------------------------------------- *)
(* Memory                                                               *)
(* -------------------------------------------------------------------- *)

let memcpy_kind (k:string) : expr = cname k i32

let cuda_memcpy (kind:string) (dst src bytes : expr) : ML expr =
  must (call "cudaMemcpy" [dst; src; bytes; memcpy_kind kind] i32)

(* [Kuiper.Ref.memcpy_*]: one element. *)
let ref_memcpy (kind:string) (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [sized; dst; src] -> cuda_memcpy kind dst src (elem_size sized (pointee dst.ty))
  | _ -> fail ("Ref.memcpy: expected 3 arguments, got " ^ show (List.length args))

(* [Kuiper.Array.Core.memcpy_*]. *)
let arr_memcpy (kind:string) (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [sized; dst; src; cnt] ->
    cuda_memcpy kind dst src (uop Mult (elem_size sized (pointee dst.ty)) cnt)
  | _ -> fail ("Array.memcpy: expected 4 arguments, got " ^ show (List.length args))

(* [Kuiper.Array.Core.memcpy_*']: the offsets are in elements. *)
let arr_memcpy' (kind:string) (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [sized; dst; dst_off; src; src_off; cnt] ->
    cuda_memcpy kind (bufsub dst dst_off) (bufsub src src_off)
                (uop Mult (elem_size sized (pointee dst.ty)) cnt)
  | _ -> fail ("Array.memcpy': expected 6 arguments, got " ^ show (List.length args))

let gpu_array_alloc (tys:list cty) (args:list expr) : ML expr =
  match tys, args with
  | [elt], [sized; len] ->
    let raw = call "KPR_GPU_ALLOC" [elem_size sized elt; len] (TBuf (TInt (Unsigned, W8))) in
    mk (ECast (raw, TBuf elt)) (TBuf elt) E_Impure
  | _ -> fail ("gpu_array_alloc: unexpected shape, " ^ show (List.length tys)
               ^ " types and " ^ show (List.length args) ^ " arguments")

let gpu_array_free (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [a] -> must (call "cudaFree" [a] i32)
  | _ -> fail "gpu_array_free: expected 1 argument"

let slice_read (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [a; i] -> mk (EOp ({ po_op = BufRead; po_ty = None }, [a; i])) (pointee a.ty) E_Impure
  | _ -> fail "slice_read: expected 2 arguments"

let slice_write (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [a; i; v] -> mk (EOp ({ po_op = BufWrite; po_ty = None }, [a; i; v])) TUnit E_Impure
  | _ -> fail "slice_write: expected 3 arguments"

let get_ref_of_array_cell (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [a; i] ->
    let r = mk (EOp ({ po_op = BufSub; po_ty = None }, [a; i])) a.ty E_Pure in
    mk (ECoerce (r, TRef (pointee a.ty))) (TRef (pointee a.ty)) E_Pure
  | _ -> fail "get_ref_of_array_cell: expected 2 arguments"

let array_vec_cpy (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [_sized; _hvc; dst; dst_off; src; src_off] ->
    call "vec_memcpy" [bufsub dst dst_off; bufsub src src_off] TUnit
  | _ -> fail ("array_vec_cpy: expected 6 arguments, got " ^ show (List.length args))

(* -------------------------------------------------------------------- *)
(* Tensor cores                                                         *)
(* -------------------------------------------------------------------- *)

let ctor_id (e:expr) : ML string =
  match (strip e).e with
  | ECtor (n, _) -> n.id
  | EQual (n, _) -> n.id
  | _ -> die "a constructor" e

let kpr_fragment (et:cty) (knd m n k layout : expr) : ML expr =
  let knd = match ctor_id knd with
            | "FragA" -> "wmma__matrix_a"
            | "FragB" -> "wmma__matrix_b"
            | "FragAcc" -> "wmma__accumulator"
            | s -> fail ("unexpected fragment kind " ^ s) in
  let layout = match ctor_id layout with
               | "FragLRM" -> [cname "wmma__row_major" i32]
               | "FragLCM" -> [cname "wmma__col_major" i32]
               | "FragLAcc" -> []
               | s -> fail ("unexpected fragment layout " ^ s) in
  pcall "kpr_fragment"
    ([cname knd i32; m; n; k; cname (c_type_name et) i32] @ layout) i32

let alloc_fragment (tys:list cty) (args:list expr) : ML expr =
  match tys, args with
  | [et], [knd; m; n; k; layout] ->
    call "KPR_INIT" [kpr_fragment et knd m n k layout] auto_t
  | _ -> fail "__alloc_fragment: unexpected shape"

let alloc_array_fragment (tys:list cty) (args:list expr) : ML expr =
  match tys, args with
  | [et], [knd; m; n; k; layout; size] ->
    call "KPR_INIT_ARR" [kpr_fragment et knd m n k layout; size] (TBuf auto_t)
  | _ -> fail "__alloc_array_fragment: unexpected shape"

(* A [strided_row_major]/[strided_col_major] dictionary: the offset bumps the
   pointer and the stride is wmma's leading dimension. *)
let strided (sl:expr) : ML (expr & expr) =
  (field ["offset"; "offset1"] 0 sl, field ["stride"; "stride1"] 1 sl)

let mem_row_major : expr = cname "wmma__mem_row_major" i32

let mma_load_accum (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [fr; sl; gm] ->
    let off, ldm = strided sl in
    call "wmma__load_matrix_sync" [fr; bufsub gm off; ldm; mem_row_major] TUnit
  | _ -> fail "mma_loadAccum: expected 3 arguments"

let mma_fill (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [_knd; _layout; fr; v] -> call "wmma__fill_fragment" [fr; v] TUnit
  | _ -> fail ("mma_fill: expected 4 arguments, got " ^ show (List.length args))

let mma_sync (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [_scal_ab; _scal_acc; _la; _lb; fa; fb; fc] ->
    call "wmma__mma_sync" [fc; fa; fb; fc] TUnit
  | _ -> fail ("mma_sync': expected 7 arguments, got " ^ show (List.length args))

(* Load an A/B fragment, applying an elementwise map.  The identity map is a
   plain load; otherwise the KPR_LOAD_MAP macro (include/kuiper/tensorcores.h)
   runs the register loop, and all we emit is the mapped value, i.e. the map
   applied to the macro-bound register [_kpr_in_v]. *)
let mma_load_map (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [fmap; fr; sl; gm] ->
    let off, ldm = strided sl in
    let gm = bufsub gm off in
    if returns_binder 1 0 fmap then
      call "wmma__load_matrix_sync" [fr; gm; ldm] TUnit
    else begin
      let et = pointee gm.ty in
      let in_v = mk (ECast (cname "_kpr_in_v" et, et)) et E_Pure in
      let mapped = apply fmap [in_v] et in
      call "KPR_LOAD_MAP" [gm; fr; ldm; mapped] TUnit
    end
  | _ -> fail "mma_load*_map: expected 4 arguments"

(* Store the accumulator, combining it elementwise with the resident tile.  A
   combine that returns the accumulator is a plain store; otherwise the
   KPR_STORE_COMB macro owns the read-modify-write, and all we emit is the
   combined value over the macro-bound [_kpr_new_v] and [_kpr_old_v]. *)
let mma_store_comb (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [gcomb; fr; sl; gm] ->
    let off, ldm = strided sl in
    let gm = bufsub gm off in
    if returns_binder 2 0 gcomb then
      seq (call "wmma__store_matrix_sync" [gm; fr; ldm; mem_row_major] TUnit)
          (call "__syncwarp" [] TUnit)
    else begin
      let et_c = pointee gm.ty in
      let bs, _ = lambda_parts gcomb in
      let et = match bs with b :: _ -> b.b_ty | [] -> et_c in
      let new_v = cname "_kpr_new_v" et in
      let old_v = cname "_kpr_old_v" et_c in
      let combined = apply gcomb [new_v; old_v] et_c in
      call "KPR_STORE_COMB" [gm; fr; ldm; combined] TUnit
    end
  | _ -> fail "mma_store_comb: expected 4 arguments"

(* Hopper WGMMA. *)
let wgmma_alloc_fragment (_tys:list cty) (_args:list expr) : ML expr =
  call "KPR_INIT" [cname "kpr_wgmma_fragment" i32] auto_t

let wgmma_fill (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [fr; x] -> call "kpr_wgmma_fill" [fr; x] TUnit
  | _ -> fail "WGMMA.fill: expected 2 arguments"

let wgmma_mem (op:string) (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [fr; sl; c] ->
    let off, stride = strided sl in
    call op [fr; bufsub c off; stride] TUnit
  | _ -> fail ("WGMMA " ^ op ^ ": expected 3 arguments")

let wgmma_mma_sync (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [a; b; fr] -> call "kpr_wgmma_mma_sync" [a; b; fr] TUnit
  | _ -> fail "WGMMA.mma_sync: expected 3 arguments"

(* -------------------------------------------------------------------- *)
(* Floats                                                               *)
(* -------------------------------------------------------------------- *)

let f32_lit (s:string) : ML expr =
  match float_lit_of_string s with
  | Some l -> mk (EConst (CFloat (l, Float32))) (TFloat Float32) E_Pure
  | None -> fail ("not a float literal: " ^ s)

let of_literal (conv:string) (t:cty) (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [s] ->
    (match (strip s).e with
     | EConst (CString v) -> pcall conv [f32_lit v] t
     | _ -> die "a string literal" s)
  | _ -> fail "of_literal: expected 1 argument"

(* -------------------------------------------------------------------- *)
(* Registration                                                         *)
(* -------------------------------------------------------------------- *)

let reg (l:string) (r:B.rule) : ML unit =
  B.register_rule (Ident.lid_of_str l) r

let prim (l:string) (n:int) (f:list cty -> list expr -> ML expr) : ML unit =
  reg l (B.Rule_prim (n, f))

(* A pure C function or macro of [n] arguments. *)
let cfun (l:string) (n:int) (c:string) (res:cty) : ML unit =
  prim l n (fun _ args -> pcall c args res)

(* An impure one. *)
let cproc (l:string) (n:int) (c:string) (res:cty) : ML unit =
  prim l n (fun _ args -> call c args res)

(* A C constant. *)
let cconst (l:string) (e:expr) : ML unit =
  prim l 0 (fun _ _ -> e)

let ty (l:string) (t:cty) : ML unit =
  reg l (B.Rule_type (fun _ -> t))

let f16 = "Kuiper.Float16.Base."
let bf16 = "Kuiper.BFloat16.Base."
let f32 = "Kuiper.Float32.Base."
let f64 = "Kuiper.Float64.Base."
let casts = "Kuiper.Float.Casts.Base."

(* The math functions, as (Kuiper name, arity, f16, bf16, f32, f64). *)
let math_fns : list (string & int & string & string & string & string) = [
  ("sqrt",  1, "kpr_hsqrt",  "kpr_bf16sqrt",  "sqrtf",  "sqrt");
  ("rsqrt", 1, "kpr_hrsqrt", "kpr_bf16rsqrt", "rsqrtf", "rsqrt");
  ("sin",   1, "kpr_hsin",   "kpr_bf16sin",   "sinf",   "sin");
  ("cos",   1, "kpr_hcos",   "kpr_bf16cos",   "cosf",   "cos");
  ("tan",   1, "kpr_htan",   "kpr_bf16tan",   "tanf",   "tan");
  ("asin",  1, "kpr_hasin",  "kpr_bf16asin",  "asinf",  "asin");
  ("acos",  1, "kpr_hacos",  "kpr_bf16acos",  "acosf",  "acos");
  ("atan",  1, "kpr_hatan",  "kpr_bf16atan",  "atanf",  "atan");
  ("sinh",  1, "kpr_hsinh",  "kpr_bf16sinh",  "sinhf",  "sinh");
  ("cosh",  1, "kpr_hcosh",  "kpr_bf16cosh",  "coshf",  "cosh");
  ("tanh",  1, "kpr_htanh",  "kpr_bf16tanh",  "tanhf",  "tanh");
  ("ceil",  1, "kpr_hceil",  "kpr_bf16ceil",  "ceilf",  "ceil");
  ("floor", 1, "kpr_hfloor", "kpr_bf16floor", "floorf", "floor");
  ("round", 1, "kpr_hround", "kpr_bf16round", "roundf", "round");
  ("fabs",  1, "kpr_hfabs",  "kpr_bf16fabs",  "fabsf",  "fabs");
  ("erf",   1, "kpr_herf",   "kpr_bf16erf",   "erff",   "erf");
  ("log2",  1, "kpr_hlog2",  "kpr_bf16log2",  "log2f",  "log2");
  ("log10", 1, "kpr_hlog10", "kpr_bf16log10", "log10f", "log10");
  ("exp2",  1, "kpr_hexp2",  "kpr_bf16exp2",  "exp2f",  "exp2");
  ("fexp",  1, "hexp",       "kpr_bf16exp",   "expf",   "exp");
  ("flog",  1, "hlog",       "kpr_bf16log",   "logf",   "log");
  ("fexpm1", 1, "kpr_hexpm1", "kpr_bf16expm1", "expm1f", "expm1");
  ("flog1p", 1, "kpr_hlog1p", "kpr_bf16log1p", "log1pf", "log1p");
  ("pow",   2, "kpr_hpow",   "kpr_bf16pow",   "powf",   "pow");
  ("atan2", 2, "kpr_hatan2", "kpr_bf16atan2", "atan2f", "atan2");
  ("fmin",  2, "kpr_hfmin",  "kpr_bf16fmin",  "fminf",  "fmin");
  ("fmax",  2, "kpr_hfmax",  "kpr_bf16fmax",  "fmaxf",  "fmax");
  ("fmod",  2, "kpr_hfmod",  "kpr_bf16fmod",  "fmodf",  "fmod");
  ("copysign", 2, "kpr_hcopysign", "kpr_bf16copysign", "copysignf", "copysign");
  ("fma",   3, "kpr_hfma",   "kpr_bf16fma",   "fmaf",   "fma");
]

let register_floats () : ML unit =
  let f32t = TFloat Float32 in
  let f64t = TFloat Float64 in
  List.iter (fun (nm, n, h, b, f, d) ->
    cfun (f16 ^ nm) n h half_t;
    cfun (bf16 ^ nm) n b bf16_t;
    cfun (f32 ^ nm) n f f32t;
    cfun (f64 ^ nm) n d f64t) math_fns;

  (* Half arithmetic goes through the intrinsics: operators on [half] depend
     on the CUDA version. *)
  cfun (f16 ^ "add") 2 "__hadd" half_t;
  cfun (f16 ^ "mul") 2 "__hmul" half_t;
  cfun (f16 ^ "sub") 2 "__hsub" half_t;
  cfun (f16 ^ "div") 2 "__hdiv" half_t;
  cfun (f16 ^ "eq")  2 "kpr_f16_eq"  tbool;
  cfun (f16 ^ "lt")  2 "kpr_f16_lt"  tbool;
  cfun (f16 ^ "lte") 2 "kpr_f16_lte" tbool;
  cfun (f16 ^ "bit_eq") 2 "kpr_f16_bit_eq" tbool;
  cfun (f16 ^ "of_int") 1 "__ll2half_rn" half_t;
  prim (f16 ^ "of_literal") 1 (of_literal "__float2half_rn" half_t);
  cconst (f16 ^ "zero") (pcall "__float2half_rn" [f32_lit "0.0"] half_t);
  cconst (f16 ^ "one")  (pcall "__float2half_rn" [f32_lit "1.0"] half_t);
  cconst (f16 ^ "largest")  (cname "HLF_MAX" half_t);
  cconst (f16 ^ "infinity") (cname "HLF_INFINITY" half_t);

  cfun (bf16 ^ "add") 2 "kpr_bf16add" bf16_t;
  cfun (bf16 ^ "mul") 2 "kpr_bf16mul" bf16_t;
  cfun (bf16 ^ "sub") 2 "kpr_bf16sub" bf16_t;
  cfun (bf16 ^ "div") 2 "kpr_bf16div" bf16_t;
  cfun (bf16 ^ "eq")  2 "kpr_bf16_eq"  tbool;
  cfun (bf16 ^ "lt")  2 "kpr_bf16_lt"  tbool;
  cfun (bf16 ^ "lte") 2 "kpr_bf16_lte" tbool;
  cfun (bf16 ^ "bit_eq") 2 "kpr_bf16_bit_eq" tbool;
  cfun (bf16 ^ "of_int") 1 "__ll2bfloat16_rn" bf16_t;
  prim (bf16 ^ "of_literal") 1 (of_literal "__float2bfloat16" bf16_t);
  cconst (bf16 ^ "zero") (pcall "__float2bfloat16" [f32_lit "0.0"] bf16_t);
  cconst (bf16 ^ "one")  (pcall "__float2bfloat16" [f32_lit "1.0"] bf16_t);
  cconst (bf16 ^ "largest")
    (pcall "__ushort_as_bfloat16"
       [mk (EConst (CInt (0x7F7F, Hex, Some (Unsigned, W16)))) (TInt (Unsigned, W16)) E_Pure]
       bf16_t);
  cconst (bf16 ^ "infinity") (pcall "__float2bfloat16" [cname "INFINITY" f32t] bf16_t);

  cconst (f32 ^ "largest")  (cname "FLT_MAX" f32t);
  cconst (f32 ^ "infinity") (cname "INFINITY" f32t);
  cconst (f64 ^ "largest")  (cname "DBL_MAX" f64t);
  cconst (f64 ^ "infinity") (cname "INFINITY" f64t);

  (* Bit equality compares representations, which no C operator does. *)
  cfun "FStar.Float32.bit_eq" 2 "kpr_f32_bit_eq" tbool;
  cfun "FStar.Float64.bit_eq" 2 "kpr_f64_bit_eq" tbool;

  (* GPU-only Float32 primitives with explicit rounding and flush-to-zero. *)
  cproc "Kuiper.Float32.add_rn_ftz" 2 "kpr_f32_add_rn_ftz" f32t;
  cproc "Kuiper.Float32.fma_rn_ftz" 3 "kpr_f32_fma_rn_ftz" f32t;
  cproc "Kuiper.Float32.mul_rn_ftz" 2 "kpr_f32_mul_rn_ftz" f32t;
  cproc "Kuiper.Float32.exp2_approx_ftz" 1 "kpr_f32_exp2_approx_ftz" f32t;
  cproc "Kuiper.Float32.rcp_approx_ftz" 1 "kpr_f32_rcp_approx_ftz" f32t;
  cproc "Kuiper.Float32.FastMath.sin" 1 "__sinf" f32t;
  cproc "Kuiper.Float32.FastMath.cos" 1 "__cosf" f32t;
  cproc "Kuiper.Float32.FastMath.log" 1 "__logf" f32t;
  cproc "Kuiper.Float32.FastMath.exp" 1 "__expf" f32t;
  cproc "Kuiper.Float32.FastMath.divide" 2 "__fdividef" f32t;
  cproc "Kuiper.Float32.FastMath.fma_rn" 3 "__fmaf_rn" f32t;
  cproc "Kuiper.Float32.FastMath.sub_rn" 2 "__fsub_rn" f32t;
  cproc "Kuiper.Float32.rsqrt_approx_ftz" 1 "kpr_f32_rsqrt_approx_ftz" f32t;

  (* Casts. *)
  let h2f (x:expr) = pcall "__half2float" [x] f32t in
  let b2f (x:expr) = pcall "__bfloat162float" [x] f32t in
  let f2h (x:expr) = pcall "__float2half_rn" [x] half_t in
  let f2b (x:expr) = pcall "__float2bfloat16" [x] bf16_t in
  let cast (t:cty) (x:expr) = mk (ECast (x, t)) t E_Pure in
  let reg_cast (nm:string) (f:expr -> ML expr) : ML unit =
    prim (casts ^ nm) 1 (fun _ args ->
      match args with
      | [x] -> f x
      | _ -> fail (nm ^ ": expected 1 argument")) in
  reg_cast "cast_f16_to_f32" h2f;
  reg_cast "cast_f16_to_f64" (fun x -> cast f64t (h2f x));
  reg_cast "cast_f32_to_f16" f2h;
  reg_cast "cast_f32_to_f64" (cast f64t);
  reg_cast "cast_bf16_to_f32" b2f;
  reg_cast "cast_f32_to_bf16" f2b;
  reg_cast "cast_f16_to_bf16" (fun x -> f2b (h2f x));
  reg_cast "cast_bf16_to_f16" (fun x -> f2h (b2f x));
  reg_cast "cast_bf16_to_f64" (fun x -> cast f64t (b2f x));
  reg_cast "cast_f64_to_bf16" (fun x -> f2b (cast f32t x));
  reg_cast "cast_f64_to_f16" (fun x -> f2h (cast f32t x));
  reg_cast "cast_f64_to_f32" (cast f32t)

let assert_like (macro:string) (_tys:list cty) (args:list expr) : ML expr =
  match args with
  | [b] -> if is_const_true b then unit_expr else call macro [b] TUnit
  | _ -> fail (macro ^ ": expected 1 argument")

(* [fst]/[snd].  Kernels sometimes rebuild the tuple of shared memory arrays
   and project out of it; without this the pair (ending in an [int]) would be
   built at runtime.  Reduce a projection of a literal pair, and compile any
   other to a match, as Custard would. *)
let tuple_proj (i:int) (tys:list cty) (args:list expr) : ML expr =
  match args with
  | [x] ->
    let t1, t2 =
      match tys, tuple_components x.ty with
      | [a; b], _ -> (a, b)
      | _, Some (a, b) -> (a, b)
      | _ -> fail "fst/snd: cannot determine the component types" in
    let res = if i = 0 then t1 else t2 in
    (match (strip x).e with
     | ECtor (_, [a; b]) | ETuple [a; b] when a.eff = E_Pure && b.eff = E_Pure ->
       if i = 0 then a else b
     | _ ->
       let v = fresh "p" in
       let mktup2 = { ns = ["FStar"; "Pervasives"; "Native"]; id = "Mktuple2"; spec = None } in
       let p = if i = 0 then PCtor (mktup2, [PVar v; PWild]) else PCtor (mktup2, [PWild; PVar v]) in
       mk (EMatch (x, [(p, None, mk (EVar v) res E_Pure)])) res x.eff)
  | _ -> fail "fst/snd: expected 1 argument"

let _ =
  (* Types *)
  ty (f16 ^ "t") half_t;
  ty (bf16 ^ "t") bf16_t;
  ty "Kuiper.Kernel.Stream.stream_t" stream_t;
  ty "Kuiper.TensorCore.Base.fragment" auto_t;
  ty "Kuiper.TensorCore.WGMMA.fragment" auto_t;

  (* Assertions *)
  prim "Kuiper.Assert.dassert" 1 (assert_like "KPR_ASSERT");
  prim "Kuiper.Assert.dguard" 1 (assert_like "KPR_GUARD");

  (* Predefined variables; fixup.sed turns [_x] into [.x]. *)
  prim "Kuiper.Base.get_gdim" 1 (fun _ _ -> cname "gridDim_x" usize);
  prim "Kuiper.Base.get_bdim" 1 (fun _ _ -> cname "blockDim_x" usize);
  prim "Kuiper.Base.get_bid" 1 (fun _ _ -> cname "blockIdx_x" usize);
  prim "Kuiper.Base.get_tid" 1 (fun _ _ -> cname "threadIdx_x" usize);

  (* Barriers *)
  prim "Kuiper.Barrier.barrier_wait" 1 (fun _ _ -> call "__syncthreads" [] TUnit);
  prim "Kuiper.Barrier.Warp.warp_barrier_wait" 1 (fun _ _ -> call "__syncwarp" [] TUnit);

  (* Tensor cores *)
  prim "Kuiper.TensorCore.Base.__alloc_fragment" 5 alloc_fragment;
  prim "Kuiper.TensorCore.Base.__alloc_array_fragment" 6 alloc_array_fragment;
  prim "Kuiper.TensorCore.Base.mma_loadAccum" 3 mma_load_accum;
  prim "Kuiper.TensorCore.Base.mma_fill" 4 mma_fill;
  prim "Kuiper.TensorCore.Base.mma_sync'" 7 mma_sync;
  prim "Kuiper.TensorCore.Base.mma_loadA_map" 4 mma_load_map;
  prim "Kuiper.TensorCore.Base.mma_loadB_map" 4 mma_load_map;
  prim "Kuiper.TensorCore.Base.mma_loadA_map_cm" 4 mma_load_map;
  prim "Kuiper.TensorCore.Base.mma_loadB_map_cm" 4 mma_load_map;
  prim "Kuiper.TensorCore.Base.mma_store_comb" 4 mma_store_comb;
  prim "Kuiper.TensorCore.WGMMA.alloc_fragment" 1 wgmma_alloc_fragment;
  prim "Kuiper.TensorCore.WGMMA.fill" 2 wgmma_fill;
  prim "Kuiper.TensorCore.WGMMA.load_accum" 3 (wgmma_mem "kpr_wgmma_load_accum");
  prim "Kuiper.TensorCore.WGMMA.store" 3 (wgmma_mem "kpr_wgmma_store");
  prim "Kuiper.TensorCore.WGMMA.mma_sync" 3 wgmma_mma_sync;

  (* Floats *)
  register_floats ();

  (* References and arrays *)
  prim "Kuiper.Ref.memcpy_host_to_device" 3 (ref_memcpy "cudaMemcpyHostToDevice");
  prim "Kuiper.Ref.memcpy_device_to_host" 3 (ref_memcpy "cudaMemcpyDeviceToHost");
  prim "Kuiper.Ref.memcpy_device_to_device" 3 (ref_memcpy "cudaMemcpyDeviceToDevice");
  prim "Kuiper.Array.Core.memcpy_host_to_device" 4 (arr_memcpy "cudaMemcpyHostToDevice");
  prim "Kuiper.Array.Core.memcpy_device_to_host" 4 (arr_memcpy "cudaMemcpyDeviceToHost");
  prim "Kuiper.Array.Core.memcpy_device_to_device" 4 (arr_memcpy "cudaMemcpyDeviceToDevice");
  prim "Kuiper.Array.Core.memcpy_host_to_device'" 6 (arr_memcpy' "cudaMemcpyHostToDevice");
  prim "Kuiper.Array.Core.memcpy_device_to_host'" 6 (arr_memcpy' "cudaMemcpyDeviceToHost");
  prim "Kuiper.Array.Core.memcpy_device_to_device'" 6 (arr_memcpy' "cudaMemcpyDeviceToDevice");
  prim "Kuiper.Array.Core.gpu_array_alloc" 2 gpu_array_alloc;
  prim "Kuiper.Array.Core.gpu_array_free" 1 gpu_array_free;
  prim "Kuiper.Array.Core.slice_read" 2 slice_read;
  prim "Kuiper.Array.Core.slice_write" 3 slice_write;
  prim "Kuiper.Array.Core.get_ref_of_array_cell" 2 get_ref_of_array_cell;
  prim "Kuiper.Array.Vectorized.array_vec_cpy" 6 array_vec_cpy;

  (* Atomics *)
  List.iter (fun (s, t) ->
    cproc ("Kuiper.AtomicOps.gpu_faa_" ^ s) 2 ("atomic_add_" ^ s) t)
    [("u32", u32); ("u64", TInt (Unsigned, W64));
     ("f32", TFloat Float32); ("f64", TFloat Float64)];

  (* Kernels and streams *)
  prim "Kuiper.Kernel.Base.launch_kernel_full" 2 launch;
  prim "Kuiper.Kernel.Stream.fresh_stream" 1 (fun _ _ -> call "KPR_FRESH_STREAM" [] stream_t);
  prim "Kuiper.Kernel.Stream.destroy_stream" 1 (fun _ args ->
    must (call "cudaStreamDestroy" args i32));
  prim "Kuiper.Kernel.Base.sync_stream" 1 (fun _ args ->
    must (call "cudaStreamSynchronize" args i32));
  prim "Kuiper.Kernel.Base.sync_device" 1 (fun _ _ ->
    must (call "cudaDeviceSynchronize" [] i32));

  (* SizeT *)
  prim "Kuiper.SizeT.sizet_and" 2 (fun _ args ->
    match args with
    | [x; y] -> uop BAnd x y
    | _ -> fail "sizet_and: expected 2 arguments");
  prim "Kuiper.SizeT.sizet_to_u32" 1 (fun _ args ->
    match args with
    | [x] -> mk (ECast (x, u32)) u32 E_Pure
    | _ -> fail "sizet_to_u32: expected 1 argument");

  (* Pairs *)
  prim "FStar.Pervasives.Native.fst" 1 (tuple_proj 0);
  prim "FStar.Pervasives.Native.snd" 1 (tuple_proj 1);

  (* For loops *)
  prim "Kuiper.For.for_loop" 3 for_loop;
  prim "Kuiper.For.for_loop'" 3 for_loop
