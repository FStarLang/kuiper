module ExtractKuiper

friend FStarC.Extraction.Krml
open FStarC.Extraction.Krml

open FStar
open FStarC
open FStarC.Effect
open FStar.List.Tot
open FStarC.Extraction
open FStarC.Extraction.ML
open FStarC.Extraction.ML.Syntax
open FStarC.Const
open FStarC.Errors
open FStarC.Pprint
open FStarC.Class.Show
open FStarC.Class.PP
open FStarC.Class.Tagged

instance _ : tagged mlexpr' = {
  tag_of = (function
    | MLE_Const  .. -> "MLE_Const"
    | MLE_Var    .. -> "MLE_Var"
    | MLE_Name   .. -> "MLE_Name"
    | MLE_Let    .. -> "MLE_Let"
    | MLE_App    .. -> "MLE_App"
    | MLE_TApp   .. -> "MLE_TApp"
    | MLE_Fun    .. -> "MLE_Fun"
    | MLE_Match  .. -> "MLE_Match"
    | MLE_Coerce .. -> "MLE_Coerce"
    | MLE_CTor   .. -> "MLE_CTor"
    | MLE_Seq    .. -> "MLE_Seq"
    | MLE_Tuple  .. -> "MLE_Tuple"
    | MLE_Record .. -> "MLE_Record"
    | MLE_Proj   .. -> "MLE_Proj"
    | MLE_If     .. -> "MLE_If"
    | MLE_Raise  .. -> "MLE_Raise"
    | MLE_Try    .. -> "MLE_Try"
  );
}
instance _ : tagged mlexpr = {
  tag_of = (fun t -> tag_of t.expr);
}

open ExtractionUtils

let mlloc_to_range (l : mlloc) : Range.range =
  let (line, file) = l in
  let p = Range.mk_pos line 0 in
  Range.mk_range file p p

let zero_for_deref = EQualified (["C"], "_zero_for_deref")

let deref e = EBufRead (e, zero_for_deref)

let dbg = Debug.get_toggle "kuiper"

let rec drop n (lst : list 'a) : list 'a =
  match lst with
  | [] -> []
  | _ when n <= 0 -> lst
  | _ -> List.tl (drop (n - 1) lst)


let kpr_translate_type_without_decay : translate_type_without_decay_t = fun env t ->
  let cb = translate_type_without_decay env in
  let x = type_hta t in
  if None? x then raise NotSupportedByKrmlExtension;
  match Some?.v x with
  | "Kuiper.TensorCore.Base.fragment", [et; knd; m; n; k; layout] ->
    // Note: it is difficult to try to construct a proper type
    // here because
    // 1) the indices are already erased to unit, and we cannot
    //    recover them here
    // 2) if we want to generate the templated types, i.e.
    //    something like wmma::fragment<half, 16, 16, 16, wmma::row_major>,
    //    we cannot just TQualified as karamel will sanitize the string,
    //    and karamel also has no notion of templated types. Though this
    //    seems not too difficult to add, it's probably easier to just
    //    define macros like kpr_fragment_half_16_16_16_row_major that
    //    expand to the proper templated type.
    TQualified ([], "auto_AMP") // sed subtitutes this to auto&

  | "Kuiper.Float16.Base.t",               [] -> TInt Half
  | "Kuiper.BFloat16.Base.t",              [] -> TInt BFloat16
  | "Kuiper.Float32.Base.t",               [] -> TInt Float
  | "Kuiper.Float64.Base.t",               [] -> TInt Double
  | _ -> raise NotSupportedByKrmlExtension

let cudaMemcpyDeviceToHost = EQualified ([], "cudaMemcpyDeviceToHost")
let cudaMemcpyHostToDevice = EQualified ([], "cudaMemcpyHostToDevice")
let cudaMemcpyDeviceToDevice = EQualified ([], "cudaMemcpyDeviceToDevice")

let get_record_field fname (e:mlexpr) : ML mlexpr =
  let assoc' k v =
    match List.assoc k v with
    | Some r -> r
    | None ->
      raise_error (mlloc_to_range e.loc) Fatal_ExtractionUnsupported [
        text "get_record_field: field not found:" ^/^ text k ^/^ pp v
      ]
  in
  match (unmagic e).expr with
  | MLE_Record (_, _, flds) -> assoc' fname flds
  | _ ->
    raise_error (mlloc_to_range e.loc) Fatal_ExtractionUnsupported [
      text "Expected a record for the size, got:" ^/^ pp e
    ]

let get_sizet (e : mlexpr) : ML mlexpr = get_record_field "size" e

(* Generate a sizeof expression for a type. A bit hacky, since it generates
   something like sizeof((ty)0). The way to avoid the zero would to be add
   a ESizeof to krml. Note: sizeof((int)) fails, so we need something in there. *)
let sizeof (ty : typ) : expr = ESizeof ty

(* e times sz, where sz is a sizeof *)
let mul_by_sz (sz e : expr) : expr =
  (* Note the cast below to please krml, since ESizeof t is always SizeT.
  The real one, not the fake one. *)
  let sz = ECast (sz, TInt fake_SizeT) in
  EApp (EOp (Mult, fake_SizeT), [ sz; e ])

// repeated names get a numeric prefix, and we have both strided_row_major and strided_col_major
let get_strided_row_major_offset (e : mlexpr) : ML mlexpr =
  try get_record_field "offset" e with | _ -> get_record_field "offset1" e
let get_strided_row_major_stride (e : mlexpr) : ML mlexpr =
  try get_record_field "stride" e with | _ -> get_record_field "stride1" e

let _MUST (e : expr) : expr =
    EApp (EQualified ([], "MUST"), [e])

let _mlMUST (e : mlexpr) : mlexpr =
    with_ty ml_unit_ty <|
      MLE_App (with_ty ml_unit_ty <| MLE_Name ([], "MUST"), [e])

let ctr = mk_ref 0
let last_name : ref mlident = mk_ref "<bogus>"

let extra_unit_binder = {mlbinder_name = "extra_unit"; mlbinder_ty = ml_unit_ty; mlbinder_attrs = []}

let rec takeWhile f xs =
  match xs with
  | [] -> ([], [])
  | x::xs' ->
    if f x then
      let (ys, zs) = takeWhile f xs' in
      (x::ys, zs)
    else
      ([], xs)

let get_one_binder (e:mlexpr) : ML (mlbinder & mlexpr) =
  match e.expr with
  | MLE_Fun ([b], body) -> b, body
  | MLE_Fun (b::bs, body) ->
    b, { e with expr = MLE_Fun (bs, body) } (* type is wrong, but it doesn't matter *)
  | _ ->
    raise_error0 Fatal_ExtractionUnsupported [
      text "Expected a single binder in a lambda, got: " ^/^ pp e
    ]

let remove_trailing_units (e : mlexpr) : mlexpr =
  let hd, args = head_and_args e in
  let units, non_units = List.rev args |> takeWhile (fun a -> match a.expr with
                              | MLE_Const MLC_Unit -> true
                              | _ -> false) in
  { e with expr = MLE_App (hd, List.rev non_units ) }

let mkarr t1 t2 = MLTY_Fun (t1, E_IMPURE, t2)

let eta (e : mlexpr) : mlexpr =
  let e' = with_ty (mkarr ml_unit_ty e.mlty) <| MLE_Fun ([extra_unit_binder], e) in
  let e'' = with_ty e.mlty <| MLE_App (e', [ml_unit]) in
  e''

(* Registry mapping record type paths to their field definitions.
   Populated by kpr_translate_type_decl when record type declarations are encountered. *)
let record_fields_registry : ref (list (string & list (mlsymbol & mlty))) = mk_ref []

(* Look up record fields for a named type. First checks the registry (populated
   by translate_type_decl), then falls back to reconstructing from the F* TC
   environment. The fallback is needed because some record types (e.g. those
   marked noextract, or in modules where the type decl is dropped) never reach
   translate_type_decl. *)
let lookup_record_fields (g : env) (mpath : mlpath) : ML (option (list (mlsymbol & mlty))) =
  match List.assoc (string_of_mlpath mpath) !record_fields_registry with
  | Some fields -> Some fields
  | None ->
    (* Fallback: reconstruct from the TC environment *)
    let open FStarC.Ident in
    let open FStarC.Syntax.Syntax in
    let lid = lid_of_path (fst mpath @ [snd mpath]) FStarC.Range.dummyRange in
    let tcenv = UEnv.tcenv_of_uenv g.uenv in
    match FStarC.TypeChecker.Env.lookup_sigelt tcenv lid with
    | Some se -> (
      match List.tryFind (function RecordType _ -> true | _ -> false) se.sigquals with
      | Some (RecordType (_, field_ids)) -> (
        let (_, ctor_lids) = FStarC.TypeChecker.Env.datacons_of_typ tcenv lid in
        match ctor_lids with
        | [ctor_lid] ->
          let (_, ctor_typ) = FStarC.TypeChecker.Env.lookup_datacon tcenv ctor_lid in
          let mlt = ML.Term.term_as_mlty g.uenv ctor_typ in
          let mlt = ML.Util.eraseTypeDeep (ML.Util.udelta_unfold g.uenv) mlt in
          let field_tys = ML.Util.argTypes mlt in
          if List.length field_ids = List.length field_tys
          then (
            let fields = List.map2 (fun id ty ->
              (string_of_id id, ty)) field_ids field_tys in
            (* Cache for future lookups *)
            record_fields_registry := (string_of_mlpath mpath, fields) :: !record_fields_registry;
            Some fields
          ) else None
        | _ -> None
      )
      | _ -> None
    )
    | None -> None

(* Explode record-typed free variables into individual field variables.
   Returns (new_fvs, substituted_body, call_args). *)
let rec explode_fvs (g : env) (fvs : list (string & mlty)) (e : mlexpr) :
    ML (list (string & mlty) & mlexpr & list mlexpr) =
  match fvs with
  | [] -> ([], e, [])
  | (v, t) :: rest ->
    match t with
    | MLTY_Named ([], path) -> (
      match lookup_record_fields g path with
      | Some fields ->
        if !dbg then
          Format.print2 "KPR hoist: exploding %s (type %s) into fields\n" v (string_of_mlpath path);
        let new_fvs = List.map (fun (f, ft) -> (v ^ "__" ^ f, ft)) fields in
        let record_expr = with_ty t <| MLE_Record (fst path, snd path,
          List.map (fun (f, ft) -> (f, with_ty ft <| MLE_Var (v ^ "__" ^ f))) fields) in
        let e' = ml_subst e v record_expr in
        let proj_args = List.map (fun (f, ft) ->
          with_ty ft <| MLE_Proj (with_ty t <| MLE_Var v, (fst path, f))) fields in
        let (rest_fvs, e'', rest_args) = explode_fvs g rest e' in
        (new_fvs @ rest_fvs, e'', proj_args @ rest_args)
      | None ->
        if !dbg then
          Format.print2 "KPR hoist: fv %s has named type %s but NOT in record registry\n" v (string_of_mlpath path);
        let (rest_fvs, e', rest_args) = explode_fvs g rest e in
        ((v, t) :: rest_fvs, e', (with_ty t <| MLE_Var v) :: rest_args)
    )
    | _ ->
      let (rest_fvs, e', rest_args) = explode_fvs g rest e in
      ((v, t) :: rest_fvs, e', (with_ty t <| MLE_Var v) :: rest_args)

let hoist (g : env) (e : mlexpr) : ML mlexpr =
  let e0 = e in
  // let e = remove_trailing_units e in // ???
  let e = eta e in
  let et = e.mlty in
  let fvs0 = freevars_of_mlexpr e |> List.unique in
  (* get the fvs in the order they appear in the environment (mind
     the rev-- the env head is the closest binder). This makes the
     order much more predictable and conceptually nicer. *)
  let fvs =
    g.names |> List.rev |> List.collect (fun n ->
      match List.tryFind (fun (v, _) -> v = n.pretty) fvs0 with
      | Some (v, t) -> [(v, t)]
      | None -> []
    )
  in
  if List.length fvs <> List.length fvs0
  then
    raise_error0 Fatal_ExtractionUnsupported [
      text "Hoist: free variables do not match:" ^/^ pp e0;
      text "fvs0 =" ^/^ pp fvs0;
      text "fvs =" ^/^ pp fvs;
    ];
  // Format.print1_warning "fvs = %s\n" (show fvs);
  (* Explode record-typed free variables into individual fields *)
  // Disabled for now... we're not relying on it anymore
  (* let (fvs, e, call_args) = explode_fvs g fvs e in *)
  let call_args = List.map (fun (v, t) -> with_ty t <| MLE_Var v) fvs in
  let e = collapse_record_proj e in
  let mk_binder (v, t) = {mlbinder_name = v; mlbinder_ty = t; mlbinder_attrs = []} in

  if None? !krml_current_decl then
    raise_error0 Fatal_ExtractionUnsupported [
      text "Hoist: internal error: no current declaration in context for:" ^/^ pp e0
    ];

  (* Get a fresh stable name *)
  if Some?.v !krml_current_decl <> !last_name then (
    last_name := Some?.v !krml_current_decl;
    ctr := 0
  ) else
    ctr := !ctr + 1;

  let fresh = "__hoisted_" ^ !last_name ^ "_" ^ string_of_int !ctr in

  let bs =  List.map mk_binder fvs @ [extra_unit_binder] in
  let kbs = translate_binders g bs in
  let g0 = { g with names = []; names_t = [] } in
  let lambda = translate_expr (add_binders g0 bs) e in
  let flags = [
    Krml.Comment ("  hoisted when extracting " ^ Option.dflt "<unknown>" !krml_current_decl);
    Krml.Private;
    Krml.Prologue "__global__";
  ]
  in
  let decl = DFunction (None, flags, 0, translate_type (add_binders g0 bs) et, ([], fresh), kbs, lambda) in
  translate_decl_accum := !translate_decl_accum @ [decl];

  let nm = with_ty ml_unit_ty <| MLE_Name ([], fresh) in
  let call = MLE_App (nm, call_args) in
  let e' = {e with expr = call} in
  if !dbg then (
    Format.print3_warning
      "Hoisted %s into %s, creating the declaration\n%s\n" (mlexpr_to_string e) fresh (show decl);
    Format.print1_warning "The translated expression is %s\n" (mlexpr_to_string e')
  );
  e'

(* Returns (possibly) the size of each element and length of array.
The type of the array has already been erased, we cannot get it here. *)
let parse_shmem_desc (e : mlexpr) : ML (option (mlexpr & mlexpr)) =
  let open FStarC.Class.Monad in
  match e.expr with
  | MLE_CTor (fv, [_ty; sized; len]) when string_of_mlpath fv = "Kuiper.SHMem.SHArray" ->
    (* Note: we cannot use sizeof of the type here, since it has
    already been erased into unit. *)
    let sized_a = get_sizet sized in
    return (sized_a, len)
  | _ ->
    Format.print1 "Not a shmem_desc: %s\n" (mlexpr_to_string e);
    None

let extract_kcall (cb : mlexpr -> ML expr) (env : Krml.env) (kdesc : mlexpr) : ML (option expr) =
  let open FStarC.Class.Monad in
  let assoc' k v =
    match List.assoc k v with
    | Some r -> r
    | None ->
      raise_error (mlloc_to_range kdesc.loc) Fatal_ExtractionUnsupported [
        text "launch_kernel: field not found:" ^/^ text k
      ]
  in
  let! nblk, nthr, smem_bytesz, hd, rest_args =
    match kdesc.expr with
    | MLE_Record (_, _, fields) ->
      let nblk = assoc' "nblk" fields in
      let nthr = assoc' "nthr" fields in
      let shmems_desc = assoc' "shmems_desc" fields in
      (* Returns list of shared memory arrays declared.
          For each one: type, element size in bytes, and number of elements, all
          as ML terms. *)
      // Format.print1 "GG mlexpr shmems: %s\n" (show shmems_desc);
      let! parsed : list mlexpr = mlexpr_as_list shmems_desc in
      let! parsed : list (mlexpr & mlexpr) = mapM parse_shmem_desc parsed in
      // Format.print1 "GG mlexpr shmems parsed: %s\n" (show parsed);
      let ml_shmem : mlexpr =
        MLE_Name ([], "KPR_SHMEM")
        |> with_ty (ml_fun ml_unit_ty ml_bytearr)
        |> (fun x -> MLE_App (x, [ml_unit]))
        |> with_ty ml_bytearr
      in
      let ml_shmem_at (e : mlexpr) : mlexpr =
        with_ty ml_bytearr <|
        MLE_App (
          (MLE_Name ([], "KPR_SHMEM_AT")
          |> with_ty (ml_fun ml_unit_ty ml_bytearr)),
          [e])
        // |> (fun x -> MLE_App (x, [ml_unit]))
      in
      (* returns the tuple + total size *)
      let mk_c_sh (desc : list (mlexpr & mlexpr)) : ML (mlexpr & mlexpr) =
        let rec aux (off : mlexpr) (desc : list (mlexpr & mlexpr)) : ML (mlexpr & mlexpr) =
          match desc with
          | [] -> intlit 123, off
          | (e_sz, len) :: desc' ->
            let off' = sizet_add off (sizet_mul e_sz len) in
            // let this_one =
            //   let nm = ["FStar"; "Buffer"], "sub" in
            //   let args = [ml_shmem; off; ml_unit] in // need the ml unit so krml rule kicks in
            //   with_ty ml_bytearr <|
            //   MLE_App (with_ty MLTY_Top (MLE_TApp (with_ty MLTY_Top (MLE_Name nm), [ml_uint8])),
            //            args)
            // in
            let this_one = ml_shmem_at off in
            let rest, sz = aux off' desc' in
            mk_tuple2 this_one rest, sz
        in
        aux (sizet_lit 0) desc
      in
      let c_sh, shmem_bytesz = mk_c_sh parsed in
      let kf = assoc' "f" fields in
      // let rec drop_n_binders (e:mlexpr) n =
      //   match e.expr with
      //   | MLE_Fun (bs, body) when List.length bs = n -> body
      //   | MLE_Fun (bs, body) when List.length bs > n ->
      //     let bs = drop n bs in
      //     { e with expr = MLE_Fun (bs, body) }
      //   | MLE_Fun (bs, body) when List.length bs < n ->
      //     drop_n_binders body (n - List.length bs)
      //   | _ -> failwith ("launch_kernel: not enough binders: " ^ show e)
      // in
      // let rec drop_last_n_args (e:mlexpr) n =
      //   match e.expr with
      //   | MLE_App (head, args) when List.length args = n -> head
      //   | MLE_App (head, args) when List.length args > n ->
      //     let args = drop n args in
      //     { e with expr = MLE_App (head, args) }
      //   | MLE_App (head, args) when List.length args < n ->
      //     drop_last_n_args head (n - List.length args)
      //   | _ -> failwith ("launch_kernel: not enough arguments: " ^ show e)
      // in
      let apply_lam (f : mlexpr) (v : mlexpr) : ML mlexpr =
        let b, body = get_one_binder f in
        ml_subst body b.mlbinder_name v
      in
      let ml_blockidx : mlexpr =
        MLE_Name ([], "blockIdx_x")
        |> with_ty (MLTY_Var "FStar.SizeT.t")
      in
      let ml_threadidx : mlexpr =
        MLE_Name ([], "threadIdx_x")
        |> with_ty (MLTY_Var "FStar.SizeT.t") // fixme should be MLTY_Named
      in
      let kf = apply_lam kf c_sh in
      // let kf = apply_lam kf ml_unit in
      let kf = apply_lam kf ml_blockidx in
      let kf = apply_lam kf ml_threadidx in
      let kf = apply_lam kf ml_unit in
      let kf = collapse_tuple_proj kf in
      let kf = collapse_tuple_matches kf in
      let kf = hoist env kf in
      let hd, rest_args = head_and_args kf in
      return (nblk, nthr, shmem_bytesz, hd, rest_args)

    | _ ->
      raise_error (mlloc_to_range kdesc.loc) Fatal_ExtractionUnsupported [
        text "launch_kernel: not a record:" ^/^ pp kdesc
      ]
  in
  let smem_bytesz = cb smem_bytesz in
  let shmem_is_nonzero : bool =
    match smem_bytesz with
    | EConstant (_w, "0") -> false
    | _ -> true
  in
  let assert_shmem_size : expr =
    (* If smem_bytesz is zero, skip the check. Could later find a bigger
    range on which to skip. *)
    if shmem_is_nonzero then
      EApp (EQualified ([], "KPR_SHMEM_FITS"), [ smem_bytesz ])
    else
      EUnit
  in
  let shmem_setup : expr =
    if shmem_is_nonzero then
      let kk : expr = EQualified ([], "cudaFuncSetAttribute") in
      let aa : expr = EQualified ([], "cudaFuncAttributeMaxDynamicSharedMemorySize") in
      _MUST <|
        EApp (kk, [ cb hd; aa; smem_bytesz ])
    else
      EUnit
  in
  let e' =
    let kcall : expr = EQualified ([], "KPR_KCALL") in
    EApp (kcall, [ cb hd; cb nblk; cb nthr; smem_bytesz ] @ List.map cb rest_args)
  in
  // Format.print1_warning "New kcall: %s\n" (show e');
  return <|
    ESequence [assert_shmem_size; shmem_setup; e']

let kpr_translate_alloc_fragment (cb : mlexpr -> ML expr) et knd m n k layout =
    let knd =
      match cta knd with
      | Some ("Kuiper.TensorCore.Base.FragA",     [], []) -> EQualified ([], "wmma::matrix_a")
      | Some ("Kuiper.TensorCore.Base.FragB",     [], []) -> EQualified ([], "wmma::matrix_b")
      | Some ("Kuiper.TensorCore.Base.FragAcc",   [], []) -> EQualified ([], "wmma::accumulator")
      | x -> raise_error (mlloc_to_range knd.loc) Fatal_ExtractionUnsupported [
        text "unexpected knd in __alloc_fragment:" ^/^ pp knd
      ]
    in
    let layout : option expr =
      match cta layout with
      | Some ("Kuiper.TensorCore.Base.FragLRM",    [], []) -> Some <| EQualified ([], "wmma::row_major")
      | Some ("Kuiper.TensorCore.Base.FragLCM",    [], []) -> Some <| EQualified ([], "wmma::col_major")
      | Some ("Kuiper.TensorCore.Base.FragLAcc",   [], []) -> None
      | x -> raise_error (mlloc_to_range layout.loc) Fatal_ExtractionUnsupported [
        text "unexpected layout in __alloc_fragment:" ^/^ pp layout
      ]
    in
    let faketype =
      match et with
      | MLTY_Named ([], (["Kuiper"; "Float16"; "Base"], "t")) -> EQualified ([], "half")
      | MLTY_Named ([], (["Kuiper"; "BFloat16"; "Base"], "t")) -> EQualified ([], "__nv_bfloat16")
      | MLTY_Named ([], (["Kuiper"; "Float32"; "Base"], "t")) -> EQualified ([], "float")
      | MLTY_Named ([], (["Kuiper"; "Float64"; "Base"], "t")) -> EQualified ([], "double")
    in
    let args =
      [ knd; cb m; cb n; cb k; faketype ]
      @ (match layout with | Some l -> [l] | None -> [])
    in
      EApp (EQualified ([], "kpr_fragment"), args)

let kpr_translate_expr : translate_expr_t = fun env e ->
  let e = flatten_app e in
  if !dbg
  then Format.print1_warning "ExtractKuiper.kpr_translate_expr %s\n" (mlexpr_to_string e);
  let cb = translate_expr env in
  let cb_ty = translate_type env in
  let x = hta e in
  if None? x then raise NotSupportedByKrmlExtension;
  match Some?.v x with
  (******** ASSERTIONS ********)
  | "Kuiper.Assert.dassert", [], [ x ] ->
    if x.expr = MLE_Const (MLC_Bool true) then
      EUnit
    else (
      if x.expr = MLE_Const (MLC_Bool false) then
        log_issue0 Warning_DeprecatedGeneric [
          text "Emitting a 'false' assert in function " ^/^ pp !krml_current_decl;
        ];
      EApp (EQualified ([], "KPR_ASSERT"), [ cb x ])
    )

  | "Kuiper.Assert.dguard", [], [ x ] ->
    if x.expr = MLE_Const (MLC_Bool true) then
      EUnit
    else (
      if x.expr = MLE_Const (MLC_Bool false) then
        log_issue0 Warning_DeprecatedGeneric [
          text "Emitting a 'false' guard in function " ^/^ pp !krml_current_decl;
        ];
      EApp (EQualified ([], "KPR_GUARD"), [ cb x ])
    )

  (******** PREDEFINED VARS ********)

  | "Kuiper.Base.get_gdim", [], [ _unit; _erasednblk; _erasednbid ] ->
    EQualified ([], "gridDim_x")

  | "Kuiper.Base.get_bdim", [], [ _unit; _erasednthr; _erasedntid ] ->
    EQualified ([], "blockDim_x")

  (******** BARRIERS ********)

  | "Kuiper.Barrier.barrier_wait", [], [ _unit; _n; _contract; _it; _tid ] ->
    EApp (EQualified ([], "__syncthreads"), [ EUnit ])

  (******** TENSOR CORE OPERATIONS, FRAGMENTS, ETC ********)

  | "Kuiper.TensorCore.Base.__alloc_array_fragment", [et], [ knd; m; n; k; layout; size ] ->
      EApp (EQualified ([], "KPR_INIT_ARR"),
        [kpr_translate_alloc_fragment cb et knd m n k layout; cb size])

  | "Kuiper.TensorCore.Base.__alloc_fragment", [et], [ knd; m; n; k; layout ] ->
      EApp (EQualified ([], "KPR_INIT"),
        [kpr_translate_alloc_fragment cb et knd m n k layout])

  | "Kuiper.TensorCore.Base.mma_loadA", [et], [ m; n; k; fr; l; strided_l; gm; f; m0; f0 ]
  | "Kuiper.TensorCore.Base.mma_loadA_cm", [et], [ m; n; k; fr; l; strided_l; gm; f; m0; f0 ]
  | "Kuiper.TensorCore.Base.mma_loadB", [et], [ m; n; k; fr; l; strided_l; gm; f; m0; f0 ] ->
    let fr = cb fr in
    let ldm = cb <| get_strided_row_major_stride strided_l in
    let offset = cb <| get_strided_row_major_offset strided_l in
    // Note: use of EBufSub relies on the type of gm unfolding to an array.
    // If IArray/VArray/any other layer defines a new inductive, karamel will complain.
    let gm = EBufSub (cb gm, offset) in
    EApp (EQualified ([], "wmma::load_matrix_sync"), [ fr; gm; ldm ])

  | "Kuiper.TensorCore.Base.mma_loadAccum", [et], [m; n; k; fr; l; strided_l; gm; f; m0; f0 ] ->
    // TODO remove duplicated code
    let fr = cb fr in
    let layout = EQualified ([], "wmma::mem_row_major") in // FAKE the API only supports this one for now
    let ldm = cb <| get_strided_row_major_stride strided_l in
    let offset = cb <| get_strided_row_major_offset strided_l in
    // Note: use of EBufSub relies on the type of gm unfolding to an array.
    // If IArray/VArray/any other layer defines a new inductive, karamel will complain.
    let gm = EBufSub (cb gm, offset) in
    EApp (EQualified ([], "wmma::load_matrix_sync"), [ fr; gm; ldm; layout ])

  | "Kuiper.TensorCore.Base.mma_fill", [et], [ knd; m; n; k; ly; fr; i; _v0 ] ->
    let fr = cb fr in
    EApp (EQualified ([], "wmma::fill_fragment"), [ fr; cb i ])

  // FIXME: the second case below is wrong, et_acc is a type, but apparently
  // we are detecting it as an erased expression and slapping a unit (expression)
  // argument for it.
  // FIXME: for whatever reason, the C fragment gets a cast like *(auto *)&f,
  // which is not allowed. We remove it via sed in fixup.sed. Figure out why.
  | "Kuiper.TensorCore.Base.mma_sync'", [et_ab; et_acc], [ scal_ab; scal_acc; m; n; k; la; lb; fa; fb; fc; ea; eb; ec ]
  | "Kuiper.TensorCore.Base.mma_sync'", [et_ab], [et_acc; scal_ab; scal_acc; m; n; k; la; lb; fa; fb; fc; ea; eb; ec ] ->
    let fa = cb fa in
    let fb = cb fb in
    let fc = cb fc in
    EApp (EQualified ([], "wmma::mma_sync"), [ fc; fa; fb; fc ])
  | "Kuiper.TensorCore.Base.mma_sync'", targs, args ->
    raise_error (mlloc_to_range e.loc) Fatal_ExtractionUnsupported [
      text "unexpected types in mma_sync:" ^/^ pp e
    ]

  | "Kuiper.TensorCore.Base.mma_store", [et], [ m; n; k; fr; l; strided_l; gm; f0; m0 ] ->
    let fr = cb fr in
    let layout = EQualified ([], "wmma::mem_row_major") in // FAKE the API only supports this one for now
    let ldm = cb <| get_strided_row_major_stride strided_l in
    let offset = cb <| get_strided_row_major_offset strided_l in
    // Note: use of EBufSub relies on the type of gm unfolding to an array.
    // If IArray/VArray/any other layer defines a new inductive, karamel will complain.
    let gm = EBufSub (cb gm, offset) in
    EApp (EQualified ([], "wmma::store_matrix_sync"), [ gm; fr; ldm; layout])

  (******** FLOAT ARITHMETIC *******)

   (* For halfs, using operators worked locally but failed on CI, probably
   depends on CUDA version. Just use the intrinsics. *)
  // TODO: review exactly which variant to use here. There are also
  // __hexp and __hlog that are faster but numerically worse.
  | "Kuiper.Float16.Base.zero", [], [] -> EConstant (Half, "__float2half_rn(0.0f)")
  | "Kuiper.Float16.Base.one",  [], [] -> EConstant (Half, "__float2half_rn(1.0f)")
  | "Kuiper.Float16.Base.add",  [], [] -> EQualified ([], "__hadd")
  | "Kuiper.Float16.Base.mul",  [], [] -> EQualified ([], "__hmul")
  | "Kuiper.Float16.Base.sub",  [], [] -> EQualified ([], "__hsub")
  | "Kuiper.Float16.Base.div",  [], [] -> EQualified ([], "__hdiv")
  | "Kuiper.Float16.Base.fexp", [], [] -> EQualified ([], "hexp")
  | "Kuiper.Float16.Base.flog", [], [] -> EQualified ([], "hlog")
  | "Kuiper.Float16.Base.eq",   [], [] -> EOp (Eq, Half)
  | "Kuiper.Float16.Base.lt",   [], [] -> EOp (Lt, Half)
  | "Kuiper.Float16.Base.lte",  [], [] -> EOp (Lte, Half)
  | "Kuiper.Float16.Base.largest",  [], [] -> EConstant (Half, "HLF_MAX")
  | "Kuiper.Float16.Base.infinity", [], [] -> EConstant (Half, "HLF_INFINITY")
  | "Kuiper.Float16.Base.of_literal", [], [s] ->
    begin match s.expr with
    | MLE_Const (MLC_String v) -> EConstant (Half, v)
    | _ -> failwith "Float16.of_literal: expected a string literal"
    end
  | "Kuiper.Float16.Base.of_int", [], [i] -> EApp (EQualified ([], "__ll2half_rn"), [cb i])

  | "Kuiper.BFloat16.Base.zero", [], [] -> EConstant (BFloat16, "__float2bfloat16(0.0f)")
  | "Kuiper.BFloat16.Base.one",  [], [] -> EConstant (BFloat16, "__float2bfloat16(1.0f)")
  | "Kuiper.BFloat16.Base.add",  [], [] -> EQualified ([], "kpr_bf16add")
  | "Kuiper.BFloat16.Base.mul",  [], [] -> EQualified ([], "kpr_bf16mul")
  | "Kuiper.BFloat16.Base.sub",  [], [] -> EQualified ([], "kpr_bf16sub")
  | "Kuiper.BFloat16.Base.div",  [], [] -> EQualified ([], "kpr_bf16div")
  | "Kuiper.BFloat16.Base.fexp", [], [] -> EQualified ([], "kpr_bf16exp")
  | "Kuiper.BFloat16.Base.flog", [], [] -> EQualified ([], "kpr_bf16log")
  | "Kuiper.BFloat16.Base.eq",   [], [] -> EOp (Eq, BFloat16)
  | "Kuiper.BFloat16.Base.lt",   [], [] -> EOp (Lt, BFloat16)
  | "Kuiper.BFloat16.Base.lte",  [], [] -> EOp (Lte, BFloat16)
  | "Kuiper.BFloat16.Base.largest",  [], [] -> EConstant (BFloat16, "__ushort_as_bfloat16(0x7F7F);")
  | "Kuiper.BFloat16.Base.infinity", [], [] -> EConstant (BFloat16, "__float2bfloat16(INFINITY);")
  | "Kuiper.BFloat16.Base.of_literal", [], [s] ->
    begin match s.expr with
    | MLE_Const (MLC_String v) -> EConstant (BFloat16, v)
    | _ -> failwith "BFloat16.of_literal: expected a string literal"
    end
  | "Kuiper.BFloat16.Base.of_int", [], [i] -> EApp (EQualified ([], "__ll2bfloat16_rn"), [cb i])

  | "Kuiper.Float32.Base.zero", [], [] -> EConstant (Float, "0.0f")
  | "Kuiper.Float32.Base.one",  [], [] -> EConstant (Float, "1.0f")
  | "Kuiper.Float32.Base.add",  [], [] -> EOp (Add, Float)
  | "Kuiper.Float32.Base.mul",  [], [] -> EOp (Mult, Float)
  | "Kuiper.Float32.Base.sub",  [], [] -> EOp (Sub, Float)
  | "Kuiper.Float32.Base.div",  [], [] -> EOp (Div, Float)
  | "Kuiper.Float32.Base.fexp", [], [] -> EQualified ([], "expf")
  | "Kuiper.Float32.Base.flog", [], [] -> EQualified ([], "logf")
  | "Kuiper.Float32.Base.eq",   [], [] -> EOp (Eq, Float)
  | "Kuiper.Float32.Base.lt",   [], [] -> EOp (Lt, Float)
  | "Kuiper.Float32.Base.lte",  [], [] -> EOp (Lte, Float)
  | "Kuiper.Float32.Base.valid",  [], [] -> EQualified ([], "kpr_fisvalid")
  | "Kuiper.Float32.Base.largest",  [], [] -> EConstant (Float, "FLT_MAX")
  | "Kuiper.Float32.Base.infinity", [], [] -> EConstant (Float, "INFINITY")
  | "Kuiper.Float32.Base.of_literal", [], [s] ->
    begin match s.expr with
    | MLE_Const (MLC_String v) -> EConstant (Float, v)
    | _ -> failwith "Float32.of_literal: expected a string literal"
    end
  | "Kuiper.Float32.Base.of_int", [], [i] -> ECast (cb i, TInt Float)

  | "Kuiper.Float64.Base.zero", [], [] -> EConstant (Double, "0.0")
  | "Kuiper.Float64.Base.one",  [], [] -> EConstant (Double, "1.0")
  | "Kuiper.Float64.Base.add",  [], [] -> EOp (Add, Double)
  | "Kuiper.Float64.Base.mul",  [], [] -> EOp (Mult, Double)
  | "Kuiper.Float64.Base.sub",  [], [] -> EOp (Sub, Double)
  | "Kuiper.Float64.Base.div",  [], [] -> EOp (Div, Double)
  | "Kuiper.Float64.Base.fexp",  [], [] -> EQualified ([], "exp")
  | "Kuiper.Float64.Base.flog",  [], [] -> EQualified ([], "log")
  | "Kuiper.Float64.Base.lt",   [], [] -> EOp (Lt, Double)
  | "Kuiper.Float64.Base.lte",  [], [] -> EOp (Lte, Double)
  | "Kuiper.Float64.Base.eq",   [], [] -> EOp (Eq, Double)
  | "Kuiper.Float64.Base.largest",  [], [] -> EConstant (Double, "DBL_MAX")
  | "Kuiper.Float64.Base.infinity", [], [] -> EConstant (Double, "INFINITY")
  | "Kuiper.Float64.Base.of_literal", [], [s] ->
    begin match s.expr with
    | MLE_Const (MLC_String v) -> EConstant (Double, v)
    | _ -> failwith "Float64.of_literal: expected a string literal"
    end
  | "Kuiper.Float64.Base.of_int", [], [i] -> ECast (cb i, TInt Double)

  (* Transcendental / math primitives *)

  | "Kuiper.Float16.Base.sqrt",  [], [] -> EQualified ([], "kpr_hsqrt")
  | "Kuiper.Float16.Base.rsqrt", [], [] -> EQualified ([], "kpr_hrsqrt")
  | "Kuiper.Float16.Base.sin",   [], [] -> EQualified ([], "kpr_hsin")
  | "Kuiper.Float16.Base.cos",   [], [] -> EQualified ([], "kpr_hcos")
  | "Kuiper.Float16.Base.tan",   [], [] -> EQualified ([], "kpr_htan")
  | "Kuiper.Float16.Base.asin",  [], [] -> EQualified ([], "kpr_hasin")
  | "Kuiper.Float16.Base.acos",  [], [] -> EQualified ([], "kpr_hacos")
  | "Kuiper.Float16.Base.atan",  [], [] -> EQualified ([], "kpr_hatan")
  | "Kuiper.Float16.Base.sinh",  [], [] -> EQualified ([], "kpr_hsinh")
  | "Kuiper.Float16.Base.cosh",  [], [] -> EQualified ([], "kpr_hcosh")
  | "Kuiper.Float16.Base.tanh",  [], [] -> EQualified ([], "kpr_htanh")
  | "Kuiper.Float16.Base.ceil",  [], [] -> EQualified ([], "kpr_hceil")
  | "Kuiper.Float16.Base.floor", [], [] -> EQualified ([], "kpr_hfloor")
  | "Kuiper.Float16.Base.round", [], [] -> EQualified ([], "kpr_hround")
  | "Kuiper.Float16.Base.fabs",  [], [] -> EQualified ([], "kpr_hfabs")
  | "Kuiper.Float16.Base.erf",   [], [] -> EQualified ([], "kpr_herf")
  | "Kuiper.Float16.Base.log2",  [], [] -> EQualified ([], "kpr_hlog2")
  | "Kuiper.Float16.Base.log10", [], [] -> EQualified ([], "kpr_hlog10")
  | "Kuiper.Float16.Base.exp2",  [], [] -> EQualified ([], "kpr_hexp2")
  | "Kuiper.Float16.Base.pow",   [], [] -> EQualified ([], "kpr_hpow")
  | "Kuiper.Float16.Base.atan2", [], [] -> EQualified ([], "kpr_hatan2")
  | "Kuiper.Float16.Base.fmin",  [], [] -> EQualified ([], "kpr_hfmin")
  | "Kuiper.Float16.Base.fmax",  [], [] -> EQualified ([], "kpr_hfmax")
  | "Kuiper.Float16.Base.fmod",  [], [] -> EQualified ([], "kpr_hfmod")
  | "Kuiper.Float16.Base.copysign", [], [] -> EQualified ([], "kpr_hcopysign")
  | "Kuiper.Float16.Base.fma",   [], [] -> EQualified ([], "kpr_hfma")

  | "Kuiper.BFloat16.Base.sqrt",  [], [] -> EQualified ([], "kpr_bf16sqrt")
  | "Kuiper.BFloat16.Base.rsqrt", [], [] -> EQualified ([], "kpr_bf16rsqrt")
  | "Kuiper.BFloat16.Base.sin",   [], [] -> EQualified ([], "kpr_bf16sin")
  | "Kuiper.BFloat16.Base.cos",   [], [] -> EQualified ([], "kpr_bf16cos")
  | "Kuiper.BFloat16.Base.tan",   [], [] -> EQualified ([], "kpr_bf16tan")
  | "Kuiper.BFloat16.Base.asin",  [], [] -> EQualified ([], "kpr_bf16asin")
  | "Kuiper.BFloat16.Base.acos",  [], [] -> EQualified ([], "kpr_bf16acos")
  | "Kuiper.BFloat16.Base.atan",  [], [] -> EQualified ([], "kpr_bf16atan")
  | "Kuiper.BFloat16.Base.sinh",  [], [] -> EQualified ([], "kpr_bf16sinh")
  | "Kuiper.BFloat16.Base.cosh",  [], [] -> EQualified ([], "kpr_bf16cosh")
  | "Kuiper.BFloat16.Base.tanh",  [], [] -> EQualified ([], "kpr_bf16tanh")
  | "Kuiper.BFloat16.Base.ceil",  [], [] -> EQualified ([], "kpr_bf16ceil")
  | "Kuiper.BFloat16.Base.floor", [], [] -> EQualified ([], "kpr_bf16floor")
  | "Kuiper.BFloat16.Base.round", [], [] -> EQualified ([], "kpr_bf16round")
  | "Kuiper.BFloat16.Base.fabs",  [], [] -> EQualified ([], "kpr_bf16fabs")
  | "Kuiper.BFloat16.Base.erf",   [], [] -> EQualified ([], "kpr_bf16erf")
  | "Kuiper.BFloat16.Base.log2",  [], [] -> EQualified ([], "kpr_bf16log2")
  | "Kuiper.BFloat16.Base.log10", [], [] -> EQualified ([], "kpr_bf16log10")
  | "Kuiper.BFloat16.Base.exp2",  [], [] -> EQualified ([], "kpr_bf16exp2")
  | "Kuiper.BFloat16.Base.pow",   [], [] -> EQualified ([], "kpr_bf16pow")
  | "Kuiper.BFloat16.Base.atan2", [], [] -> EQualified ([], "kpr_bf16atan2")
  | "Kuiper.BFloat16.Base.fmin",  [], [] -> EQualified ([], "kpr_bf16fmin")
  | "Kuiper.BFloat16.Base.fmax",  [], [] -> EQualified ([], "kpr_bf16fmax")
  | "Kuiper.BFloat16.Base.fmod",  [], [] -> EQualified ([], "kpr_bf16fmod")
  | "Kuiper.BFloat16.Base.copysign", [], [] -> EQualified ([], "kpr_bf16copysign")
  | "Kuiper.BFloat16.Base.fma",   [], [] -> EQualified ([], "kpr_bf16fma")

  | "Kuiper.Float32.Base.sqrt",  [], [] -> EQualified ([], "sqrtf")
  | "Kuiper.Float32.Base.rsqrt", [], [] -> EQualified ([], "rsqrtf")
  | "Kuiper.Float32.Base.sin",   [], [] -> EQualified ([], "sinf")
  | "Kuiper.Float32.Base.cos",   [], [] -> EQualified ([], "cosf")
  | "Kuiper.Float32.Base.tan",   [], [] -> EQualified ([], "tanf")
  | "Kuiper.Float32.Base.asin",  [], [] -> EQualified ([], "asinf")
  | "Kuiper.Float32.Base.acos",  [], [] -> EQualified ([], "acosf")
  | "Kuiper.Float32.Base.atan",  [], [] -> EQualified ([], "atanf")
  | "Kuiper.Float32.Base.sinh",  [], [] -> EQualified ([], "sinhf")
  | "Kuiper.Float32.Base.cosh",  [], [] -> EQualified ([], "coshf")
  | "Kuiper.Float32.Base.tanh",  [], [] -> EQualified ([], "tanhf")
  | "Kuiper.Float32.Base.ceil",  [], [] -> EQualified ([], "ceilf")
  | "Kuiper.Float32.Base.floor", [], [] -> EQualified ([], "floorf")
  | "Kuiper.Float32.Base.round", [], [] -> EQualified ([], "roundf")
  | "Kuiper.Float32.Base.fabs",  [], [] -> EQualified ([], "fabsf")
  | "Kuiper.Float32.Base.erf",   [], [] -> EQualified ([], "erff")
  | "Kuiper.Float32.Base.log2",  [], [] -> EQualified ([], "log2f")
  | "Kuiper.Float32.Base.log10", [], [] -> EQualified ([], "log10f")
  | "Kuiper.Float32.Base.exp2",  [], [] -> EQualified ([], "exp2f")
  | "Kuiper.Float32.Base.pow",   [], [] -> EQualified ([], "powf")
  | "Kuiper.Float32.Base.atan2", [], [] -> EQualified ([], "atan2f")
  | "Kuiper.Float32.Base.fmin",  [], [] -> EQualified ([], "fminf")
  | "Kuiper.Float32.Base.fmax",  [], [] -> EQualified ([], "fmaxf")
  | "Kuiper.Float32.Base.fmod",  [], [] -> EQualified ([], "fmodf")
  | "Kuiper.Float32.Base.copysign", [], [] -> EQualified ([], "copysignf")
  | "Kuiper.Float32.Base.fma",   [], [] -> EQualified ([], "fmaf")

  | "Kuiper.Float64.Base.sqrt",  [], [] -> EQualified ([], "sqrt")
  | "Kuiper.Float64.Base.rsqrt", [], [] -> EQualified ([], "rsqrt")
  | "Kuiper.Float64.Base.sin",   [], [] -> EQualified ([], "sin")
  | "Kuiper.Float64.Base.cos",   [], [] -> EQualified ([], "cos")
  | "Kuiper.Float64.Base.tan",   [], [] -> EQualified ([], "tan")
  | "Kuiper.Float64.Base.asin",  [], [] -> EQualified ([], "asin")
  | "Kuiper.Float64.Base.acos",  [], [] -> EQualified ([], "acos")
  | "Kuiper.Float64.Base.atan",  [], [] -> EQualified ([], "atan")
  | "Kuiper.Float64.Base.sinh",  [], [] -> EQualified ([], "sinh")
  | "Kuiper.Float64.Base.cosh",  [], [] -> EQualified ([], "cosh")
  | "Kuiper.Float64.Base.tanh",  [], [] -> EQualified ([], "tanh")
  | "Kuiper.Float64.Base.ceil",  [], [] -> EQualified ([], "ceil")
  | "Kuiper.Float64.Base.floor", [], [] -> EQualified ([], "floor")
  | "Kuiper.Float64.Base.round", [], [] -> EQualified ([], "round")
  | "Kuiper.Float64.Base.fabs",  [], [] -> EQualified ([], "fabs")
  | "Kuiper.Float64.Base.erf",   [], [] -> EQualified ([], "erf")
  | "Kuiper.Float64.Base.log2",  [], [] -> EQualified ([], "log2")
  | "Kuiper.Float64.Base.log10", [], [] -> EQualified ([], "log10")
  | "Kuiper.Float64.Base.exp2",  [], [] -> EQualified ([], "exp2")
  | "Kuiper.Float64.Base.pow",   [], [] -> EQualified ([], "pow")
  | "Kuiper.Float64.Base.atan2", [], [] -> EQualified ([], "atan2")
  | "Kuiper.Float64.Base.fmin",  [], [] -> EQualified ([], "fmin")
  | "Kuiper.Float64.Base.fmax",  [], [] -> EQualified ([], "fmax")
  | "Kuiper.Float64.Base.fmod",  [], [] -> EQualified ([], "fmod")
  | "Kuiper.Float64.Base.copysign", [], [] -> EQualified ([], "copysign")
  | "Kuiper.Float64.Base.fma",   [], [] -> EQualified ([], "fma")

  (******** FLOAT CASTS *******)

  | "Kuiper.Float.Casts.Base.cast_f16_to_f32", [], [x] ->
    EApp (EQualified ([], "__half2float"), [cb x])
  | "Kuiper.Float.Casts.Base.cast_f16_to_f64", [], [x] ->
    ECast (EApp (EQualified ([], "__half2float"), [cb x]), TInt Double)
  | "Kuiper.Float.Casts.Base.cast_f32_to_f16", [], [x] ->
    EApp (EQualified ([], "__float2half_rn"), [cb x])
  | "Kuiper.Float.Casts.Base.cast_f32_to_f64", [], [x] ->
    ECast (cb x, TInt Double)
  | "Kuiper.Float.Casts.Base.cast_bf16_to_f32", [], [x] ->
    EApp (EQualified ([], "__bfloat162float"), [cb x])
  | "Kuiper.Float.Casts.Base.cast_f32_to_bf16", [], [x] ->
    EApp (EQualified ([], "__float2bfloat16"), [cb x])
  | "Kuiper.Float.Casts.Base.cast_f16_to_bf16", [], [x] ->
    EApp (EQualified ([], "__float2bfloat16"), [EApp (EQualified ([], "__half2float"), [cb x])])
  | "Kuiper.Float.Casts.Base.cast_bf16_to_f16", [], [x] ->
    EApp (EQualified ([], "__float2half_rn"), [EApp (EQualified ([], "__bfloat162float"), [cb x])])
  | "Kuiper.Float.Casts.Base.cast_bf16_to_f64", [], [x] ->
    ECast (EApp (EQualified ([], "__bfloat162float"), [cb x]), TInt Double)
  | "Kuiper.Float.Casts.Base.cast_f64_to_bf16", [], [x] ->
    EApp (EQualified ([], "__float2bfloat16"), [ECast (cb x, TInt Float)])
  | "Kuiper.Float.Casts.Base.cast_f64_to_f16", [], [x] ->
    EApp (EQualified ([], "__float2half_rn"), [ECast (cb x, TInt Float)])
  | "Kuiper.Float.Casts.Base.cast_f64_to_f32", [], [x] ->
    ECast (cb x, TInt Float)

  (******** REFERENCES ********)

  (* Sadly these two are still primitive. *)
  | "Kuiper.Ref.gpu_memcpy_host_to_device", [ty], [ sz; dst_gr; src_r; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_gr; cb src_r; sz; cudaMemcpyHostToDevice ])

  | "Kuiper.Ref.gpu_memcpy_device_to_host", [ty], [ sz; dst_r; src_gr; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_r; cb src_gr; sz; cudaMemcpyDeviceToHost ])

  (******** ARRAY ********)

  | "Kuiper.Array.Core.get_ref_of_array_cell", [ty], [ a; i ] ->
    EBufSub (cb a, cb i)

  | "Kuiper.Array.Core.gpu_array_alloc", [ty], [ sz; len ] ->
    let sz : expr = sizeof (cb_ty ty) in
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ sz; cb len ]),
           TBuf (translate_type env ty))

  | "Kuiper.Array.Core.gpu_array_free", [ty], [ a; _v ] ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb a])

  | "Kuiper.Array.Core.slice_read", [ty], [ _i; _j; a; _f; idx; _s ] ->
    EBufRead (cb a, cb idx)

  | "Kuiper.Array.Core.slice_write", [ty], [ _i; _j; a; idx; v; _s ] ->
    EBufWrite (cb a, cb idx, cb v)

  | "Kuiper.Array.Core.gpu_memcpy_host_to_device", [ty], [ sz; _elen; dst_ga; src_a; cnt; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    let bytesize : expr = mul_by_sz sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_ga; cb src_a; bytesize; cudaMemcpyHostToDevice ])

  | "Kuiper.Array.Core.gpu_memcpy_host_to_device'", [ty],
        [ sz; _dst_sz; dst_ga; dst_off; _src_sz; src_a; src_off; cnt; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    let dst_off = cb dst_off in (* element offset, not byte offset *)
    let dst_ga = cb dst_ga in
    let dst_ga = EBufSub (dst_ga, dst_off) in
    let src_off = cb src_off in (* element offset, not byte offset *)
    let src_a = cb src_a in
    let src_a = EBufSub (src_a, src_off) in
    let bytesize : expr = mul_by_sz sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ dst_ga; src_a; bytesize; cudaMemcpyHostToDevice ])

  | "Kuiper.Array.Core.gpu_memcpy_device_to_host", [ty],
  [ sz; _elen; dst_a; src_ga; cnt; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    let bytesize : expr = mul_by_sz sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_a; cb src_ga; bytesize; cudaMemcpyDeviceToHost ])

  | "Kuiper.Array.Core.gpu_memcpy_device_to_host'", [ty],
        [ sz; _dst_sz; dst_a; dst_off; _src_sz; src_ga; src_off; cnt; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    let dst_off = cb dst_off in (* element offset, not byte offset *)
    let dst_ga = cb dst_a in
    let dst_ga = EBufSub (dst_ga, dst_off) in
    let src_off = cb src_off in (* element offset, not byte offset *)
    let src_a = cb src_ga in
    let src_a = EBufSub (src_a, src_off) in
    let bytesize : expr = mul_by_sz sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ dst_ga; src_a; bytesize; cudaMemcpyDeviceToHost ])

  | "Kuiper.Array.Core.gpu_memcpy_device_to_device", [ty], [ sz; _elen; dst_a; src_ga; cnt; f; v; gv ] ->
    let sz : expr = sizeof (cb_ty ty) in
    let bytesize : expr = mul_by_sz sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_a; cb src_ga; bytesize; cudaMemcpyDeviceToDevice ])


  (******** VECTORIZED ARRAY ********)

  | "Kuiper.Array.Vectorized.array_vec_cpy",
    [ et ],
    [ _sized; _has_vec_cpy;
      dst_arr; dst_off; _dst_slice_i; _dst_slice_j;
      src_arr; src_off; _src_slice_i; _src_slice_j;
      _f; _ss; _ds; _sq1; _sq2; _sq3; _sq4 ] ->
    let dst_arr = EBufSub (cb dst_arr, cb dst_off) in
    let src_arr = EBufSub (cb src_arr, cb src_off) in
    EApp (EQualified ([], "vec_memcpy"), [ dst_arr; src_arr; ])

  (******** ATOMIC OPS ********)

  | "Kuiper.AtomicOps.gpu_faa_u32", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_u32"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_u64", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_u64"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_f32", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_f32"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_f64", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_f64"), [cb r; cb v])

  (******** KERNEL CALL ********)

  (* The single kcall! *)
  | "Kuiper.Kernel.Base.launch_kernel_full", [], [ _full_pre; _full_post; kdesc; _epoch ] ->
    begin match extract_kcall cb env kdesc with
    | Some e' -> e'
    | None ->
      raise_error (mlloc_to_range e.loc) Fatal_ExtractionUnsupported [
        text "failed to translate kcall:" ^/^ pp e
      ]
    end

  | "Kuiper.Kernel.Base.sync_device", [], [_unit; _epoch] ->
    _MUST <| EApp (EQualified ([], "cudaDeviceSynchronize"), [ EUnit ])

  (* Misc stuff missing from F*? Without these, they extract to names
     and depend on being linked with that module. *)
  | "FStar.UInt64.zero" , [], [] -> EConstant (Krml.UInt64, "0")
  | "FStar.UInt64.one"  , [], [] -> EConstant (Krml.UInt64, "1")
  | "FStar.UInt32.zero" , [], [] -> EConstant (Krml.UInt32, "0")
  | "FStar.UInt32.one"  , [], [] -> EConstant (Krml.UInt32, "1")
  | "FStar.UInt16.zero" , [], [] -> EConstant (Krml.UInt16, "0")
  | "FStar.UInt16.one"  , [], [] -> EConstant (Krml.UInt16, "1")
  | "FStar.UInt8.zero"  , [], [] -> EConstant (Krml.UInt8, "0")
  | "FStar.UInt8.one"   , [], [] -> EConstant (Krml.UInt8, "1")
  | "FStar.Int64.zero"  , [], [] -> EConstant (Krml.Int64, "0")
  | "FStar.Int64.one"   , [], [] -> EConstant (Krml.Int64, "1")
  | "FStar.Int32.zero"  , [], [] -> EConstant (Krml.Int32, "0")
  | "FStar.Int32.one"   , [], [] -> EConstant (Krml.Int32, "1")
  | "FStar.Int16.zero"  , [], [] -> EConstant (Krml.Int16, "0")
  | "FStar.Int16.one"   , [], [] -> EConstant (Krml.Int16, "1")
  | "FStar.Int8.zero"   , [], [] -> EConstant (Krml.Int8, "0")
  | "FStar.Int8.one"    , [], [] -> EConstant (Krml.Int8, "1")

  | "Kuiper.SizeT.sizet_and",    [], [ x; y ] -> EApp (EOp (BAnd, fake_SizeT), [ cb x; cb y ])
  | "Kuiper.SizeT.sizet_to_u32", [], [ sz ]   -> ECast (cb sz, TInt UInt32)

  (* For loops *)
  | "Kuiper.For.for_loop", [], [ lo; hi; _pre; _post; f ] ->
    let env_v = extend env "i" in
    let cb_v = translate_expr env_v in
    let v_binder : binder = {
      name = "i";
      typ = TInt fake_SizeT;
      mut = true;
      meta = [];
    } in
    ELet (v_binder, cb lo,
      EGFor (EUnit,
             EApp (EOp (Lt, fake_SizeT), [ EBound 0; cb_v hi ]),
             EAssign (EBound 0,
               EApp (EOp (Add, fake_SizeT), [ EBound 0; EConstant (fake_SizeT, "1") ])),
             EApp (cb_v f, [EBound 0])))

  | "Kuiper.For.for_loop'", [], [ lo; hi; _pre; _post; _frame; f ] ->
    let env_v = extend env "i" in
    let cb_v = translate_expr env_v in
    let v_binder : binder = {
      name = "i";
      typ = TInt fake_SizeT;
      mut = true;
      meta = [];
    } in
    ELet (v_binder, cb lo,
      EGFor (EUnit,
             EApp (EOp (Lt, fake_SizeT), [ EBound 0; cb_v hi ]),
             EAssign (EBound 0,
               EApp (EOp (Add, fake_SizeT), [ EBound 0; EConstant (fake_SizeT, "1") ])),
             EApp (cb_v f, [EBound 0])))

  | _ -> raise NotSupportedByKrmlExtension

(* Observe record type declarations and store their field info in the registry.
   Always falls through to the default handler. *)
let kpr_translate_type_decl : translate_type_decl_t = fun env td ->
  (if !dbg then
    Format.print3 "KPR translate_type_decl: %s.%s (defn=%s)\n"
      (String.concat "." env.module_name) td.tydecl_name
      (match td.tydecl_defn with
       | Some (MLTD_Record _) -> "Record"
       | Some (MLTD_Abbrev _) -> "Abbrev"
       | Some (MLTD_DType _)  -> "DType"
       | None                 -> "None");
   match td.tydecl_defn with
   | Some (MLTD_Record fields) ->
     let type_path = string_of_mlpath (env.module_name, td.tydecl_name) in
     if !dbg then
       Format.print2 "KPR record registry: registering %s (%s fields)\n" type_path (show (List.length fields));
     if None? (List.assoc type_path !record_fields_registry) then
       record_fields_registry := (type_path, fields) :: !record_fields_registry
   | _ -> ());
  raise NotSupportedByKrmlExtension

let _ =
  register_pre_translate_type_without_decay kpr_translate_type_without_decay;
  register_pre_translate_type_decl kpr_translate_type_decl;
  register_pre_translate_expr kpr_translate_expr
