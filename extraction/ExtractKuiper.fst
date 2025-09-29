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

(* Really pushing it *)
let int_lit (i : int) : expr =
  EQualified ([], show i)

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

let zero_for_deref = EQualified (["C"], "_zero_for_deref")

let deref e = EBufRead (e, zero_for_deref)

let dbg = Debug.get_toggle "kuiper"

let rec drop n (lst : list 'a) : list 'a =
  match lst with
  | [] -> []
  | _ when n <= 0 -> lst
  | _ -> List.tl (drop (n - 1) lst)

exception Failed of string

let kpr_translate_type_without_decay : translate_type_without_decay_t = fun env t ->
  let cb = translate_type_without_decay env in
  let x = type_hta t in
  if None? x then raise NotSupportedByKrmlExtension;
  match Some?.v x with
  | "Kuiper.Ref.gpu_ref",     [t]      -> TBuf (cb t)
  | "Kuiper.Array.gpu_array", [t; len] -> TBuf (cb t)

  | "Kuiper.TensorCore.fragment", [et; knd; m; n; k; layout] ->
    TQualified (([], "auto")) // :-)

  | "Kuiper.Float16.t",               [] -> TInt Half
  | "Kuiper.Float32.t",               [] -> TInt Float
  | "Kuiper.Float64.t",               [] -> TInt Double
  | "Kuiper.VectorType.float4", [] -> TQualified ([], "float4")
  | _ -> raise NotSupportedByKrmlExtension

let cudaMemcpyDeviceToHost = EQualified ([], "cudaMemcpyDeviceToHost")
let cudaMemcpyHostToDevice = EQualified ([], "cudaMemcpyHostToDevice")
let cudaMemcpyDeviceToDevice = EQualified ([], "cudaMemcpyDeviceToDevice")

let get_record_field fname (e:mlexpr) : mlexpr =
  let assoc' k v =
    match List.assoc k v with
    | Some r -> r
    | None -> failwith ("get_record_field: field not found: " ^ k ^ "  ---  " ^ show v)
  in
  match (unmagic e).expr with
  | MLE_Record (_, _, flds) -> assoc' fname flds
  | _ -> raise (Failed ("Expected a single-field record for the size, got: " ^ show e))

let get_sizet (e : mlexpr) : mlexpr = get_record_field "size" e
let get_strided_row_major_offset (e : mlexpr) : mlexpr = get_record_field "offset" e
let get_strided_row_major_stride (e : mlexpr) : mlexpr = get_record_field "stride" e

let _MUST (e : expr) : expr =
    EApp (EQualified ([], "MUST"), [e])

let ctr = mk_ref 0

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

let hoist (g : env) (e : mlexpr) : mlexpr =
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
  let mk_binder (v, t) = {mlbinder_name = v; mlbinder_ty = t; mlbinder_attrs = []} in
  let fresh = "__hoisted_" ^ string_of_int !ctr in
  let arr_t = List.fold_right mkarr (List.map snd fvs) (mkarr ml_unit_ty et) in
  ctr := !ctr + 1;

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
  let call = MLE_App (nm, List.map (fun (v, t) -> with_ty t <| MLE_Var v) fvs ) in
  let e' = {e with expr = call} in
  if !dbg then (
    Format.print3_warning
      "Hoisted %s into %s, creating the declaration\n%s\n" (mlexpr_to_string e) fresh (show decl);
    Format.print1_warning "The translated expression is %s\n" (mlexpr_to_string e')
  );
  e'

(* Returns (possibly) the size of each element and length of array.
The type of the array has already been erased, we cannot get it here. *)
let parse_shmem_desc (e : mlexpr) : option (mlexpr & mlexpr) =
  let open FStarC.Class.Monad in
  match e.expr with
  | MLE_CTor (fv, [_ty; sized; len]) when string_of_mlpath fv = "Kuiper.SHMem.SHArray" ->
    let sized_a = get_sizet sized in
    return (sized_a, len)
  | _ ->
    Format.print1 "Not a shmem_desc: %s\n" (mlexpr_to_string e);
    None

let extract_kcall (env : Krml.env) (kdesc : mlexpr) : option mlexpr =
  let open FStarC.Class.Monad in
  let assoc' k v =
    match List.assoc k v with
    | Some r -> r
    | None -> failwith ("launch_kernel: field not found: " ^ k)
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
      let mk_c_sh (desc : list (mlexpr & mlexpr)) : mlexpr & mlexpr =
        let rec aux (off : mlexpr) (desc : list (mlexpr & mlexpr)) : mlexpr & mlexpr =
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
      let get_one_binder (e:mlexpr) : mlbinder & mlexpr =
        match e.expr with
        | MLE_Fun ([b], body) -> b, body
        | MLE_Fun (b::bs, body) ->
          b, { e with expr = MLE_Fun (bs, body) } (* type is wrong, but it doesn't matter *)
        | _ -> failwith ("launch_kernel: no binder for: " ^ show e)
      in
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
      let apply_lam (f : mlexpr) (v : mlexpr) : mlexpr =
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
      let kf = hoist env kf in
      let hd, rest_args = head_and_args kf in
      return (nblk, nthr, shmem_bytesz, hd, rest_args)

    | _ ->
      failwith ("launch_kernel: not a record: " ^ show kdesc)
  in
  let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "KPR_KCALL") in
  let e' =
  with_ty ml_unit_ty <|
    MLE_App (kcall, [ hd; nblk; nthr; smem_bytesz ] @ rest_args)
  in
  // Format.print1_warning "New kcall: %s\n" (show e');
  return e'

let kpr_translate_alloc_fragment cb et knd m n k layout =
    let macro_suff, knd =
      match cta knd with
      | Some ("Kuiper.TensorCore.FragA",     [], []) -> "", EQualified ([], "wmma::matrix_a")
      | Some ("Kuiper.TensorCore.FragB",     [], []) -> "", EQualified ([], "wmma::matrix_b")
      | Some ("Kuiper.TensorCore.FragAcc",   [], []) -> "_C", EQualified ([], "wmma::accumulator")
      | x -> raise (Failed <| "unexpected knd in __alloc_fragment: " ^ show (x, tag_of knd))
    in
    let layout : option expr =
      match cta layout with
      | Some ("Kuiper.TensorCore.FragLRM",    [], []) -> Some <| EQualified ([], "wmma::row_major")
      | Some ("Kuiper.TensorCore.FragLCM",    [], []) -> Some <| EQualified ([], "wmma::column_major")
      | Some ("Kuiper.TensorCore.FragLAcc",   [], []) -> None
      | x -> raise (Failed <| "unexpected layout in __alloc_fragment: " ^ show (x, tag_of layout))
    in
    let faketype =
      match et with
      | MLTY_Named ([], (["Kuiper"; "Float16"], "t")) -> EQualified ([], "half")
      | MLTY_Named ([], (["Kuiper"; "Float32"], "t")) -> EQualified ([], "float")
      | MLTY_Named ([], (["Kuiper"; "Float64"], "t")) -> EQualified ([], "double")
    in
    (* Tries to remove the size_t cast in literals, just to make the code
       more readable. *)
    let ss x =
      match x with
      | EConstant (SizeT, s) -> int_lit (FStarC.Util.int_of_string s)
      | _ -> x
    in
    let args =
      [ faketype; knd; ss (cb m); ss (cb n); ss (cb k) ]
      @ (match layout with | Some l -> [l] | None -> [])
    in
      EApp (EQualified ([], "KPR_FRAGMENT_TYPE" ^ macro_suff), args)

let kpr_translate_expr : translate_expr_t = fun env e ->
  let e = flatten_app e in
  if !dbg
  then Format.print1_warning "ExtractKuiper.kpr_translate_expr %s\n" (mlexpr_to_string e);
  let cb = translate_expr env in
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

  | "Kuiper.Barrier.barrier_wait", [], [ _unit; _n; _p; _q; _it; _tid ] ->
    EApp (EQualified ([], "__syncthreads"), [ EUnit ])

  (******** TENSOR CORE OPERATIONS, FRAGMENTS, ETC ********)

  | "Kuiper.TensorCore.__alloc_array_fragment", [et], [ knd; m; n; k; layout; size ] ->
    // EBufCreate (Stack,
      EApp (EQualified ([], "KPR_INIT"),
        [EApp (EQualified ([], "KPR_ARRAY_FRAGMENT_TYPE"), [kpr_translate_alloc_fragment cb et knd m n k layout; cb size])])
      // ,EConstant (SizeT, "1")
    // )

  | "Kuiper.TensorCore.__alloc_fragment", [et], [ knd; m; n; k; layout ] ->
    // EBufCreate (Stack,
      EApp (EQualified ([], "KPR_INIT"),
        [kpr_translate_alloc_fragment cb et knd m n k layout])
      // ,EConstant (SizeT, "1")
    // )

  | "Kuiper.TensorCore.mma_loadA", [et], [ m; n; k; fr; l; strided_l; gm; f; m0; f0 ]
  | "Kuiper.TensorCore.mma_loadB", [et], [ m; n; k; fr; l; strided_l; gm; f; m0; f0 ] ->
    let fr = cb fr in
    let ldm = cb <| get_strided_row_major_stride strided_l in
    let offset = cb <| get_strided_row_major_offset strided_l in
    let gm = cb gm in
    // Cannot use EBufSub: gm is a matrix (varray), not a karamel buffer
    let gm = EApp (EQualified ([], "kpr_offset"), [gm; offset]) in
    EApp (EQualified ([], "wmma::load_matrix_sync"), [ fr; gm; ldm ])

  | "Kuiper.TensorCore.mma_loadAccum", [et], [m; n; k; fr; l; strided_l; gm; f; m0; f0 ] ->
    // TODO remove duplicated code
    let fr = cb fr in
    let layout = EQualified ([], "wmma::mem_row_major") in // FAKE the API only supports this one for now
    let ldm = cb <| get_strided_row_major_stride strided_l in
    let offset = cb <| get_strided_row_major_offset strided_l in
    let gm = cb gm in
    // Cannot use EBufSub: gm is a matrix (varray), not a karamel buffer
    let gm = EApp (EQualified ([], "kpr_offset"), [gm; offset]) in
    EApp (EQualified ([], "wmma::load_matrix_sync"), [ fr; gm; ldm; layout ])

  | "Kuiper.TensorCore.mma_fill", [et], [ knd; m; n; k; ly; fr; i; _v0 ] ->
    let fr = cb fr in
    EApp (EQualified ([], "wmma::fill_fragment"), [ fr; cb i ])

  // FIXME: the second case below is wrong, et_acc is a type, but apparently
  // we are detecting it as an erased expression and slapping a unit (expression)
  // argument for it.
  // FIXME: for whatever reason, the C fragment gets a cast like *(auto *)&f,
  // which is not allowed. We remove it via sed in fixup.sed. Figure out why.
  | "Kuiper.TensorCore.mma_sync'", [et_ab; et_acc], [ scal_ab; scal_acc; m; n; k; la; lb; fa; fb; fc; ea; eb; ec ]
  | "Kuiper.TensorCore.mma_sync'", [et_ab], [et_acc; scal_ab; scal_acc; m; n; k; la; lb; fa; fb; fc; ea; eb; ec ] ->
    let fa = cb fa in
    let fb = cb fb in
    let fc = cb fc in
    EApp (EQualified ([], "wmma::mma_sync"), [ fc; fa; fb; fc ])
  | "Kuiper.TensorCore.mma_sync'", targs, args ->
    raise (Failed <| "unexpected types in mma_sync: " ^ show (targs, args))

  | "Kuiper.TensorCore.mma_store", [et], [ m; n; k; fr; l; strided_l; gm; f0; m0 ] ->
    let fr = cb fr in
    let layout = EQualified ([], "wmma::mem_row_major") in // FAKE the API only supports this one for now
    let ldm = cb <| get_strided_row_major_stride strided_l in
    let offset = cb <| get_strided_row_major_offset strided_l in
    let gm = cb gm in
    // Cannot use EBufSub: gm is a matrix (varray), not a karamel buffer
    let gm = EApp (EQualified ([], "kpr_offset"), [gm; offset]) in
    EApp (EQualified ([], "wmma::store_matrix_sync"), [ gm; fr; ldm; layout])

  (******** FLOAT ARITHMETIC *******)

   (* For halfs, using operators worked locally but failed on CI, probably
   depends on CUDA version. Just use the intrinsics. *)
  | "Kuiper.Float16.zero", [], [] -> EConstant (Half, "0.0f")
  | "Kuiper.Float16.one",  [], [] -> EConstant (Half, "1.0f")
  | "Kuiper.Float16.add",  [], [] -> EQualified ([], "__hadd")
  | "Kuiper.Float16.sub",  [], [] -> EQualified ([], "__hsub")
  | "Kuiper.Float16.mul",  [], [] -> EQualified ([], "__hmul")
  | "Kuiper.Float16.div",  [], [] -> EQualified ([], "__hdiv")
  | "Kuiper.Float16.exp",  [], [] -> EQualified ([], "__hexp")

  | "Kuiper.Float32.zero", [], [] -> EConstant (Float, "0.0f")
  | "Kuiper.Float32.one",  [], [] -> EConstant (Float, "1.0f")
  | "Kuiper.Float32.add",  [], [] -> EOp (Add, Float)
  | "Kuiper.Float32.sub",  [], [] -> EOp (Sub, Float)
  | "Kuiper.Float32.mul",  [], [] -> EOp (Mult, Float)
  | "Kuiper.Float32.div",  [], [] -> EOp (Div, Float)
  | "Kuiper.Float32.exp",  [], [] -> EQualified ([], "exp")

  | "Kuiper.Float64.zero", [], [] -> EConstant (Double, "0.0l")
  | "Kuiper.Float64.one",  [], [] -> EConstant (Double, "1.0l")
  | "Kuiper.Float64.add",  [], [] -> EOp (Add, Double)
  | "Kuiper.Float64.sub",  [], [] -> EOp (Sub, Double)
  | "Kuiper.Float64.mul",  [], [] -> EOp (Mult, Double)
  | "Kuiper.Float64.div",  [], [] -> EOp (Div, Double)
  | "Kuiper.Float64.exp",  [], [] -> EQualified ([], "exp")

  (******** REFERENCES ********)

  | "Kuiper.Ref.gpu_alloc0", [ty], [ sz; _unit ] ->
    let sz : mlexpr = get_sizet sz in
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ cb sz; EConstant (SizeT, "1") ]),
           TBuf (translate_type env ty))

  | "Kuiper.Ref.gpu_free", [ty], [ r; _v ] ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb r])

  | "Kuiper.Ref.gpu_read", [ty], [ e; _perm; _v ] ->
    deref (cb e)

  | "Kuiper.Ref.gpu_write", [ty], [ e1; e2; _v0 ] ->
    EBufWrite (cb e1, zero_for_deref, cb e2)

  | "Kuiper.Ref.gpu_memcpy_host_to_device", [ty], [ sz; dst_gr; src_r; f; v; gv ] ->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_gr; cb src_r; cb sz ; cudaMemcpyHostToDevice ])

  | "Kuiper.Ref.gpu_memcpy_device_to_host", [ty], [ sz; dst_r; src_gr; f; v; gv ] ->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_r; cb src_gr; cb sz; cudaMemcpyDeviceToHost ])

  | "Kuiper.Ref.gpu_memcpy_device_to_device", [ty], [ sz; dst_gr; src_r; f; v; gv ] ->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_gr; cb src_r; cb sz ; cudaMemcpyDeviceToDevice ])

  (******** ARRAY ********)

  | "Kuiper.Array.gpu_array_alloc", [ty], [ sz; len ] ->
    let sz : mlexpr = get_sizet sz in
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ cb sz; cb len ]),
           TBuf (translate_type env ty))

  | "Kuiper.Array.gpu_array_free", [ty], [ _sz; a; _v ] ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb a])

  | "Kuiper.Array.gpu_array_read", [ty], [ _sz; _i; _j; a; _f; idx; _s ] ->
    EBufRead (cb a, cb idx)

  | "Kuiper.Array.gpu_array_write", [ty], [ _sz; _i; _j; a; idx; v; _s ] ->
    EBufWrite (cb a, cb idx, cb v)

  | "Kuiper.Array.gpu_memcpy_host_to_device", [ty], [ sz; _elen; dst_ga; src_a; cnt; f; v; gv ] ->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_ga; cb src_a; bytesize; cudaMemcpyHostToDevice ])

  | "Kuiper.Array.gpu_memcpy_host_to_device'", [ty],
        [ sz; _dst_sz; dst_ga; dst_off; _src_sz; src_a; src_off; cnt; f; v; gv ] ->
    let sz : expr = cb <| get_sizet sz in
    let mul_by_sz (e:expr) = EApp (EOp (Mult, SizeT), [ sz; e ]) in
    let dst_off = mul_by_sz (cb dst_off) in
    let dst_ga = cb dst_ga in
    let dst_ga = EBufSub (dst_ga, dst_off) in
    let src_off = mul_by_sz (cb src_off) in
    let src_a = cb src_a in
    let src_a = EBufSub (src_a, src_off) in
    let bytesize : expr = mul_by_sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ dst_ga; src_a; bytesize; cudaMemcpyHostToDevice ])

  | "Kuiper.Array.gpu_memcpy_device_to_host", [ty], [ sz; _elen; dst_a; src_ga; cnt; f; v; gv ] ->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_a; cb src_ga; bytesize; cudaMemcpyDeviceToHost ])

  | "Kuiper.Array.gpu_memcpy_device_to_host'", [ty],
        [ sz; _dst_sz; dst_a; dst_off; _src_sz; src_ga; src_off; cnt; f; v; gv ] ->
    let sz : expr = cb <| get_sizet sz in
    let mul_by_sz (e:expr) = EApp (EOp (Mult, SizeT), [ sz; e ]) in
    let dst_off = mul_by_sz (cb dst_off) in
    let dst_ga = cb dst_a in
    let dst_ga = EBufSub (dst_ga, dst_off) in
    let src_off = mul_by_sz (cb src_off) in
    let src_a = cb src_ga in
    let src_a = EBufSub (src_a, src_off) in
    let bytesize : expr = mul_by_sz (cb cnt) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ dst_ga; src_a; bytesize; cudaMemcpyDeviceToHost ])

  | "Kuiper.Array.gpu_memcpy_device_to_device", [ty], [ sz; _elen; dst_a; src_ga; cnt; f; v; gv ] ->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_a; cb src_ga; bytesize; cudaMemcpyDeviceToDevice ])


  (******** VECTORIZED ARRAY ********)

  | "Kuiper.Array.Vectorized.gpu_array_vec4_read", [], [ _sz; _i; _j; a; _f; idx; _s ] ->
    EApp (EQualified ([], "KPR_VECTZD_READ"), [ cb a; cb idx ])
  | "Kuiper.Array.Vectorized.gpu_array_vec4_write", [], [ _sz; _i; _j; a; idx; v; _s ] ->
    EApp (EQualified ([], "KPR_VECTZD_WRITE"), [ cb a; cb idx; cb v ])

  (******** ATOMIC OPS ********)

  | "Kuiper.AtomicOps.gpu_faa_u32", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_u32"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_u64", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_u64"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_f32", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_f32"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_f64", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_f64"), [cb r; cb v])

  (******** VECTOR OPS ********)
  | "Kuiper.VectorType.make_float4", [], [ x; y; z; w; ] ->
    EApp (EQualified ([], "make_float4"), [cb x; cb y; cb z; cb w])
  | "Kuiper.VectorType.getx", [], [ v ] -> EApp (EQualified ([], "KPR_PROJ_X"), [ cb v ])
  | "Kuiper.VectorType.gety", [], [ v ] -> EApp (EQualified ([], "KPR_PROJ_Y"), [ cb v ])
  | "Kuiper.VectorType.getz", [], [ v ] -> EApp (EQualified ([], "KPR_PROJ_Z"), [ cb v ])
  | "Kuiper.VectorType.getw", [], [ v ] -> EApp (EQualified ([], "KPR_PROJ_W"), [ cb v ])

  (******** KERNEL CALL ********)

  (* The single kcall! *)
  | "Kuiper.Kernel.Base.launch_kernel_full", [], [ _full_pre; _full_post; kdesc; _epoch ] ->
    begin match extract_kcall env kdesc with
    | Some e' -> cb e'
    | None -> failwith "failed to translate kcall"
    end

  | "Kuiper.Kernel.Base.sync_device", [], [_unit; _epoch] ->
    EApp (EQualified ([], "cudaDeviceSynchronize"), [ EUnit ])

  (* Misc stuff missing from F*? *)
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

  | "Kuiper.SizeT.sizet_and",    [], [ x; y ] -> EApp (EOp (BAnd, SizeT), [ cb x; cb y ])
  | "Kuiper.SizeT.sizet_to_u32", [], [ sz ]   -> ECast (cb sz, TInt UInt32)


  | _ -> raise NotSupportedByKrmlExtension

let _ =
  register_pre_translate_type_without_decay kpr_translate_type_without_decay;
  register_pre_translate_expr kpr_translate_expr
