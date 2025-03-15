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

open FStarC.Class.Show

let rec drop n (lst : list 'a) : list 'a =
  match lst with
  | [] -> []
  | _ when n <= 0 -> lst
  | _ -> List.tl (drop (n - 1) lst)

exception Failed of string

module BU = FStarC.Util

let flatten_app e =
  let rec aux args e =
    match e.expr with
    | MLE_App (head, args0) -> aux (args0@args) head
    | _ -> (
      match args with
      | [] -> e
      | _ -> {e with expr=MLE_App (e, args)}
    )
  in
  aux [] e

let dbg = Debug.get_toggle "extraction.gpu"

let gpu_translate_type_without_decay : translate_type_without_decay_t = fun env t ->
  match t with
  | MLTY_Named ([arg1; arg2], p) when
    (let p = Syntax.string_of_mlpath p in
     p = "Kuiper.Array.gpu_array"
    )
    ->
      TBuf (translate_type_without_decay env arg1)

  | MLTY_Named ([arg], p) when
    (let p = Syntax.string_of_mlpath p in
     p = "Kuiper.Ref.gpu_ref"
    )
    ->
      TBuf (translate_type_without_decay env arg)

  | MLTY_Named ([], p) when (let p = Syntax.string_of_mlpath p in p = "Kuiper.Float16.t") -> TInt Half
  | MLTY_Named ([], p) when (let p = Syntax.string_of_mlpath p in p = "Kuiper.Float32.t") -> TInt Float
  | MLTY_Named ([], p) when (let p = Syntax.string_of_mlpath p in p = "Kuiper.Float64.t") -> TInt Double

  | _ -> raise NotSupportedByKrmlExtension

let head_and_args (e : mlexpr) : mlexpr & list mlexpr =
  let rec aux acc e =
    match e.expr with
    | MLE_App (head, args) -> aux (args @ acc) head
    | _ -> (e, acc)
  in
  aux [] e

let zero_for_deref = EQualified (["C"], "_zero_for_deref")
let cudaMemcpyDeviceToHost = EQualified ([], "cudaMemcpyDeviceToHost")
let cudaMemcpyHostToDevice = EQualified ([], "cudaMemcpyHostToDevice")
let cudaMemcpyDeviceToDevice = EQualified ([], "cudaMemcpyDeviceToDevice")

let rec unmagic (e : mlexpr) : mlexpr =
  match e.expr with
  | MLE_Coerce (e, _, _) -> unmagic e
  | _ -> e

let get_sizet (e : mlexpr) : mlexpr =
  match (unmagic e).expr with
  | MLE_Record (_, _, [(_, sz)]) -> sz
  | _ -> raise (Failed ("Expected a single-field record for the size, got: " ^ show e))

let _MUST (e : expr) : expr =
    EApp (EQualified ([], "MUST"), [e])

let remove (ks : list string) (vs : list (string & mlty)) : list (string & mlty) =
  List.filter (fun (v, _) -> not (List.existsb (fun k -> v = k) ks)) vs

let rec freevars_of_mlexpr (e : mlexpr) : list (string & mlty) =
  match e.expr with
  | MLE_Const _ -> []
  | MLE_Var v -> [(v, e.mlty)]
  | MLE_Name p -> []
  | MLE_Let (lb, e2) ->
    let freevars_of_lb lb =
      freevars_of_mlexpr lb.mllb_def
    in
    List.collect freevars_of_lb lb._2 @
    (freevars_of_mlexpr e2 |> remove (List.map (fun lb -> lb.mllb_name) lb._2))
  | MLE_App (head, args) ->
    let fvs = freevars_of_mlexpr head in
    List.fold_left (fun acc arg -> acc @ freevars_of_mlexpr arg) fvs args
  | MLE_TApp (head, args) ->
    freevars_of_mlexpr head
  | MLE_Fun (bs, e2) ->
    let fvs = freevars_of_mlexpr e2 in
    let bs = List.map (fun b -> b.mlbinder_name) bs in
    remove bs fvs
  | MLE_Match (e, branches) ->
    let freevars_branch branch =
      let (p, _, e2) = branch in
      let rec pat_bound (p:mlpattern) : list string =
        match p with
        | MLP_Var v -> [v]
        | MLP_CTor (_, args) -> List.collect pat_bound args
        | MLP_Record (_, fields) -> List.map (fun (f, _) -> f) fields
        | MLP_Tuple ps -> List.collect pat_bound ps
        | _ -> []
      in
      let bs = pat_bound p in
      let fvs = freevars_of_mlexpr e2 in
      remove bs fvs
    in
    freevars_of_mlexpr e @ List.collect freevars_branch branches
  | MLE_Coerce (e, _, _) -> freevars_of_mlexpr e
  | MLE_CTor (_, args) ->
    List.collect freevars_of_mlexpr args
  | MLE_Seq es ->
    List.collect freevars_of_mlexpr es
  | MLE_Tuple es ->
    List.collect freevars_of_mlexpr es
  | MLE_Record (_, _, fields) ->
    List.collect (fun (_, e) -> freevars_of_mlexpr e) fields
  | MLE_Proj (e, _) ->
    freevars_of_mlexpr e
  | MLE_If (e1, e2, e3) ->
    let fvs1 = freevars_of_mlexpr e1 in
    let fvs2 = freevars_of_mlexpr e2 in
    let fvs3 = match e3 with
      | Some e -> freevars_of_mlexpr e
      | None -> []
    in
    fvs1 @ fvs2 @ fvs3
  | MLE_Raise (_, args) ->
    List.collect freevars_of_mlexpr args
  | MLE_Try (e, branches) ->
    let fvs = freevars_of_mlexpr e in
    let freevars_branch branch =
      let (_, _, e2) = branch in
      freevars_of_mlexpr e2
    in
    fvs @ List.collect freevars_branch branches

  | _ -> failwith "freevars_of_mlexpr: missing case"

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
    failwith ("Hoist: free variables do not match: " ^ show e0 ^ "\n" ^
                   "fvs0 = " ^ show fvs0 ^ "\n" ^
                   "fvs = " ^ show fvs);
  // BU.print1_warning "fvs = %s\n" (show fvs);
  let mk_binder (v, t) = {mlbinder_name = v; mlbinder_ty = t; mlbinder_attrs = []} in
  let fresh = "__hoisted_" ^ string_of_int !ctr in
  let arr_t = List.fold_right mkarr (List.map snd fvs) (mkarr ml_unit_ty et) in
  ctr := !ctr + 1;

  let bs =  List.map mk_binder fvs @ [extra_unit_binder] in
  let kbs = translate_binders g bs in
  let g0 = { g with names = []; names_t = [] } in
  let lambda = translate_expr (add_binders g0 bs) e in
  let flags = [
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
    BU.print3_warning
      "Hoisted %s into %s, creating the declaration\n%s\n" (mlexpr_to_string e) fresh (show decl);
    BU.print1_warning "The translated expression is %s\n" (mlexpr_to_string e')
  );
  e'

let gpu_translate_expr : translate_expr_t = fun env e ->
  let e = flatten_app e in
  if !dbg
  then BU.print1_warning "ExtractKuiper.gpu_translate_expr %s\n" (mlexpr_to_string e);
  let cb = translate_expr env in
  match e.expr with

  (******** ASSERTIONS ********)
  | MLE_App ({ expr = MLE_Name p } , [ x ])
    when string_of_mlpath p = "Kuiper.Assert.dassert" ->
    EApp (EQualified ([], "KPR_ASSERT"), [ cb x ])

  | MLE_App ({ expr = MLE_Name p } , [ x ])
    when string_of_mlpath p = "Kuiper.Assert.dguard" ->
    EApp (EQualified ([], "KPR_GUARD"), [ cb x ])

  (******** SIZET, missing from F* ********)
  | MLE_App ({ expr = MLE_Name p } , [ x; y ])
    when string_of_mlpath p = "Kuiper.SizeT.sizet_and" ->
    EApp (EOp (BAnd, SizeT), [ cb x; cb y ])

  (******** PREDEFINED VARS ********)

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasednblk; _erasednbid])
    when string_of_mlpath p = "Kuiper.Base.get_gdim" ->
    EApp (EQualified ([], "gridDim_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasednblk; _erasednbid])
    when string_of_mlpath p = "Kuiper.Base.get_bid" ->
    EApp (EQualified ([], "blockIdx_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasednthr; _erasedntid])
    when string_of_mlpath p = "Kuiper.Base.get_bdim" ->
    EApp (EQualified ([], "blockDim_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasednthr; _erasedntid])
    when string_of_mlpath p = "Kuiper.Base.get_tid" ->
    EApp (EQualified ([], "threadIdx_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ sz ])
    when string_of_mlpath p = "Kuiper.SizeT.sizet_to_u32" ->
    ECast (cb sz, TInt UInt32)

  (******** BARRIERS ********)

  | MLE_App ({ expr = MLE_Name p } , [_;u1;u2;u3;u4])
    when string_of_mlpath p = "Kuiper.Barrier.RPM.mbarrier_wait" ->
    EApp (EQualified ([], "__syncthreads"), [ EUnit ])

  (******** FLOAT ARITHMETIC *******)

  | MLE_Name p
    when string_of_mlpath p = "Kuiper.Float16.zero" ->
    EConstant (Half, "0.0f")
  | MLE_Name p
    when string_of_mlpath p = "Kuiper.Float16.one" ->
    EConstant (Half, "1.0f")

   (* Using operators worked locally but failed on CI, probably
   depends on CUDA version. Just use the intrinsics. *)
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float16.add" ->
    EApp (EQualified ([], "__hadd"), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float16.sub" ->
    EApp (EQualified ([], "__hsub"), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x ])
    when string_of_mlpath p = "Kuiper.Float16.neg" ->
    EApp (EQualified ([], "__hneg"), [cb x])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float16.mul" ->
    EApp (EQualified ([], "__hmul"), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float16.div" ->
    EApp (EQualified ([], "__hdiv"), [cb x; cb y])

  | MLE_App ({ expr = MLE_Name p }, [ x ])
    when string_of_mlpath p = "Kuiper.Float16.exp" ->
    EApp (EQualified ([], "__hexp"), [ cb x ])

  | MLE_Name p
    when string_of_mlpath p = "Kuiper.Float32.zero" ->
    EConstant (Float, "0.0f")
  | MLE_Name p
    when string_of_mlpath p = "Kuiper.Float32.one" ->
    EConstant (Float, "1.0f")

  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float32.add" ->
    EApp (EOp (Add, Float), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float32.sub" ->
    EApp (EOp (Sub, Float), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x ])
    when string_of_mlpath p = "Kuiper.Float32.neg" ->
    EApp (EOp (Sub, Float), [EConstant (Float, "0.0f"); cb x])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float32.mul" ->
    EApp (EOp (Mult, Float), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float32.div" ->
    EApp (EOp (Div, Float), [cb x; cb y])

  | MLE_App ({ expr = MLE_Name p }, [ x ])
    when string_of_mlpath p = "Kuiper.Float32.exp" ->
    EApp (EQualified ([], "exp"), [ cb x ])

  | MLE_Name p
    when string_of_mlpath p = "Kuiper.Float64.zero" ->
    EConstant (Double, "0.0l")
  | MLE_Name p
    when string_of_mlpath p = "Kuiper.Float64.one" ->
    EConstant (Double, "1.0l")

  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float64.add" ->
    EApp (EOp (Add, Double), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float64.sub" ->
    EApp (EOp (Sub, Double), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x ])
    when string_of_mlpath p = "Kuiper.Float64.neg" ->
    EApp (EOp (Sub, Double), [EConstant (Double, "0.0l"); cb x])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float64.mul" ->
    EApp (EOp (Mult, Double), [cb x; cb y])
  | MLE_App ({ expr = MLE_Name p }, [ x; y ])
    when string_of_mlpath p = "Kuiper.Float64.div" ->
    EApp (EOp (Div, Double), [cb x; cb y])

  | MLE_App ({ expr = MLE_Name p }, [ x ])
    when string_of_mlpath p = "Kuiper.Float64.exp" ->
    EApp (EQualified ([], "exp"), [ cb x ])

  (******** REFERENCES ********)

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, [ ty ]) }, [ sz; _unit ])
    when string_of_mlpath p = "Kuiper.Ref.gpu_alloc0" ->
    let sz : mlexpr = get_sizet sz in
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ cb sz; EConstant (SizeT, "1") ]),
           TBuf (translate_type env ty))

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, _) }, [ r; _v ])
    when string_of_mlpath p = "Kuiper.Ref.gpu_free" ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb r])

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, _) }, [ e; _perm; _v ])
    when string_of_mlpath p = "Kuiper.Ref.gpu_read" ->
    EBufRead (cb e, zero_for_deref)

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, _) }, [ e1; e2; _v0 ])
    when string_of_mlpath p = "Kuiper.Ref.gpu_write" ->
    EBufWrite (cb e1, zero_for_deref, cb e2)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: dst_gr :: src_r :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Ref.gpu_memcpy_host_to_device"->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_gr; cb src_r; cb sz ; cudaMemcpyHostToDevice ])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: dst_r :: src_gr :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Ref.gpu_memcpy_device_to_host"->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_r; cb src_gr; cb sz; cudaMemcpyDeviceToHost ])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: dst_gr :: src_r :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Ref.gpu_memcpy_device_to_device"->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_gr; cb src_r; cb sz ; cudaMemcpyDeviceToDevice ])

  (******** ARRAY ********)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: len :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_array_alloc" ->
    let sz : mlexpr = get_sizet sz in
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ cb sz; cb len ]),
           TBuf (translate_type env ty))

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: r :: v :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_array_free" ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb r])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, _) }, sz :: i :: j :: r :: f :: idx :: s :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_array_read" ->
    EBufRead (cb r, cb idx)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, _) }, sz :: i :: j :: r :: idx :: v :: s :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_array_write" ->
    EBufWrite (cb r, cb idx, cb v)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) },
             sz :: _elen :: dst_ga :: src_a :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_memcpy_host_to_device" ->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_ga; cb src_a; bytesize; cudaMemcpyHostToDevice ])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) },
             sz :: _dst_sz :: dst_ga :: dst_off :: _src_sz :: src_a :: src_off :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_memcpy_host_to_device'" ->
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

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) },
             sz :: _elen :: dst_a :: src_ga :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_memcpy_device_to_host"->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_a; cb src_ga; bytesize; cudaMemcpyDeviceToHost ])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) },
             sz :: _dst_sz :: dst_a :: dst_off :: _src_sz :: src_ga :: src_off :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_memcpy_device_to_host'" ->
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

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: elen :: dst_a :: src_ga :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "Kuiper.Array.gpu_memcpy_device_to_device"->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb dst_a; cb src_ga; bytesize; cudaMemcpyDeviceToDevice ])


  (******** ATOMIC OPS ********)

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "Kuiper.AtomicOps.gpu_faa_u32" ->
    (* Can we cast here instead of using a wrapper? *)
    EApp (EQualified ([], "atomic_add_u32"), [cb r; cb v])

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "Kuiper.AtomicOps.gpu_faa_u64" ->
    EApp (EQualified ([], "atomic_add_u64"), [cb r; cb v])

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "Kuiper.AtomicOps.gpu_faa_f32" ->
    EApp (EQualified ([], "atomic_add_f32"), [cb r; cb v])

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "Kuiper.AtomicOps.gpu_faa_f64" ->
    EApp (EQualified ([], "atomic_add_f64"), [cb r; cb v])

  (******** OBTAIN SHMEM ********)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sized_a :: sz :: earr :: [])
    when string_of_mlpath p = "Kuiper.Kernel.Base.obtain_shmem" ->
    // let sz : mlexpr = get_sizet sz in
    // let e_size = get_sizet sized_a in
    ECast (EApp (EQualified ([], "KPR_SHMEM"), [EUnit]),
           TBuf (translate_type env ty))

  (******** KERNEL CALLS ********)

  | MLE_App ({ expr = MLE_Name p }, [
        nblk;
        nthr;
        _pre;
        _post;
        _a;
        sized_a;
        smem_sz;
        _shared_pre;
        _shared_post;
        _setup;
        { expr = MLE_Fun (_, body) };
        _epoch
      ])
    when string_of_mlpath p = "Kuiper.Kernel.Base.launch_kernel_n_m_shmem_async" ->
    let body = hoist env body in
    let hd, args = head_and_args body in
    (* Filter out unit arguments. Not great, not sure why they remain *)
    let args' = List.filter (fun a -> match a.expr with
                                      | MLE_Const MLC_Unit -> false
                                      | _ -> true) args in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "KPR_KCALL") in
    let e_size = get_sizet sized_a in
    let e' =
      with_ty ml_unit_ty <|
        MLE_App (kcall, [ hd;
                          nblk;
                          nthr;
                          e_size;
                          smem_sz ]
                        @ args')
    in
    cb e'

  (* New one! *)
  | MLE_App ({ expr = MLE_Name p }, [ _full_pre; _full_post; kdesc; _epoch ])
    when string_of_mlpath p = "Kuiper.Kernel.Base.launch_kernel_full" ->
    let assoc' k v =
      match List.assoc k v with
      | Some r -> r
      | None -> failwith ("launch_kernel: field not found: " ^ k)
    in
    let nblk, nthr, e_size, smem_sz, hd, rest_args =
      match kdesc.expr with
      | MLE_Record (_, _, fields) ->
        let nblk = assoc' "nblk" fields in
        let nthr = assoc' "nthr" fields in
        let sized_a = assoc' "shmem_type_is_sized" fields in
        let smem_sz = assoc' "shmem_sz" fields in
        let kf = assoc' "f" fields in
        let rec drop_n_binders (e:mlexpr) n =
          match e.expr with
          | MLE_Fun (bs, body) when List.length bs = n -> body
          | MLE_Fun (bs, body) when List.length bs > n ->
            let bs = drop n bs in
            { e with expr = MLE_Fun (bs, body) }
          | MLE_Fun (bs, body) when List.length bs < n ->
            drop_n_binders body (n - List.length bs)
          | _ -> failwith ("launch_kernel: not enough binders: " ^ show e)
        in
        let rec drop_last_n_args (e:mlexpr) n =
          match e.expr with
          | MLE_App (head, args) when List.length args = n -> head
          | MLE_App (head, args) when List.length args > n ->
            let args = drop n args in
            { e with expr = MLE_App (head, args) }
          | MLE_App (head, args) when List.length args < n ->
            drop_last_n_args head (n - List.length args)
          | _ -> failwith ("launch_kernel: not enough arguments: " ^ show e)
        in
        let kf = drop_n_binders kf 4 in
        let kf = hoist env kf in
        let hd, rest_args = head_and_args kf in
        if Nil? rest_args then
          failwith ("launch_kernel: no arguments to kernel: " ^ show kf);
        let e_size = get_sizet sized_a in
        nblk, nthr, e_size, smem_sz, hd, rest_args

      | _ ->
        failwith ("launch_kernel: not a record: " ^ show kdesc)
    in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "KPR_KCALL") in
    let e' =
      with_ty ml_unit_ty <|
        MLE_App (kcall, [ hd; nblk; nthr; e_size; smem_sz ] @ rest_args)
    in
    cb e'

  | MLE_App ({ expr = MLE_Name p }, [
        _unit;
        _epoch
      ])
    when string_of_mlpath p = "Kuiper.Kernel.Base.sync_device" ->
    EApp (EQualified ([], "cudaDeviceSynchronize"), [ EUnit ])

  (* Misc stuff missing from F*? *)
  | MLE_Name p when string_of_mlpath p = "FStar.UInt64.zero" -> EConstant (Krml.UInt64, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt64.one"  -> EConstant (Krml.UInt64, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt32.zero" -> EConstant (Krml.UInt32, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt32.one"  -> EConstant (Krml.UInt32, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt16.zero" -> EConstant (Krml.UInt16, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt16.one"  -> EConstant (Krml.UInt16, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt8.zero"  -> EConstant (Krml.UInt8, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.UInt8.one"   -> EConstant (Krml.UInt8, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.Int64.zero"  -> EConstant (Krml.Int64, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.Int64.one"   -> EConstant (Krml.Int64, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.Int32.zero"  -> EConstant (Krml.Int32, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.Int32.one"   -> EConstant (Krml.Int32, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.Int16.zero"  -> EConstant (Krml.Int16, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.Int16.one"   -> EConstant (Krml.Int16, "1")
  | MLE_Name p when string_of_mlpath p = "FStar.Int8.zero"   -> EConstant (Krml.Int8, "0")
  | MLE_Name p when string_of_mlpath p = "FStar.Int8.one"    -> EConstant (Krml.Int8, "1")

  | _ -> raise NotSupportedByKrmlExtension

let _ =
  register_pre_translate_type_without_decay gpu_translate_type_without_decay;
  register_pre_translate_expr gpu_translate_expr
