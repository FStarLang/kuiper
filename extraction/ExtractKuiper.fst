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

let rec drop n (lst : list 'a) : list 'a =
  match lst with
  | [] -> []
  | _ when n <= 0 -> lst
  | _ -> List.tl (drop (n - 1) lst)

exception Failed of string

module BU = FStarC.Util

let intlit (i : int) : mlexpr =
  with_ty ml_int_ty <| MLE_Const (MLC_Int (show i, None))

let ml_uint8      = MLTY_Named ([], (["FStar"; "UInt8"], "t"))
let ml_bytearr    = MLTY_Named ([ml_uint8], (["FStar"; "Buffer"], "t"))
let ml_sizet      = MLTY_Named ([], (["FStar"; "SizeT"], "t"))
let ml_fun t1 t2  = MLTY_Fun (t1, E_PURE, t2)

let sizet_add x y =
  let fv = with_ty ml_unit_ty <| MLE_Name (["FStar"; "SizeT"], "add") in
  with_ty ml_sizet <| MLE_App (fv, [x; y])

let sizet_mul x y =
  let fv = with_ty ml_unit_ty <| MLE_Name (["FStar"; "SizeT"], "mul") in
  with_ty ml_sizet <| MLE_App (fv, [x; y])

let sizet_lit (i : int) : mlexpr =
  let mk = with_ty ml_int_ty <| MLE_Name (["FStar"; "SizeT"], "uint_to_t") in
  with_ty ml_sizet <| MLE_App (mk, [intlit i])

let mk_tuple2 x y : mlexpr =
  with_ty ml_unit_ty <| MLE_Tuple [x; y]

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
    raise_error0 Fatal_ExtractionUnsupported [
      text "Hoist: free variables do not match:" ^/^ pp e0;
      text "fvs0 =" ^/^ pp fvs0;
      text "fvs =" ^/^ pp fvs;
    ];
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
    Krml.Comment ("  hoisted when extracting " ^ BU.dflt "<unknown>" !krml_current_decl);
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


(* head fv, type args, and args *)
let hta (e : mlexpr) : option (string & list mlty & list mlexpr) =
  (* there is probably no need for these two to recurse. *)
  let rec get_args (e : mlexpr) : mlexpr & list mlexpr =
    match e.expr with
    | MLE_App (e', args') ->
      let e'', args'' = get_args e' in
      e'', args'' @ args'
    | _ -> e, []
  in
  let rec get_tyargs (e : mlexpr) : mlexpr & list mlty =
    match e.expr with
    | MLE_TApp (e', args') ->
      let e'', args'' = get_tyargs e' in
      e'', args'' @ args'
    | _ -> e, []
  in
  let e, args = get_args e in
  let e, tyargs = get_tyargs e in
  match e.expr with
  | MLE_Name p -> Some (string_of_mlpath p, tyargs, args)
  | _ -> None

(* Substitutes the variable [v] in the expression [e] with the expression [e'].
   i.e e[v := e']. *)
let rec ml_visit (pre post : mlexpr -> mlexpr) (e : mlexpr) : mlexpr =
  let e = pre e in
  let e =
    match e.expr with
    | MLE_Const _ -> e
    | MLE_Var _ -> e
    | MLE_Name _ -> e
    | MLE_Let ((flavor, lbs), e2) ->
      let lbs' = lbs |> List.map (fun lb -> {lb with mllb_def = ml_visit pre post lb.mllb_def}) in
      let e2' = ml_visit pre post e2 in
      { e with expr = MLE_Let ((flavor, lbs'), e2') }
    | MLE_App (head, args) ->
      let head' = ml_visit pre post head in
      let args' = List.map (fun arg -> ml_visit pre post arg) args in
      { e with expr = MLE_App (head', args') }
    | MLE_TApp (head, tyargs) ->
      let head' = ml_visit pre post head in
      { e with expr = MLE_TApp (head', tyargs) }
    | MLE_Fun (bs, e2) ->
      (* fully named, no clashses should occur. *)
      let e2' = ml_visit pre post e2 in
      { e with expr = MLE_Fun (bs, e2') }
    | MLE_Match (e1, branches) ->
      let e1' = ml_visit pre post e1 in
      let branches' =
        branches |> List.map (fun (p, e2, e3) ->
          let e2' = BU.map_opt e2 (fun e2 -> ml_visit pre post e2) in
          let e3' = ml_visit pre post e3 in
          (p, e2', e3')
        )
      in
      { e with expr = MLE_Match (e1', branches') }
    | MLE_Coerce (e1, t1, t2) ->
      let e1' = ml_visit pre post e1 in
      { e with expr = MLE_Coerce (e1', t1, t2) }
    | MLE_Seq es ->
      let es' = List.map (fun e -> ml_visit pre post e) es in
      { e with expr = MLE_Seq es' }
    | MLE_Tuple es ->
      let es' = List.map (fun e -> ml_visit pre post e) es in
      { e with expr = MLE_Tuple es' }
    | MLE_Record (p, t, fields) ->
      let fields' = List.map (fun (f, e) -> f, ml_visit pre post e) fields in
      { e with expr = MLE_Record (p, t, fields') }
    | MLE_Proj (e1, f) ->
      let e1' = ml_visit pre post e1 in
      { e with expr = MLE_Proj (e1', f) }
    | MLE_If (e1, e2, e3) ->
      let e1' = ml_visit pre post e1 in
      let e2' = ml_visit pre post e2 in
      let e3' = BU.map_opt e3 (fun e3 -> ml_visit pre post e3) in
      { e with expr = MLE_If (e1', e2', e3') }
    | MLE_Raise (p, args) ->
      let args' = List.map (fun arg -> ml_visit pre post arg) args in
      { e with expr = MLE_Raise (p, args') }
    | MLE_Try (e1, branches) ->
      let e1' = ml_visit pre post e1 in
      let branches' =
        branches |> List.map (fun (p, e2, e3) ->
          let e2' = BU.map_opt e2 (fun e2 -> ml_visit pre post e2) in
          let e3' = ml_visit pre post e3 in
          (p, e2', e3')
        )
      in
      { e with expr = MLE_Try (e1', branches') }
    | MLE_CTor (p, args) ->
      let args' = List.map (fun arg -> ml_visit pre post arg) args in
      { e with expr = MLE_CTor (p, args') }
  in
  post e

(* Substitutes the variable [v] in the expression [e] with the expression [e'].
   i.e e[v := e']. *)
let ml_subst (e : mlexpr) (v : mlident) (e' : mlexpr) : mlexpr =
  let subst1 (e : mlexpr) : mlexpr =
    match e.expr with
    | MLE_Var v' when v = v' -> e'
    | _ -> e
  in
  ml_visit subst1 id e

let collapse_tuple_proj (e : mlexpr) : mlexpr =
  let subst1 (e0 : mlexpr) : mlexpr =
    match e0.expr with
    | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name f }, _) }, [e]) when string_of_mlpath f = "FStar.Pervasives.Native.fst" -> (
      let e = unmagic e in
      match e.expr with
      | MLE_Tuple [x;y] ->
        x
      | _ -> e0
    )
    | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name f }, _) }, [e]) when string_of_mlpath f = "FStar.Pervasives.Native.snd" -> (
      let e = unmagic e in
      match e.expr with
      | MLE_Tuple [x;y] ->
        y
      | _ -> e0
    )
    | _ -> e0
  in
  ml_visit id subst1 e

let is_lid (s:string) (e : mlexpr) : bool =
  match e.expr with
  | MLE_Name p -> (string_of_mlpath p) = s
  | MLE_Var v -> v = s
  | _ -> false

let rec mlexpr_as_list (e : mlexpr) : option (list mlexpr) =
  let open FStarC.Class.Monad in
  match e.expr with
  | MLE_CTor (fv, []) when string_of_mlpath fv = "Prims.Nil" ->
    return []
  | MLE_CTor (fv, [hd; tl]) when string_of_mlpath fv = "Prims.Cons" ->
    let! tl' = mlexpr_as_list tl in
    return (hd :: tl')
  | _ ->
    BU.print1 "Not a list: %s\n" (mlexpr_to_string e);
    None

(* Returns (possibly) the size of each element and length of array.
The type of the array has already been erased, we cannot get it here. *)
let parse_shmem_desc (e : mlexpr) : option (mlexpr & mlexpr) =
  let open FStarC.Class.Monad in
  match e.expr with
  | MLE_CTor (fv, [_ty; sized; len]) when string_of_mlpath fv = "Kuiper.Kernel.Desc.SHArray" ->
    let sized_a = get_sizet sized in
    return (sized_a, len)
  | _ ->
    BU.print1 "Not a shmem_desc: %s\n" (mlexpr_to_string e);
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
      // BU.print1 "GG mlexpr shmems: %s\n" (show shmems_desc);
      let! parsed : list mlexpr = mlexpr_as_list shmems_desc in
      let! parsed : list (mlexpr & mlexpr) = mapM parse_shmem_desc parsed in
      // BU.print1 "GG mlexpr shmems parsed: %s\n" (show parsed);
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
      // BU.print1 "GGG computed c_sh = %s\n" (show c_sh);
      let kf = assoc' "f" fields in
      // BU.print1 "GGG kf = %s\n" (mlexpr_to_string kf);
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
        (* |> with_ty (MLTY_Fun (ml_unit_ty, E_IMPURE, MLTY_Var "FStar.SizeT.t")) *)
        (* |> (fun x -> MLE_App (x, [ml_unit])) *)
        |> with_ty (MLTY_Var "FStar.SizeT.t")
      in
      let ml_threadidx : mlexpr =
        MLE_Name ([], "threadIdx_x")
        (* |> with_ty (MLTY_Fun (ml_unit_ty, E_IMPURE, MLTY_Var "FStar.SizeT.t")) *)
        (* |> (fun x -> MLE_App (x, [ml_unit])) *)
        |> with_ty (MLTY_Var "FStar.SizeT.t") // fixme should be MLTY_Named
      in
      let remove_stupid_let (e : mlexpr) : mlexpr =
        match e.expr with
        | MLE_Let ((NonRec, [lb]), e) ->
          let { mllb_name = id; mllb_def = e' } = lb in
          ml_subst e id e'
        | _ -> failwith ("remove_stupid_let: not there. " ^ show e)
      in
      let kf = apply_lam kf c_sh in
      // let kf = remove_stupid_let kf in // ???
      // FIXME: concretizing the shmem argument does not work, for some reason
      // it shows up as erased in the original MLexpr.
      // let kf = apply_lam kf ml_unit in
      let kf = apply_lam kf ml_blockidx in
      let kf = apply_lam kf ml_threadidx in
      let kf = apply_lam kf ml_unit in
      let kf = collapse_tuple_proj kf in
      let kf = hoist env kf in
      let hd, rest_args = head_and_args kf in
      if Nil? rest_args then // is this really a problem?
        failwith ("launch_kernel: no arguments to kernel: " ^ show kf);
      // let e_size = get_sizet sized_a in
      return (nblk, nthr, shmem_bytesz, hd, rest_args)

    | _ ->
      failwith ("launch_kernel: not a record: " ^ show kdesc)
  in
  let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "KPR_KCALL") in
  let e' =
  with_ty ml_unit_ty <|
    MLE_App (kcall, [ hd; nblk; nthr; smem_bytesz ] @ rest_args)
  in
  // BU.print1_warning "New kcall: %s\n" (show e');
  return e'


let gpu_translate_expr : translate_expr_t = fun env e ->
  let e = flatten_app e in
  if !dbg
  then BU.print1_warning "ExtractKuiper.gpu_translate_expr %s\n" (mlexpr_to_string e);
  let cb = translate_expr env in
  let x = hta e in
  if None? x then raise NotSupportedByKrmlExtension;
  match Some?.v x with
  (******** ASSERTIONS ********)
  | "Kuiper.Assert.dassert", [], [ x ] ->
    if x.expr = MLE_Const (MLC_Bool true)
    then EUnit
    else
      EApp (EQualified ([], "KPR_ASSERT"), [ cb x ])

  | "Kuiper.Assert.dguard", [], [ x ] ->
    if x.expr = MLE_Const (MLC_Bool true)
    then EUnit
    else
      EApp (EQualified ([], "KPR_GUARD"), [ cb x ])

  (******** SIZET, missing from F* ********)
  | "Kuiper.SizeT.sizet_and", [], [ x; y ] ->
    EApp (EOp (BAnd, SizeT), [ cb x; cb y ])

  (******** PREDEFINED VARS ********)

  | "Kuiper.Base.get_gdim", [], [ _unit; _erasednblk; _erasednbid ] ->
    EApp (EQualified ([], "gridDim_x"), [ EUnit ])

  | "Kuiper.Base.get_bdim", [], [ _unit; _erasednthr; _erasedntid ] ->
    EApp (EQualified ([], "blockDim_x"), [ EUnit ])

  | "Kuiper.SizeT.sizet_to_u32", [], [ sz ] ->
    ECast (cb sz, TInt UInt32)

  (******** BARRIERS ********)

  | "Kuiper.Barrier.barrier_wait", [], [ _unit; _n; _p; _q; _it; _tid ] ->
    EApp (EQualified ([], "__syncthreads"), [ EUnit ])

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
    EBufRead (cb e, zero_for_deref)

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


  (******** ATOMIC OPS ********)

  | "Kuiper.AtomicOps.gpu_faa_u32", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_u32"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_u64", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_u64"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_f32", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_f32"), [cb r; cb v])
  | "Kuiper.AtomicOps.gpu_faa_f64", [], [ r; v; _ev ] -> EApp (EQualified ([], "atomic_add_f64"), [cb r; cb v])

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

  | _ -> raise NotSupportedByKrmlExtension

let _ =
  register_pre_translate_type_without_decay gpu_translate_type_without_decay;
  register_pre_translate_expr gpu_translate_expr
