module ExtractionUtils

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

let rec unmagic (e : mlexpr) : mlexpr =
  match e.expr with
  | MLE_Coerce (e, _, _) -> unmagic e
  | _ -> e

(* head term, type args, and args *)
let xta (e : mlexpr) : mlexpr & list mlty & list mlexpr =
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
  (e, tyargs, args)

(* head fv , type args, and args *)
let hta (e : mlexpr) : option (string & list mlty & list mlexpr) =
  let h, tyargs, args = xta e in
  match h.expr with
  | MLE_Name p -> Some (string_of_mlpath p, tyargs, args)
  | _ -> None

(* similar, but for a constructor at the head *)
let cta (e : mlexpr) : ML (option (string & list mlty & list mlexpr)) =
  let h, tyargs, args = xta e in
  match h.expr with
  | MLE_CTor (p, args') ->
    if Cons? args then
      failwith "I don't think this happens.";
    Some (string_of_mlpath p, tyargs, args' @ args)
  | _ -> None

let type_hta (e : mlty) : option (string & list mlty) =
  match e with
  | MLTY_Named (args, p) -> Some (string_of_mlpath p, args)
  | _ -> None


(* Substitutes the variable [v] in the expression [e] with the expression [e'].
   i.e e[v := e']. *)
let rec ml_visit (pre post : mlexpr -> ML mlexpr) (e : mlexpr) : ML mlexpr =
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
          let e2' = Option.map (fun e2 -> ml_visit pre post e2) e2 in
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
      let e3' = Option.map (fun e3 -> ml_visit pre post e3) e3 in
      { e with expr = MLE_If (e1', e2', e3') }
    | MLE_Raise (p, args) ->
      let args' = List.map (fun arg -> ml_visit pre post arg) args in
      { e with expr = MLE_Raise (p, args') }
    | MLE_Try (e1, branches) ->
      let e1' = ml_visit pre post e1 in
      let branches' =
        branches |> List.map (fun (p, e2, e3) ->
          let e2' = Option.map (fun e2 -> ml_visit pre post e2) e2 in
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
let ml_subst (e : mlexpr) (v : mlident) (e' : mlexpr) : ML mlexpr =
  let subst1 (e : mlexpr) : mlexpr =
    match e.expr with
    | MLE_Var v' when v = v' -> e'
    | _ -> e
  in
  ml_visit subst1 id e

let collapse_tuple_proj (e : mlexpr) : ML mlexpr =
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

// This stage is essential to allow to match on shared memory descriptors
// like: let (ar1, (ar2, _)) = sh in ...
// Otherwise karamel complains about a cast into Top remaining.
let collapse_tuple_matches (e : mlexpr) : ML mlexpr =
  let rec subst1 (e0 : mlexpr) : ML mlexpr =
    match e0.expr with
    | MLE_Let ((NonRec, [{ mllb_name = v; mllb_def = def; }]), body) -> (
      let is_var v e =
        match (unmagic e).expr with
        | MLE_Var v' -> v = v'
        | _ -> false
      in
      match body.expr with
      | MLE_Match (sc, _) when is_var v sc ->
        subst1 <| ml_subst body v def
      | _ -> e0
    )

    | MLE_Match (sc, [b]) -> (
      let sc = unmagic sc in
      match sc.expr with
      | MLE_Tuple [x; y] -> (
        match b with
        | MLP_Tuple [MLP_Var v1; MLP_Var v2], None, body ->
          let e' = ml_subst body v1 x in
          let e' = ml_subst e' v2 y in
          e'
        | MLP_Tuple [MLP_Var v; p], None, body ->
          let e' = ml_subst body v x in
          let e' = with_ty e0.mlty (MLE_Match (y, [p, None, e'])) in
          subst1 e'
        | _ -> e0
      )
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

let rec mlexpr_as_list (e : mlexpr) : ML (option (list mlexpr)) =
  let open FStarC.Class.Monad in
  match e.expr with
  | MLE_CTor (fv, []) when string_of_mlpath fv = "Prims.Nil" ->
    return []
  | MLE_CTor (fv, [hd; tl]) when string_of_mlpath fv = "Prims.Cons" ->
    let! tl' = mlexpr_as_list tl in
    return (hd :: tl')
  | _ ->
    Format.print1 "Not a list: %s\n" (mlexpr_to_string e);
    None

let intlit (i : int) : ML mlexpr =
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

let sizet_lit (i : int) : ML mlexpr =
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

let head_and_args (e : mlexpr) : mlexpr & list mlexpr =
  let rec aux acc e =
    match e.expr with
    | MLE_App (head, args) -> aux (args @ acc) head
    | _ -> (e, acc)
  in
  aux [] e

let rec freevars_of_mlexpr (e : mlexpr) : ML (list (string & mlty)) =
  let remove (ks : list string) (vs : list (string & mlty)) : ML (list (string & mlty)) =
    List.filter (fun (v, _) -> not (List.existsb (fun k -> v = k) ks)) vs
  in
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
      let rec pat_bound (p:mlpattern) : ML (list string) =
        match p with
        | MLP_Var v -> [v]
        | MLP_CTor (_, args) -> List.collect pat_bound args
        | MLP_Record (_, fields) ->
          List.concatMap (fun (_, p) -> pat_bound p) fields
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
