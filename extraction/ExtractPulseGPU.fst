module ExtractPulseGPU

friend FStar.Extraction.Krml
open FStar.Extraction.Krml

open FStar
open FStar.Compiler
open FStar.Compiler.Effect
open FStar.List.Tot
open FStar.Extraction
open FStar.Extraction.ML
open FStar.Extraction.ML.Syntax
open FStar.Const

open FStar.Class.Show

exception Failed of string

module BU = FStar.Compiler.Util

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
     p = "GPU.Array.gpu_array"
    )
    ->
      TBuf (translate_type_without_decay env arg1)

  | MLTY_Named ([arg], p) when
    (let p = Syntax.string_of_mlpath p in
     p = "GPU.Ref.gpu_ref"
    )
    ->
      TBuf (translate_type_without_decay env arg)

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

let gpu_translate_expr : translate_expr_t = fun env e ->
  let e = flatten_app e in
  if !dbg
  then BU.print1_warning "ExtractPulse.gpu_translate_expr %s\n" (mlexpr_to_string e);
  let cb = translate_expr env in
  match e.expr with

  (******** PREDEFINED VARS ********)

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasedn])
    when string_of_mlpath p = "GPU.Base.block_idx_x" ->
    EApp (EQualified ([], "blockIdx_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasedn])
    when string_of_mlpath p = "GPU.Base.block_dim_x" ->
    EApp (EQualified ([], "blockDim_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ _unit; _erasedn])
    when string_of_mlpath p = "GPU.Base.thread_idx_x" ->
    EApp (EQualified ([], "threadIdx_x"), [ EUnit ])

  | MLE_App ({ expr = MLE_Name p } , [ sz ])
    when string_of_mlpath p = "GPU.SizeT.sizet_to_u32" ->
    ECast (cb sz, TInt UInt32)

  (******** BARRIERS ********)

  | MLE_App ({ expr = MLE_Name p } , [u1;u2;u3;u4])
    when string_of_mlpath p = "GPU.Barrier.RPM.mbarrier_wait" ->
    EApp (EQualified ([], "__syncthreads"), [ EUnit ])

  (******** REFERENCES ********)

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, [ ty ]) }, [ sz; _unit ])
    when string_of_mlpath p = "GPU.Ref.gpu_alloc0" ->
    let sz : mlexpr = get_sizet sz in
    ECast (EApp (EQualified ([], "PULSE_GPU_ALLOC"), [ cb sz ]),
           TBuf (translate_type env ty))

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, _) }, [ r; _v ])
    when string_of_mlpath p = "GPU.Ref.gpu_free" ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb r])

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, _) }, [ e; _perm; _v ])
    when string_of_mlpath p = "GPU.Ref.gpu_read" ->
    EBufRead (cb e, zero_for_deref)

  | MLE_App ({ expr = MLE_TApp({ expr = MLE_Name p }, _) }, [ e1; e2 ])
    when string_of_mlpath p = "GPU.Ref.gpu_write" ->
    EBufWrite (cb e1, zero_for_deref, cb e2)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: r :: gr :: f :: v :: gv :: [])
    when string_of_mlpath p = "GPU.Ref.gpu_memcpy_host_to_device"->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb gr; cb r; cb sz ; cudaMemcpyHostToDevice ])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: r :: gr :: f :: v :: gv :: [])
    when string_of_mlpath p = "GPU.Ref.gpu_memcpy_device_to_host"->
    let sz : mlexpr = get_sizet sz in
    _MUST <| EApp (EQualified ([], "cudaMemcpy"), [ cb r; cb gr; cb sz; cudaMemcpyDeviceToHost ])

  (******** ARRAY ********)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: len :: [])
    when string_of_mlpath p = "GPU.Array.gpu_array_alloc" ->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb len ]) in
    ECast (EApp (EQualified ([], "PULSE_GPU_ALLOC"), [ bytesize ]),
           TBuf (translate_type env ty))

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: r :: v :: [])
    when string_of_mlpath p = "GPU.Array.gpu_array_free" ->
    _MUST <| EApp (EQualified ([], "cudaFree"), [cb r])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, _) }, sz :: i :: j :: r :: f :: idx :: s :: [])
    when string_of_mlpath p = "GPU.Array.gpu_array_read" ->
    EBufRead (cb r, cb idx)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, _) }, sz :: i :: j :: r :: idx :: v :: s :: [])
    when string_of_mlpath p = "GPU.Array.gpu_array_write" ->
    EBufWrite (cb r, cb idx, cb v)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: elen :: a :: ga :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "GPU.Array.gpu_memcpy_host_to_device"->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    EApp (EQualified ([], "MUST"), [
      EApp (EQualified ([], "cudaMemcpy"), [ cb ga; cb a; bytesize; cudaMemcpyHostToDevice ])
    ])

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sz :: elen :: a :: ga :: cnt :: f :: v :: gv :: [])
    when string_of_mlpath p = "GPU.Array.gpu_memcpy_device_to_host"->
    let sz : mlexpr = get_sizet sz in
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb cnt ]) in
    EApp (EQualified ([], "MUST"), [
      EApp (EQualified ([], "cudaMemcpy"), [ cb a; cb ga; bytesize; cudaMemcpyDeviceToHost ])
    ])

  (******** ATOMIC OPS ********)

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "GPU.AtomicOps.gpu_faa_u32" ->
    (* Can we cast here instead of using a wrapper? *)
    EApp (EQualified ([], "atomic_add_u32"), [cb r; cb v])

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "GPU.AtomicOps.gpu_faa_u64" ->
    EApp (EQualified ([], "atomic_add_u64"), [cb r; cb v])

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "GPU.AtomicOps.gpu_faa_f32" ->
    EApp (EQualified ([], "atomic_add_f32"), [cb r; cb v])

  | MLE_App ({ expr = MLE_Name p }, r :: v :: _ev :: [])
    when string_of_mlpath p = "GPU.AtomicOps.gpu_faa_f64" ->
    EApp (EQualified ([], "atomic_add_f64"), [cb r; cb v])

  (******** OBTAIN SHMEM ********)

  | MLE_App ({ expr = MLE_TApp ({ expr = MLE_Name p }, [ty]) }, sized_a :: sz :: earr :: [])
    when string_of_mlpath p = "GPU.Kernel.obtain_shmem" ->
    // let sz : mlexpr = get_sizet sz in
    // let e_size = get_sizet sized_a in
    ECast (EApp (EQualified ([], "PULSE_SHMEM"), [EUnit]),
           TBuf (translate_type env ty))

  (******** KERNEL CALLS ********)

  | MLE_App ({ expr = MLE_Name p }, [
        _uid;
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
        { expr = MLE_Fun (_, body) }
      ])
    when string_of_mlpath p = "GPU.Kernel.launch_kernel_n_m_sync" ->
    let hd, args = head_and_args body in
    (* Filter out unit arguments. Not great, not sure why they remain *)
    let args' = List.filter (fun a -> match a.expr with
                                      | MLE_Const MLC_Unit -> false
                                      | _ -> true) args in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "PULSE_KCALL_SHMEM") in
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

  | MLE_App ({ expr = MLE_Name p }, [
        _uid;
        nblk;
        nthr;
        _pre;
        _post;
        _barrier;
        { expr = MLE_Fun (_, body) }
      ])
    when string_of_mlpath p = "GPU.Kernel.launch_kernel_n_m_barrier" ->
    let hd, args = head_and_args body in
    (* Filter out unit arguments. Not great, not sure why they remain *)
    let args' = List.filter (fun a -> match a.expr with
                                      | MLE_Const MLC_Unit -> false
                                      | _ -> true) args in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "PULSE_KCALL") in
    let e' =
      with_ty ml_unit_ty <|
        MLE_App (kcall, [ hd;
                          nblk;
                          nthr ]
                        @ args')
    in
    cb e'

  | MLE_App ({ expr = MLE_Name p }, [
        _uid;
        nblk;
        nthr;
        _pre;
        _post;
        { expr = MLE_Fun (_, body) }
      ])
    when string_of_mlpath p = "GPU.Kernel.launch_kernel_n_m" ->
    let hd, args = head_and_args body in
    (* Filter out unit arguments. Not great, not sure why they remain *)
    let args' = List.filter (fun a -> match a.expr with
                                      | MLE_Const MLC_Unit -> false
                                      | _ -> true) args in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "PULSE_KCALL") in
    let e' =
      with_ty ml_unit_ty <|
        MLE_App (kcall, [ hd;
                          nblk;
                          nthr ]
                        @ args')
    in
    cb e'

  // TODO: remove the following two cases (and use the one from above)

  | MLE_App ({ expr = MLE_Name p }, [
        _pre;
        _post;
        { expr = MLE_Fun (_, body) }
      ])
    when string_of_mlpath p = "GPU.Kernel.launch_kernel_1" ->
    let hd, args = head_and_args body in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "PULSE_KCALL") in
    (* Filter out unit arguments. Not great, not sure why they remain *)
    let args' = List.filter (fun a -> match a.expr with
                                    | MLE_Const MLC_Unit -> false
                                    | _ -> true) args in
    let e' =
      with_ty ml_unit_ty <|
        MLE_App (kcall, [ hd;
                          with_ty ml_int_ty <| MLE_Const (MLC_Int ("1", Some (Unsigned, FStar.Const.Int32)));
                          with_ty ml_int_ty <| MLE_Const (MLC_Int ("1", Some (Unsigned, FStar.Const.Int32))) ]
                        @ args')
    in
    cb e'

  | MLE_App ({ expr = MLE_Name p }, [
        _uid;
        nthr;
        _pre;
        _post;
        { expr = MLE_Fun (_, body) }
      ])
    when string_of_mlpath p = "GPU.Kernel.launch_kernel_n" ->
    let hd, args = head_and_args body in
    (* Filter out unit arguments. Not great, not sure why they remain *)
    let args' = List.filter (fun a -> match a.expr with
                                      | MLE_Const MLC_Unit -> false
                                      | _ -> true) args in
    let kcall : mlexpr = with_ty ml_unit_ty <| MLE_Name ([], "PULSE_KCALL") in
    let e' =
      with_ty ml_unit_ty <|
        MLE_App (kcall, [ hd;
                          nthr;
                          with_ty ml_int_ty <| MLE_Const (MLC_Int ("1", Some (Unsigned, FStar.Const.Int32))) ]
                        @ args')
    in
    cb e'

  | _ -> raise NotSupportedByKrmlExtension

let _ =
  register_pre_translate_type_without_decay gpu_translate_type_without_decay;
  register_pre_translate_expr gpu_translate_expr
