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

let gpu_translate_expr : translate_expr_t = fun env e ->
  let e = flatten_app e in
  if !dbg
  then BU.print1_warning "ExtractPulse.gpu_translate_expr %s\n" (mlexpr_to_string e);
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
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ cb sz ]),
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
    let bytesize : expr = EApp (EOp (Mult, SizeT), [ cb sz; cb len ]) in
    ECast (EApp (EQualified ([], "KPR_GPU_ALLOC"), [ bytesize ]),
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

  | MLE_App ({ expr = MLE_Name p }, [
        _unit;
        _epoch
      ])
    when string_of_mlpath p = "Kuiper.Kernel.Base.sync" ->
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
