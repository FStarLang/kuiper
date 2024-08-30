open Prims
exception Failed of Prims.string 
let (uu___is_Failed : Prims.exn -> Prims.bool) =
  fun projectee ->
    match projectee with | Failed uu___ -> true | uu___ -> false
let (__proj__Failed__item__uu___ : Prims.exn -> Prims.string) =
  fun projectee -> match projectee with | Failed uu___ -> uu___
let (flatten_app :
  FStar_Extraction_ML_Syntax.mlexpr -> FStar_Extraction_ML_Syntax.mlexpr) =
  fun e ->
    let rec aux args e1 =
      match e1.FStar_Extraction_ML_Syntax.expr with
      | FStar_Extraction_ML_Syntax.MLE_App (head, args0) ->
          aux (FStar_List_Tot_Base.op_At args0 args) head
      | uu___ ->
          (match args with
           | [] -> e1
           | uu___1 ->
               {
                 FStar_Extraction_ML_Syntax.expr =
                   (FStar_Extraction_ML_Syntax.MLE_App (e1, args));
                 FStar_Extraction_ML_Syntax.mlty =
                   (e1.FStar_Extraction_ML_Syntax.mlty);
                 FStar_Extraction_ML_Syntax.loc =
                   (e1.FStar_Extraction_ML_Syntax.loc)
               }) in
    aux [] e
let (dbg : Prims.bool FStar_Compiler_Effect.ref) =
  FStar_Compiler_Debug.get_toggle "extraction.gpu"
let (gpu_translate_type_without_decay :
  FStar_Extraction_Krml.translate_type_without_decay_t) =
  fun env ->
    fun t ->
      match t with
      | FStar_Extraction_ML_Syntax.MLTY_Named (arg1::arg2::[], p) when
          let p1 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
          p1 = "GPU.Array.gpu_array" ->
          let uu___ =
            FStar_Extraction_Krml.translate_type_without_decay env arg1 in
          FStar_Extraction_Krml.TBuf uu___
      | FStar_Extraction_ML_Syntax.MLTY_Named (arg::[], p) when
          let p1 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
          p1 = "GPU.Ref.gpu_ref" ->
          let uu___ =
            FStar_Extraction_Krml.translate_type_without_decay env arg in
          FStar_Extraction_Krml.TBuf uu___
      | uu___ ->
          FStar_Compiler_Effect.raise
            FStar_Extraction_Krml.NotSupportedByKrmlExtension
let (head_and_args :
  FStar_Extraction_ML_Syntax.mlexpr ->
    (FStar_Extraction_ML_Syntax.mlexpr * FStar_Extraction_ML_Syntax.mlexpr
      Prims.list))
  =
  fun e ->
    let rec aux acc e1 =
      match e1.FStar_Extraction_ML_Syntax.expr with
      | FStar_Extraction_ML_Syntax.MLE_App (head, args) ->
          aux (FStar_List_Tot_Base.op_At args acc) head
      | uu___ -> (e1, acc) in
    aux [] e
let (escape_hatch : Prims.string -> FStar_Extraction_Krml.expr) =
  fun s ->
    FStar_Extraction_Krml.EComment
      ("",
        (FStar_Extraction_Krml.EConstant (FStar_Extraction_Krml.UInt32, "0")),
        (Prims.strcat "*/ + " (Prims.strcat s " /* ")))
let (zero_for_deref : FStar_Extraction_Krml.expr) =
  FStar_Extraction_Krml.EQualified (["C"], "_zero_for_deref")
let (cudaMemcpyDeviceToHost : FStar_Extraction_Krml.expr) =
  FStar_Extraction_Krml.EQualified ([], "cudaMemcpyDeviceToHost")
let (cudaMemcpyHostToDevice : FStar_Extraction_Krml.expr) =
  FStar_Extraction_Krml.EQualified ([], "cudaMemcpyHostToDevice")
let (get_sizet :
  FStar_Extraction_ML_Syntax.mlexpr -> FStar_Extraction_ML_Syntax.mlexpr) =
  fun e ->
    match e.FStar_Extraction_ML_Syntax.expr with
    | FStar_Extraction_ML_Syntax.MLE_Record (uu___, uu___1, (uu___2, sz)::[])
        -> sz
    | uu___ ->
        let uu___1 =
          let uu___2 =
            let uu___3 =
              FStar_Class_Show.show FStar_Extraction_ML_Code.showable_mlexpr
                e in
            Prims.strcat "Expected a single-field record for the size, got: "
              uu___3 in
          Failed uu___2 in
        FStar_Compiler_Effect.raise uu___1
let (gpu_translate_expr : FStar_Extraction_Krml.translate_expr_t) =
  fun env ->
    fun e ->
      let e1 = flatten_app e in
      (let uu___1 = FStar_Compiler_Effect.op_Bang dbg in
       if uu___1
       then
         let uu___2 = FStar_Extraction_ML_Syntax.mlexpr_to_string e1 in
         FStar_Compiler_Util.print1_warning
           "ExtractPulse.gpu_translate_expr %s\n" uu___2
       else ());
      (let cb = FStar_Extraction_Krml.translate_expr env in
       match e1.FStar_Extraction_ML_Syntax.expr with
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _unit::_erasedn::[])
           when
           let uu___3 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___3 = "GPU.Base.block_idx_x" -> escape_hatch "blockIdx.x"
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _unit::_erasedn::[])
           when
           let uu___3 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___3 = "GPU.Base.block_dim_x" -> escape_hatch "blockDim.x"
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _unit::_erasedn::[])
           when
           let uu___3 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___3 = "GPU.Base.thread_idx_x" -> escape_hatch "threadIdx.x"
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            sz::[])
           when
           let uu___3 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___3 = "GPU.SizeT.sizet_to_u32" ->
           let uu___3 =
             let uu___4 = cb sz in
             (uu___4,
               (FStar_Extraction_Krml.TInt FStar_Extraction_Krml.UInt32)) in
           FStar_Extraction_Krml.ECast uu___3
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            u1::u2::u3::u4::[])
           when
           let uu___3 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___3 = "GPU.Barrier.RPM.mbarrier_wait" ->
           FStar_Extraction_Krml.EApp
             ((FStar_Extraction_Krml.EQualified ([], "__syncthreads")),
               [FStar_Extraction_Krml.EUnit])
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::_unit::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Ref.gpu_alloc0" ->
           let sz1 = get_sizet sz in
           let uu___5 =
             let uu___6 =
               let uu___7 =
                 let uu___8 = let uu___9 = cb sz1 in [uu___9] in
                 ((FStar_Extraction_Krml.EQualified ([], "PULSE_GPU_ALLOC")),
                   uu___8) in
               FStar_Extraction_Krml.EApp uu___7 in
             let uu___7 =
               let uu___8 = FStar_Extraction_Krml.translate_type env ty in
               FStar_Extraction_Krml.TBuf uu___8 in
             (uu___6, uu___7) in
           FStar_Extraction_Krml.ECast uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 uu___3);
              FStar_Extraction_ML_Syntax.mlty = uu___4;
              FStar_Extraction_ML_Syntax.loc = uu___5;_},
            r::_v::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Ref.gpu_free" ->
           let uu___6 =
             let uu___7 =
               let uu___8 =
                 let uu___9 =
                   let uu___10 = let uu___11 = cb r in [uu___11] in
                   ((FStar_Extraction_Krml.EQualified ([], "cudaFree")),
                     uu___10) in
                 FStar_Extraction_Krml.EApp uu___9 in
               [uu___8] in
             ((FStar_Extraction_Krml.EQualified ([], "MUST")), uu___7) in
           FStar_Extraction_Krml.EApp uu___6
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 uu___3);
              FStar_Extraction_ML_Syntax.mlty = uu___4;
              FStar_Extraction_ML_Syntax.loc = uu___5;_},
            e2::_perm::_v::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Ref.gpu_read" ->
           let uu___6 = let uu___7 = cb e2 in (uu___7, zero_for_deref) in
           FStar_Extraction_Krml.EBufRead uu___6
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 uu___3);
              FStar_Extraction_ML_Syntax.mlty = uu___4;
              FStar_Extraction_ML_Syntax.loc = uu___5;_},
            e11::e2::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Ref.gpu_write" ->
           let uu___6 =
             let uu___7 = cb e11 in
             let uu___8 = cb e2 in (uu___7, zero_for_deref, uu___8) in
           FStar_Extraction_Krml.EBufWrite uu___6
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::r::gr::f::v::gv::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Ref.gpu_memcpy_host_to_device" ->
           let sz1 = get_sizet sz in
           let uu___5 =
             let uu___6 =
               let uu___7 =
                 let uu___8 =
                   let uu___9 =
                     let uu___10 = cb gr in
                     let uu___11 =
                       let uu___12 = cb r in
                       let uu___13 =
                         let uu___14 = cb sz1 in
                         [uu___14; cudaMemcpyHostToDevice] in
                       uu___12 :: uu___13 in
                     uu___10 :: uu___11 in
                   ((FStar_Extraction_Krml.EQualified ([], "cudaMemcpy")),
                     uu___9) in
                 FStar_Extraction_Krml.EApp uu___8 in
               [uu___7] in
             ((FStar_Extraction_Krml.EQualified ([], "MUST")), uu___6) in
           FStar_Extraction_Krml.EApp uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::r::gr::f::v::gv::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Ref.gpu_memcpy_device_to_host" ->
           let sz1 = get_sizet sz in
           let uu___5 =
             let uu___6 =
               let uu___7 =
                 let uu___8 =
                   let uu___9 =
                     let uu___10 = cb r in
                     let uu___11 =
                       let uu___12 = cb gr in
                       let uu___13 =
                         let uu___14 = cb sz1 in
                         [uu___14; cudaMemcpyDeviceToHost] in
                       uu___12 :: uu___13 in
                     uu___10 :: uu___11 in
                   ((FStar_Extraction_Krml.EQualified ([], "cudaMemcpy")),
                     uu___9) in
                 FStar_Extraction_Krml.EApp uu___8 in
               [uu___7] in
             ((FStar_Extraction_Krml.EQualified ([], "MUST")), uu___6) in
           FStar_Extraction_Krml.EApp uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::len::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Array.gpu_array_alloc" ->
           let sz1 = get_sizet sz in
           let bytesize =
             let uu___5 =
               let uu___6 =
                 let uu___7 = cb sz1 in
                 let uu___8 = let uu___9 = cb len in [uu___9] in uu___7 ::
                   uu___8 in
               ((FStar_Extraction_Krml.EOp
                   (FStar_Extraction_Krml.Mult, FStar_Extraction_Krml.SizeT)),
                 uu___6) in
             FStar_Extraction_Krml.EApp uu___5 in
           let uu___5 =
             let uu___6 =
               let uu___7 = FStar_Extraction_Krml.translate_type env ty in
               FStar_Extraction_Krml.TBuf uu___7 in
             ((FStar_Extraction_Krml.EApp
                 ((FStar_Extraction_Krml.EQualified ([], "PULSE_GPU_ALLOC")),
                   [bytesize])), uu___6) in
           FStar_Extraction_Krml.ECast uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::r::v::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Array.gpu_array_free" ->
           let uu___5 =
             let uu___6 =
               let uu___7 =
                 let uu___8 =
                   let uu___9 = let uu___10 = cb r in [uu___10] in
                   ((FStar_Extraction_Krml.EQualified ([], "cudaFree")),
                     uu___9) in
                 FStar_Extraction_Krml.EApp uu___8 in
               [uu___7] in
             ((FStar_Extraction_Krml.EQualified ([], "MUST")), uu___6) in
           FStar_Extraction_Krml.EApp uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 uu___3);
              FStar_Extraction_ML_Syntax.mlty = uu___4;
              FStar_Extraction_ML_Syntax.loc = uu___5;_},
            sz::i::j::r::f::idx::s::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Array.gpu_array_read" ->
           let uu___6 =
             let uu___7 = cb r in let uu___8 = cb idx in (uu___7, uu___8) in
           FStar_Extraction_Krml.EBufRead uu___6
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 uu___3);
              FStar_Extraction_ML_Syntax.mlty = uu___4;
              FStar_Extraction_ML_Syntax.loc = uu___5;_},
            sz::i::j::r::idx::v::s::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Array.gpu_array_write" ->
           let uu___6 =
             let uu___7 = cb r in
             let uu___8 = cb idx in
             let uu___9 = cb v in (uu___7, uu___8, uu___9) in
           FStar_Extraction_Krml.EBufWrite uu___6
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::elen::a::ga::cnt::f::v::gv::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Array.gpu_memcpy_host_to_device" ->
           let sz1 = get_sizet sz in
           let bytesize =
             let uu___5 =
               let uu___6 =
                 let uu___7 = cb sz1 in
                 let uu___8 = let uu___9 = cb cnt in [uu___9] in uu___7 ::
                   uu___8 in
               ((FStar_Extraction_Krml.EOp
                   (FStar_Extraction_Krml.Mult, FStar_Extraction_Krml.SizeT)),
                 uu___6) in
             FStar_Extraction_Krml.EApp uu___5 in
           let uu___5 =
             let uu___6 =
               let uu___7 =
                 let uu___8 =
                   let uu___9 =
                     let uu___10 = cb ga in
                     let uu___11 =
                       let uu___12 = cb a in
                       [uu___12; bytesize; cudaMemcpyHostToDevice] in
                     uu___10 :: uu___11 in
                   ((FStar_Extraction_Krml.EQualified ([], "cudaMemcpy")),
                     uu___9) in
                 FStar_Extraction_Krml.EApp uu___8 in
               [uu___7] in
             ((FStar_Extraction_Krml.EQualified ([], "MUST")), uu___6) in
           FStar_Extraction_Krml.EApp uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_TApp
                ({
                   FStar_Extraction_ML_Syntax.expr =
                     FStar_Extraction_ML_Syntax.MLE_Name p;
                   FStar_Extraction_ML_Syntax.mlty = uu___1;
                   FStar_Extraction_ML_Syntax.loc = uu___2;_},
                 ty::[]);
              FStar_Extraction_ML_Syntax.mlty = uu___3;
              FStar_Extraction_ML_Syntax.loc = uu___4;_},
            sz::elen::a::ga::cnt::f::v::gv::[])
           when
           let uu___5 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___5 = "GPU.Array.gpu_memcpy_device_to_host" ->
           let sz1 = get_sizet sz in
           let bytesize =
             let uu___5 =
               let uu___6 =
                 let uu___7 = cb sz1 in
                 let uu___8 = let uu___9 = cb cnt in [uu___9] in uu___7 ::
                   uu___8 in
               ((FStar_Extraction_Krml.EOp
                   (FStar_Extraction_Krml.Mult, FStar_Extraction_Krml.SizeT)),
                 uu___6) in
             FStar_Extraction_Krml.EApp uu___5 in
           let uu___5 =
             let uu___6 =
               let uu___7 =
                 let uu___8 =
                   let uu___9 =
                     let uu___10 = cb a in
                     let uu___11 =
                       let uu___12 = cb ga in
                       [uu___12; bytesize; cudaMemcpyDeviceToHost] in
                     uu___10 :: uu___11 in
                   ((FStar_Extraction_Krml.EQualified ([], "cudaMemcpy")),
                     uu___9) in
                 FStar_Extraction_Krml.EApp uu___8 in
               [uu___7] in
             ((FStar_Extraction_Krml.EQualified ([], "MUST")), uu___6) in
           FStar_Extraction_Krml.EApp uu___5
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            r::v::_ev::[])
           when
           let uu___3 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___3 = "GPU.AtomicOps.gpu_faa_u64" ->
           let uu___3 =
             let uu___4 =
               let uu___5 = cb r in
               let uu___6 = let uu___7 = cb v in [uu___7] in uu___5 :: uu___6 in
             ((FStar_Extraction_Krml.EQualified ([], "atomicAdd")), uu___4) in
           FStar_Extraction_Krml.EApp uu___3
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _uid::nblk::nthr::_pre::_post::_barrier::{
                                                       FStar_Extraction_ML_Syntax.expr
                                                         =
                                                         FStar_Extraction_ML_Syntax.MLE_Fun
                                                         (uu___3, body);
                                                       FStar_Extraction_ML_Syntax.mlty
                                                         = uu___4;
                                                       FStar_Extraction_ML_Syntax.loc
                                                         = uu___5;_}::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Kernel.launch_kernel_n_m_barrier" ->
           let uu___6 = head_and_args body in
           (match uu___6 with
            | (hd, args) ->
                let args' =
                  FStar_Compiler_List.filter
                    (fun a ->
                       match a.FStar_Extraction_ML_Syntax.expr with
                       | FStar_Extraction_ML_Syntax.MLE_Const
                           (FStar_Extraction_ML_Syntax.MLC_Unit) -> false
                       | uu___7 -> true) args in
                let kcall =
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty
                    (FStar_Extraction_ML_Syntax.MLE_Name ([], "PULSE_KCALL")) in
                let e' =
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty
                    (FStar_Extraction_ML_Syntax.MLE_App
                       (kcall,
                         (FStar_List_Tot_Base.op_At [hd; nblk; nthr] args'))) in
                cb e')
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _uid::nblk::nthr::_pre::_post::{
                                             FStar_Extraction_ML_Syntax.expr
                                               =
                                               FStar_Extraction_ML_Syntax.MLE_Fun
                                               (uu___3, body);
                                             FStar_Extraction_ML_Syntax.mlty
                                               = uu___4;
                                             FStar_Extraction_ML_Syntax.loc =
                                               uu___5;_}::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Kernel.launch_kernel_n_m" ->
           let uu___6 = head_and_args body in
           (match uu___6 with
            | (hd, args) ->
                let args' =
                  FStar_Compiler_List.filter
                    (fun a ->
                       match a.FStar_Extraction_ML_Syntax.expr with
                       | FStar_Extraction_ML_Syntax.MLE_Const
                           (FStar_Extraction_ML_Syntax.MLC_Unit) -> false
                       | uu___7 -> true) args in
                let kcall =
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty
                    (FStar_Extraction_ML_Syntax.MLE_Name ([], "PULSE_KCALL")) in
                let e' =
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty
                    (FStar_Extraction_ML_Syntax.MLE_App
                       (kcall,
                         (FStar_List_Tot_Base.op_At [hd; nblk; nthr] args'))) in
                cb e')
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _pre::_post::{
                           FStar_Extraction_ML_Syntax.expr =
                             FStar_Extraction_ML_Syntax.MLE_Fun
                             (uu___3, body);
                           FStar_Extraction_ML_Syntax.mlty = uu___4;
                           FStar_Extraction_ML_Syntax.loc = uu___5;_}::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Kernel.launch_kernel_1" ->
           let uu___6 = head_and_args body in
           (match uu___6 with
            | (hd, args) ->
                let kcall =
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty
                    (FStar_Extraction_ML_Syntax.MLE_Name ([], "PULSE_KCALL")) in
                let args' =
                  FStar_Compiler_List.filter
                    (fun a ->
                       match a.FStar_Extraction_ML_Syntax.expr with
                       | FStar_Extraction_ML_Syntax.MLE_Const
                           (FStar_Extraction_ML_Syntax.MLC_Unit) -> false
                       | uu___7 -> true) args in
                let e' =
                  let uu___7 =
                    let uu___8 =
                      let uu___9 =
                        let uu___10 =
                          let uu___11 =
                            let uu___12 =
                              FStar_Extraction_ML_Syntax.with_ty
                                FStar_Extraction_ML_Syntax.ml_int_ty
                                (FStar_Extraction_ML_Syntax.MLE_Const
                                   (FStar_Extraction_ML_Syntax.MLC_Int
                                      ("1",
                                        (FStar_Pervasives_Native.Some
                                           (FStar_Const.Unsigned,
                                             FStar_Const.Int32))))) in
                            let uu___13 =
                              let uu___14 =
                                FStar_Extraction_ML_Syntax.with_ty
                                  FStar_Extraction_ML_Syntax.ml_int_ty
                                  (FStar_Extraction_ML_Syntax.MLE_Const
                                     (FStar_Extraction_ML_Syntax.MLC_Int
                                        ("1",
                                          (FStar_Pervasives_Native.Some
                                             (FStar_Const.Unsigned,
                                               FStar_Const.Int32))))) in
                              [uu___14] in
                            uu___12 :: uu___13 in
                          hd :: uu___11 in
                        FStar_List_Tot_Base.op_At uu___10 args' in
                      (kcall, uu___9) in
                    FStar_Extraction_ML_Syntax.MLE_App uu___8 in
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty uu___7 in
                cb e')
       | FStar_Extraction_ML_Syntax.MLE_App
           ({
              FStar_Extraction_ML_Syntax.expr =
                FStar_Extraction_ML_Syntax.MLE_Name p;
              FStar_Extraction_ML_Syntax.mlty = uu___1;
              FStar_Extraction_ML_Syntax.loc = uu___2;_},
            _uid::nthr::_pre::_post::{
                                       FStar_Extraction_ML_Syntax.expr =
                                         FStar_Extraction_ML_Syntax.MLE_Fun
                                         (uu___3, body);
                                       FStar_Extraction_ML_Syntax.mlty =
                                         uu___4;
                                       FStar_Extraction_ML_Syntax.loc =
                                         uu___5;_}::[])
           when
           let uu___6 = FStar_Extraction_ML_Syntax.string_of_mlpath p in
           uu___6 = "GPU.Kernel.launch_kernel_n" ->
           let uu___6 = head_and_args body in
           (match uu___6 with
            | (hd, args) ->
                let args' =
                  FStar_Compiler_List.filter
                    (fun a ->
                       match a.FStar_Extraction_ML_Syntax.expr with
                       | FStar_Extraction_ML_Syntax.MLE_Const
                           (FStar_Extraction_ML_Syntax.MLC_Unit) -> false
                       | uu___7 -> true) args in
                let kcall =
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty
                    (FStar_Extraction_ML_Syntax.MLE_Name ([], "PULSE_KCALL")) in
                let e' =
                  let uu___7 =
                    let uu___8 =
                      let uu___9 =
                        let uu___10 =
                          let uu___11 =
                            let uu___12 =
                              let uu___13 =
                                FStar_Extraction_ML_Syntax.with_ty
                                  FStar_Extraction_ML_Syntax.ml_int_ty
                                  (FStar_Extraction_ML_Syntax.MLE_Const
                                     (FStar_Extraction_ML_Syntax.MLC_Int
                                        ("1",
                                          (FStar_Pervasives_Native.Some
                                             (FStar_Const.Unsigned,
                                               FStar_Const.Int32))))) in
                              [uu___13] in
                            nthr :: uu___12 in
                          hd :: uu___11 in
                        FStar_List_Tot_Base.op_At uu___10 args' in
                      (kcall, uu___9) in
                    FStar_Extraction_ML_Syntax.MLE_App uu___8 in
                  FStar_Extraction_ML_Syntax.with_ty
                    FStar_Extraction_ML_Syntax.ml_unit_ty uu___7 in
                cb e')
       | uu___1 ->
           FStar_Compiler_Effect.raise
             FStar_Extraction_Krml.NotSupportedByKrmlExtension)
let (uu___449 : unit) =
  FStar_Extraction_Krml.register_pre_translate_type_without_decay
    gpu_translate_type_without_decay;
  FStar_Extraction_Krml.register_pre_translate_expr gpu_translate_expr