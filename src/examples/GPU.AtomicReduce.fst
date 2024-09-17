module GPU.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open GPU
open GPU.AtomicReduce.Kernel

module SZ = FStar.SizeT
module W = Pulse.Lib.WithPure

ghost
fn setup
  (n : sz)
  (a : gpu_array u64 n)
  (#f : perm)
  (#v_a : erased (seq u64))
  (r : gpu_ref u64)
  requires
    cpu **
    gpu_pts_to_array a #f v_a **
    gpu_pts_to r #1.0R 0uL **
    pure (SZ.v n <= 1024)
  returns 
    i_done : iname & erased (seq (gref bool))
  ensures
    (match i_done with | (i, done) ->
    cpu
    ** W.with_pure (Seq.length done == SZ.v n) (fun _ ->
       bigstar 0 (SZ.v n) (fun tid ->
        gref_pts_to (done @! tid) #0.5R false **
        inv i (inv_p (SZ.v n) a v_a r done))
    ))
{
  admit();
}

ghost
fn teardown
  (n : sz)
  (a : gpu_array u64 n)
  (#f : perm)
  (#v_a : erased (seq u64))
  (r : gpu_ref u64)
  (i : iname)
  (done : lseq (gref bool) (SZ.v n))
  // returns
  //   i_done : erased (iname & erased (seq (gref bool)))
  requires
    emp
    ** cpu
    ** pure (Seq.length done == SZ.v n) 
    ** bigstar 0 (SZ.v n) (fun tid ->
        gref_pts_to (done @! tid) #0.5R true **
        inv i (inv_p (SZ.v n) a v_a r done))
  ensures
    cpu **
    gpu_pts_to_array a #f v_a **
    gpu_pts_to r #1.0R (GPU.Seq.Common.seq_fold_left (fun x y -> UInt64.add_mod x y) 0uL v_a) **
    pure (SZ.v n <= 1024)
{
  admit();
}

// #push-options "--debug SMTFail --split_queries always" // --print_implicits"

// #set-options "--ext pulse:trace=1"
//  --debug SMTFail --split_queries always"

fn reduce
  (n : sz)
  (a : gpu_array u64 n)
  (#f : perm)
  (#v_a : erased (seq u64))
  requires
    cpu **
    pure (f == 1.0R) **
    gpu_pts_to_array a #f v_a **
    pure (SZ.v n > 0 /\ SZ.v n <= 1024)
  returns
    r : u64
  ensures 
    cpu **
    gpu_pts_to_array a #f v_a **
    pure (r == GPU.Seq.Common.seq_fold_left (fun x y -> UInt64.add_mod x y) 0uL v_a)
{
  let r = alloc 0uL;
  let gr = gpu_alloc0 #u64 ();
  Ref.gpu_memcpy_host_to_device gr r;

  with v. assert (pts_to r v);
  assert (pure (v == 0uL));

  assert (pure (n < max_blocks));

  // assert (gpu_pts_to gr #1.0R 0uL);
  
  
  // pack (x,y) as p?
  // let p = (x, y);
  // rewrite each x as p._1;
  // rewrite each y as p._2;
  // pack Inl x as o;



  let i_done = setup n a gr;
  let i = (i_done)._1;
  let done : erased (seq (gref bool)) = hide (reveal (i_done._2));
  rewrite each i_done as (i, done) by (tadmit ());
  // New fancy syntax, does not extract
  // let Mktuple2 i done = setup n a gr;

  W.elim_with_pure (Seq.length done == SZ.v n) _; 

  assert (bigstar 0 n (fun tid -> kpre  (SZ.v n) a v_a gr done i tid));

  launch_kernel_n #0 n
    #(kpre  (SZ.v n) a v_a gr done i)
    #(kpost (SZ.v n) a v_a gr done i)
    (fun etid -> kernel (hide n) a gr done i v_a etid);

  teardown n a #f #v_a gr i done;

  Ref.gpu_memcpy_device_to_host r gr #_ #_ #_;
  
  GPU.Ref.gpu_free gr;
  
  let v = !r;
  // free r;
  drop_ (pts_to r _); // cannot free, should use boxes and fix the library
  // admit();
  v
}


(*
1. Let-binding a tuple, or any pattern really
2. Easily returning multiple things, writing specs over the exploded components
3-  refinements have to match exactly
4- err locations
4.1-  in particular when function type does not match fsti
5- jonas' coerce_eq bug
*)
