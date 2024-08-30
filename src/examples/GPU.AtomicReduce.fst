module GPU.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open GPU
open GPU.AtomicReduce.Kernel

module SZ = FStar.SizeT

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
    i_done : erased (iname & erased (lseq (gref bool) (SZ.v n)))
    // i : iname
    // done : lseq (gref bool) (SZ.v n)
  ensures
    emp
    ** cpu
    ** bigstar 0 (SZ.v n) (fun tid ->
        inv (reveal i_done)._1 (inv_p (SZ.v n) a v_a r (reveal i_done)._2) **
        gref_pts_to ((reveal i_done)._2 @! tid) #0.5R false)
    ** pure (Seq.length (reveal i_done)._2 == SZ.v n)
{
  admit();
}

#push-options "--debug SMTQuery,SMTFail --split_queries always"

fn reduce
  (#nn: erased nat)
  (n : sz)
  (a : gpu_array u64 n)
  (#f : perm)
  (#v_a : erased (seq u64))
  requires
    cpu **
    gpu_pts_to_array a #f v_a **
    pure (SZ.v n <= 1024 /\ nn == SZ.v n)
  returns
    r : u64
  ensures 
    cpu **
    gpu_pts_to_array a #f v_a **
    pure (r == GPU.Seq.Common.seq_fold_left (fun x y -> UInt64.add_mod x y) 0uL v_a)
{
  let r = alloc 0uL;
  let gr = gpu_alloc0 #u64 ();
  Ref.gpu_memcpy_host_to_device r gr #_ #_ #_;

  with v. assert (pts_to r v);
  assert (pure (v == 0uL));

  assume_ (pure (n > 0));
  assert (pure (n < max_blocks));

  // assert (gpu_pts_to gr #1.0R 0uL);
  let i_done = setup n a gr;
  let i = (reveal i_done)._1;
  let done = (reveal i_done)._2;
  // let i, done = setup n a #f #v_a gr;


  launch_kernel_n #0 n
    #(kpre  nn a v_a gr done i)
    #(kpost nn a v_a gr done i)
    (
      kernel (hide n) a gr done i v_a );

}


(*
1. Let-binding a tuple, or any pattern really
2. Easily returning multiple things, writing specs over the exploded components
3-  refinements have to match exactly
4- err locations
4.1-  in particular when function type does not match fsti
5- jonas' coerce_eq bug
*)
