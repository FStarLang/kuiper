module Kuiper.ZeroKernel

#lang-pulse
open Kuiper

(* This is just testing that this file does not crash. We skip
launching the kernel when nblk or nthr are zero, because that's
a CUDA error. *)

inline_for_extraction noextract
fn kf
  ()
  requires emp
  ensures emp
{ () }

inline_for_extraction noextract
let kdesc :
  kernel_desc
    emp emp
= {
  nblk = 0sz;
  nthr = 0sz;

  shmems_desc = [];

  kpre  = (fun _ _ _ -> emp);
  kpost = (fun _ _ _ -> emp);

  f = (fun _ _ _ -> kf);

  frame = emp;
  setup    = magic();
  teardown = magic();

  block_pre  = (fun _ -> emp);
  block_post = (fun _ -> emp);

  block_setup    = magic();
  block_teardown = magic();

  block_frame = (fun _ _ -> emp);
}

fn test ()
  preserves cpu
{
  launch_sync kdesc;
}
