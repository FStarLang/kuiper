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

ghost
fn block_setup ()
  norewrite
  requires can_create_barrier 0 ** emp
  ensures  consumed_can_create_barrier ** (forall+ (tid : natlt 0). emp) ** emp
{
  no_mk_barrier ();
  forevery_emp_intro (natlt 0);
}

ghost
fn block_teardown ()
  norewrite
  requires (forall+ (tid : natlt 0). emp) ** emp
  ensures  emp
{
  forevery_emp_elim (natlt 0);
}

inline_for_extraction noextract
let kdesc :
  kernel_desc_1_n
    emp emp
= {
  nthr = 0sz;

  kpre  = (fun _ -> emp);
  kpost = (fun _ -> emp);

  f = (fun _ -> kf);

  frame = emp;
  block_setup    = block_setup;
  block_teardown = block_teardown;
  kpost_sendable = solve;
  kpre_sendable = solve;
  full_post_sendable = solve;
  full_pre_sendable = solve;
}

fn test ()
  preserves cpu
{
  launch_sync kdesc;
}
