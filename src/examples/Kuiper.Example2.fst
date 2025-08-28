module Kuiper.Example2

#lang-pulse

open Kuiper

(* Kernels need not have any arguments. Not that this is
very useful... *)
inline_for_extraction noextract
fn kf ()
  norewrite
  preserves gpu
  requires emp
  ensures emp
{
  ();
}

fn main (_:unit)
  preserves cpu
  requires emp
  returns  _ : u64
  ensures emp
{
  launch_kernel_1 (fun () -> kf ());
  1uL
}
