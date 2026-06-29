module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Bijection

ghost
fn t1_to_t2
  (#et : Type0)
  (len : nat)
  (#l : layout1 len)
  (a : array1 et l)
  (#s : chest1 et len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    from_array (l1_to_l2 l) (core a) |-> Frac f (c1_to_c2 s)
{
  let l' = l1_to_l2 l;
  let a' = from_array l' (core a);
    assert rewrites_to a' (from_array (l1_to_l2 l) (core a));

  tensor_ilower a;
  forevery_iso (bij12 len) _;
  forevery_map
    #(abs (1 @| len @| INil))
    (fun i ->
      pts_to_cell (core a) #f
        (l.imap.f ((bij12 len).gg i))
        (acc s ((bij12 len).gg i))
    )
    (fun i ->
      pts_to_cell (core a') #f
        (l'.imap.f i)
        (acc (c1_to_c2 s) i)
    )
    fn i {
      rewrite
        pts_to_cell (core a) #f
          (l.imap.f ((bij12 len).gg i))
          (acc s ((bij12 len).gg i))
      as
        pts_to_cell (core a') #f
          (l'.imap.f i)
          (acc (c1_to_c2 s) i);
      ()
    };
  tensor_iraise a';
}

ghost
fn t2_to_t1
  (#et : Type0)
  (len : nat)
  (#l : layout2 1 len)
  (a : array2 et l)
  (#s : chest2 et 1 len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    from_array (l2_to_l1 l) (core a) |-> Frac f (c2_to_c1 s)
{
  let l' = l2_to_l1 l;
  let a' = from_array l' (core a);
    assert rewrites_to a' (from_array (l2_to_l1 l) (core a));

  tensor_ilower a;
  forevery_iso (bij_sym (bij12 len)) _;
  forevery_map
    #(abs (len @| INil))
    (fun i ->
      pts_to_cell (core a) #f
        (l.imap.f ((bij_sym (bij12 len)).gg i))
        (acc s ((bij_sym (bij12 len)).gg i))
    )
    (fun i ->
      pts_to_cell (core a') #f
        (l'.imap.f i)
        (acc (c2_to_c1 s) i)
    )
    fn i {
      rewrite
        pts_to_cell (core a) #f
          (l.imap.f ((bij_sym (bij12 len)).gg i))
          (acc s ((bij_sym (bij12 len)).gg i))
      as
        pts_to_cell (core a') #f
          (l'.imap.f i)
          (acc (c2_to_c1 s) i);
      ()
    };
  tensor_iraise a';
}
