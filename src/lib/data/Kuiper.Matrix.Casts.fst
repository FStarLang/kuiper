module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Bijection

ghost
fn tensor_abij
  (#et : Type0)
  (#r1 : nat) (#s1 : shape r1)
  (#r2 : nat) (#s2 : shape r2)
  (b : abs s1 =~ abs s2)
  (#l : tlayout s1)
  (a : tensor et l)
  (#s : chest s1 et)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (layout_bij b l) |-> Frac f (chest_bij b s)
{
  let l' = layout_bij b l;
  let a' = from_array l' (core a);
    assert rewrites_to a' (from_array (layout_bij b l) (core a));

  tensor_ilower a;
  forevery_iso b _;
  forevery_map
    #(abs s2)
    (fun i ->
      pts_to_cell (core a) #f
        (l.imap.f (b.gg i))
        (acc s (b.gg i))
    )
    (fun i ->
      pts_to_cell (core a') #f
        (l'.imap.f i)
        (acc (chest_bij b s) i)
    )
    fn i {
      rewrite
        pts_to_cell (core a) #f
          (l.imap.f (b.gg i))
          (acc s (b.gg i))
      as
        pts_to_cell (core a') #f
          (l'.imap.f i)
          (acc (chest_bij b s) i);
      ()
    };
  tensor_iraise a';
}

ghost
fn t1_to_t2
  (#et : Type0)
  (#len : nat)
  (#l : layout1 len)
  (a : array1 et l)
  (#s : chest1 et len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (l1_to_l2 l) |-> Frac f (c1_to_c2 s)
{
  tensor_abij (bij12 len) a;
  rewrite each chest_bij (bij12 len) s as c1_to_c2 s;
}

ghost
fn t2_to_t1
  (#et : Type0)
  (#len : nat)
  (#l : layout2 1 len)
  (a : array2 et l)
  (#s : chest2 et 1 len)
  (#f : perm)
  requires
    a |-> Frac f s
  ensures
    relay a (l2_to_l1 l) |-> Frac f (c2_to_c1 s)
{
  tensor_abij (bij_sym (bij12 len)) a;
  rewrite each chest_bij (bij_sym (bij12 len)) s as c2_to_c1 s;
}

let c2_to_c3_roundtrip
  (#et : Type0)
  (d0 d1 : szp)
  (s : chest2 et d0 d1)
  : Lemma (c3_to_c2 d0 d1 (c2_to_c3 d0 d1 s) == s)
  =
  Kuiper.Chest.lemma_equal_intro
    (c3_to_c2 d0 d1 (c2_to_c3 d0 d1 s)) s;
  Kuiper.Chest.ext (c3_to_c2 d0 d1 (c2_to_c3 d0 d1 s)) s

let c2_to_c3_slice_page
  (#et : Type0)
  (d0 d1 : szp)
  (s : chest2 et d0 d1)
  : Lemma (slice_page (c2_to_c3 d0 d1 s) 0 == s)
  =
  Kuiper.Chest.lemma_equal_intro (slice_page (c2_to_c3 d0 d1 s) 0) s;
  Kuiper.Chest.ext (slice_page (c2_to_c3 d0 d1 s) 0) s

ghost
fn t2_to_t3
  (#et : Type0)
  (d0 d1 : szp)
  (#l : layout2 d0 d1)
  {| ctlayout l |}
  (g : tensor et l)
  (#f : perm)
  (#s : chest2 et d0 d1)
  requires
    g |-> Frac f s
  ensures
    from_array (l2_to_l3 d0 d1 #l) (core g)
      |-> Frac f (c2_to_c3 d0 d1 s)
{
  tensor_abij (bij_up (cbij23 d0 d1)) g;
}

ghost
fn t3_to_t2
  (#et : Type0)
  (d0 d1 : szp)
  (#l : layout2 d0 d1)
  {| ctlayout l |}
  (g : tensor et l)
  (#f : perm)
  (#s3 : chest3 et 1 d0 d1)
  requires
    from_array (l2_to_l3 d0 d1 #l) (core g) |-> Frac f s3
  ensures
    g |-> Frac f (c3_to_c2 d0 d1 s3)
{
  let g3 = from_array (l2_to_l3 d0 d1 #l) (core g);
  rewrite each
    from_array (l2_to_l3 d0 d1 #l) (core g)
  as g3;
  tensor_ilower g3;
  let g2 = from_array l (core g3);
  rewrite each core g3 as core g2;

  let bij = bij_up (cbij23 d0 d1);
  forevery_iso (bij_sym bij)
    (fun (idx3 : abs (1 @| d0 @| d1 @| INil)) ->
      Kuiper.Array.pts_to_cell (core g2) #f
        ((l2_to_l3 d0 d1 #l).imap.f idx3)
        (acc s3 idx3));
  forevery_ext
    (fun (idx2 : abs (d0 @| d1 @| INil)) ->
      Kuiper.Array.pts_to_cell (core g2) #f
        ((l2_to_l3 d0 d1 #l).imap.f (bij.ff idx2))
        (acc s3 (bij.ff idx2)))
    (fun (idx2 : abs (d0 @| d1 @| INil)) ->
      Kuiper.Array.pts_to_cell (core g2) #f
        (l.imap.f idx2)
        (acc (c3_to_c2 d0 d1 s3) idx2));
  tensor_iraise g2;
  rewrite
    (g2 |-> Frac f (c3_to_c2 d0 d1 s3))
  as
    (g |-> Frac f (c3_to_c2 d0 d1 s3));
}
