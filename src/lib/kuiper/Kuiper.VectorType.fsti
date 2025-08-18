module Kuiper.VectorType

open Kuiper

new
val float4 : Type0

val make_float4 (x y z w : float) : float4

val getx (v : float4) : float
val gety (v : float4) : float
val getz (v : float4) : float
val getw (v : float4) : float

val lemma_getx_projects_x (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures getx (make_float4 x y z w) == x)
  [SMTPat (getx (make_float4 x y z w))]

val lemma_gety_projects_y (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures gety (make_float4 x y z w) == y)
  [SMTPat (gety (make_float4 x y z w))]

val lemma_getz_projects_z (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures getz (make_float4 x y z w) == z)
  [SMTPat (getz (make_float4 x y z w))]

val lemma_getw_projects_w (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures getw (make_float4 x y z w) == w)
  [SMTPat (getw (make_float4 x y z w))]

val lemma_make_float4_from_float4 (v : float4)
  : Lemma (ensures make_float4 (getx v) (gety v) (getz v) (getw v) == v)
