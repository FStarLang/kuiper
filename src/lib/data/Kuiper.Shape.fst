module Kuiper.Shape

open Kuiper.Bijection
open Kuiper.Common
open Kuiper.SizeT

let rec up_down #n (#d : shape n) (v : abs d) :
  Lemma (ensures all_fit d ==> up (down v) == v)
        [SMTPat (up (down v))]
=
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: natlt t & abs ts in
    up_down is

let rec down_up #n (#d : shape n) (v : conc d) :
  Lemma (ensures all_fit d ==> down (up v) == v)
        [SMTPat (down (up v))]
=
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: szlt t & conc ts in
    down_up is

#push-options "--warn_error -271"
let rec insert_modulo (#n:nat) (i : natlt n) (d : shape n)
  : Lemma (insert_i #(n-1) i (d @! i) (modulo_i i d) == d)
          [SMTPat (insert_i #(n-1) i (d @! i) (modulo_i i d))]
  = match d with
    | INil -> ()
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i -> insert_modulo (i-1) ts

let rec modulo_insert (#n:nat) (i : natlt (n+1)) (k : nat) (d : shape n)
  : Lemma (ensures modulo_i i (insert_i i k d) == d)
          [SMTPat (modulo_i i (insert_i i k d))]
  = match i with
    | 0 -> ()
    | i ->
      match d with
      | INil -> assert false
      | ICons t ts -> modulo_insert (i-1) k ts

let rec modulo_size_lemma (#n:nat) (i : natlt n) (d : shape n)
  : Lemma (sizeof (modulo_i i d) * (d @! i) == sizeof d)
          [SMTPat (sizeof (modulo_i i d)); SMTPat (sizeof d)]
  = match d with
    | INil -> ()
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i -> modulo_size_lemma (i-1) ts

let rec insert_size_lemma (#n:nat) (i : natlt (n+1)) (k : nat) (d : shape n)
  : Lemma (sizeof (insert_i i k d) == sizeof d * k)
          [SMTPat (sizeof (insert_i i k d)); SMTPat (sizeof d)]
  = match i with
    | 0 -> ()
    | i ->
      match d with
      | INil -> assert false
      | ICons t ts -> insert_size_lemma (i-1) k ts
let rec insert_at_lemma (#n:nat) (i : natlt (n+1)) (k : nat) (d : shape n)
  : Lemma (insert_i i k d @! i == k)
          [SMTPat (insert_i i k d @! i)]
  = match i with
    | 0 -> ()
    | i ->
      match d with
      | INil -> assert false
      | ICons t ts -> insert_at_lemma (i-1) k ts
#pop-options

let rec lemma_c_bring_forward_ff_ok
  (#n : Ghost.erased nat) (i : szlt n) (d : shape n)
  (idx : conc d)
  : Lemma (c_bring_forward_ff #n i d idx == (c_conc_bring_forward_bij #n i d).cff idx)
          [SMTPat (c_bring_forward_ff #n i d idx)]
  = if i = 0sz then
      ()
    else
      let dh : Ghost.erased nat = head d in
      let dt = tail d in
      let h, t = idx <: szlt dh & conc dt in
      lemma_c_bring_forward_ff_ok (i-^1sz) (tail d) t

let rec lemma_c_bring_forward_gg_ok
  (#n : Ghost.erased nat) (i : szlt n) (d : shape n)
  (h:  szlt (d @! i)) (t : conc (modulo_i i d))
  : Lemma (c_bring_forward_gg #n i d h t == (c_conc_bring_forward_bij #n i d).cgg (h, t))
          [SMTPat (c_bring_forward_gg #n i d h t)]
  = if i = 0sz then
      ()
    else
      let dh : Ghost.erased nat = head d in
      let dt = tail d in
      let hh, tt = t <: szlt (d @! 0) & conc (modulo_i (i-^1sz) dt) in
      lemma_c_bring_forward_gg_ok (i-^1sz) (tail d) h tt

let rec bring_forward_commute (#n:nat) (i : natlt n) (d : shape n{all_fit d})
  (idx : abs d)
  : Lemma (down2 i d ((abs_bring_forward_bij i d).ff idx) ==
          (conc_bring_forward_bij i d).ff (down idx))
  = match d with
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i ->
        let idx1, idx_mod = idx <: natlt (d @! 0) & abs ts in
        bring_forward_commute (i-1) ts idx_mod

let rec bring_forward_commute2 (#n:nat) (i : natlt n) (d : shape n)
  (j : szlt (d @! i)) (idx : conc (modulo_i i d))
  : Lemma (up ((conc_bring_forward_bij i d).gg (j, idx))
           == (abs_bring_forward_bij i d).gg (SizeT.v j, up idx))
  = match d with
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i ->
        let hh, tt = idx <: szlt (d @! 0) & conc (modulo_i (i-1) ts) in
        bring_forward_commute2 (i-1) ts j tt

[@@strict_on_arguments [2]]
inline_for_extraction noextract
let rec cunflatten
  (#r : erased nat)
  (#d : shape r)
  (cd : cshape d)
  (x : szlt (sizeof d))
  : Pure (conc d)
         (requires SZ.fits (sizeof d))
         (ensures fun r -> up r == unflatten d (SZ.v x))
  = match cd with
    | CNil -> ()
    | CCons #_ #h ch #t ct ->
      let major : szlt h          = x /^ csizeof ct in
      let minor : szlt (sizeof t) = x %^ csizeof ct in
      (major, cunflatten ct minor)

[@@strict_on_arguments [2]]
inline_for_extraction noextract
let rec cflatten
  (#r : erased nat)
  (#d : shape r)
  (cd : cshape d)
  (x : conc d)
  : Pure (szlt (sizeof d))
         (requires SZ.fits (sizeof d))
         (ensures fun r -> SZ.v r == flatten d (up x))
  = match cd with
    | CNil -> 0sz
    | CCons #_ #h ch #t ct ->
      let (i1, i2) = x <: szlt h & conc t in
      i1 *^ csizeof ct +^ cflatten ct i2
