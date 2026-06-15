module Kuiper.Index

open Kuiper.Bijection
open Kuiper.Common
open Kuiper.SizeT
module SZ = Kuiper.SizeT

(* An idesc represents a tensor type, where every ICons adds a dimension.
   MxN matrix = ICons m (ICons n INil) *)
[@@erasable]
noeq
type idesc : nat -> Type =
  | INil : idesc 0
  | ICons : #n:nat -> w:nat -> tl:(idesc n) -> idesc (n+1)

unfold let ( @| ) (#n:nat) = ICons #n

let head (#n:pos) (d : idesc n) : GTot nat =
  match d with
  | ICons hd _ -> hd

let tail (#n:pos) (d : idesc n) : idesc (n-1) =
  match d with
  | ICons _ tl -> tl

[@@strict_on_arguments [1]]
let rec ( @! ) (#n:nat) (d : idesc n) (i : natlt n) : GTot nat =
  match d with
  | ICons t ts ->
    match i with
    | 0 -> t
    | i -> ts @! (i - 1)

#push-options "--warn_error -271"
let rec lemma_at_tail (#n:nat) (d : idesc n) (i : natlt (n-1))
  : Lemma (d @! (i+1) == tail d @! i)
  = match d with
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i -> lemma_at_tail ts (i - 1)
#pop-options

[@@strict_on_arguments [1]]
let rec sizeof (#r : nat) (d : idesc r) : GTot nat =
  match d with
  | INil -> 1
  | ICons t ts -> t * sizeof ts

(* Abstract index type for a tensor *)
// [@@strict_on_arguments [1]]
inline_for_extraction noextract
let rec abs #n (i : idesc n) : eqtype =
  match i with
  | INil -> unit
  | ICons h ts -> natlt h & abs ts

(* Concrete index type for a tensor. This could also be eqtype, but I don't
think that is needed and would be bad at runtime. *)
// [@@strict_on_arguments [1]]
inline_for_extraction noextract
let rec conc #n (i : idesc n) : Type0 =
  match i with
  | INil -> unit
  | ICons h ts -> szlt h & conc ts

[@@strict_on_arguments [1]; Pulse.pulse_unfold]
let rec up #n (#d : idesc n) (v : conc d) : GTot (abs d) =
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: szlt t & conc ts in
    ((SZ.v i1 <: natlt t), up is)

let rec all_fit (#n:nat) (d : idesc n) : prop =
  match d with
  | INil -> True
  | ICons t ts -> SZ.fits t /\ all_fit ts

let all_fit' (#n:nat) (d : idesc n) : prop =
  forall (i : natlt n). SZ.fits (d @! i)

let rec all_fit_iff_all_fit' (#n:nat) (d : idesc n) : Lemma (all_fit d <==> all_fit' d)
        [SMTPat (all_fit d)]
  = match d with
    | INil -> ()
    | ICons t ts ->
      calc (<==>) {
        all_fit (ICons t ts);
        <==> {}
        SZ.fits t /\ all_fit ts;
        <==> { all_fit_iff_all_fit' ts }
        SZ.fits t /\ (forall (i : natlt (n-1)). SZ.fits (ts @! i));
        <==> {}
        SZ.fits (d @! 0) /\ (forall (i : natlt (n-1)). SZ.fits (tail d @! i));
        <==> { Classical.forall_intro (lemma_at_tail d) }
        (forall (i : natlt n). SZ.fits (ICons t ts @! i));
      };
      ()

[@@strict_on_arguments [1]]
let rec down #n (#d : idesc n{all_fit d}) (v : abs d) : GTot (conc d) =
  match d with
  | INil -> ()
  | ICons t ts ->
    let i1, is = v <: natlt t & abs ts in
    ((SZ.uint_to_t i1 <: szlt t), down is)

// The refinement on v (which talks about d....) seems to be better than having
// a precondition on the lemma, otherwise the trigger does not seem to work.  A
// refinement on d itself, or squash arguments, also fail to trigger.
val up_down #n (#d : idesc n) (v : abs d) :
  Lemma (ensures all_fit d ==> up (down v) == v)
        [SMTPat (up (down v))]

val down_up #n (#d : idesc n) (v : conc d) :
  Lemma (ensures all_fit d ==> down (up v) == v)
        [SMTPat (down (up v))]

(* Remove (fix) a given dimension *)
[@@strict_on_arguments [2]]
let rec modulo_i (#n:nat) (i : natlt n) (d : idesc n) : idesc (n-1) =
  (* Cannot match on d and i simultaneously *)
  match d with
  | ICons t ts ->
    match i with
    | 0 -> ts
    | i -> ICons t (modulo_i (i-1) ts)

#push-options "--warn_error -271"
let rec all_fit_modulo (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (requires all_fit d)
          (ensures  all_fit (modulo_i i d))
          [SMTPat (all_fit d); SMTPat (all_fit (modulo_i i d))]
  = match d with
    | INil -> ()
    | ICons t ts ->
      match i with
      | 0 -> ()
      | i -> all_fit_modulo (i-1) ts
#pop-options

(* Insert a dimension. Note the n+1, one can insert at the very end. *)
[@@strict_on_arguments [3]]
let rec insert_i (#n:nat) (i : natlt (n+1)) (k : nat) (d : idesc n) : idesc (n+1) =
  match i with
  | 0 -> ICons k d
  | i -> ICons (d @! 0) (insert_i (i-1) k (modulo_i 0 d))

(* Silence warning about using '-' in patterns. It's in an implicit,
not much to do, and it works. *)
#push-options "--warn_error -271"
val insert_modulo (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (insert_i #(n-1) i (d @! i) (modulo_i i d) == d)
          [SMTPat (insert_i #(n-1) i (d @! i) (modulo_i i d))]

val modulo_insert (#n:nat) (i : natlt (n+1)) (k : nat) (d : idesc n)
  : Lemma (ensures modulo_i i (insert_i i k d) == d)
          [SMTPat (modulo_i i (insert_i i k d))]

val modulo_size_lemma (#n:nat) (i : natlt n) (d : idesc n)
  : Lemma (sizeof (modulo_i i d) * (d @! i) == sizeof d)
          [SMTPat (sizeof (modulo_i i d)); SMTPat (sizeof d)]

val insert_size_lemma (#n:nat) (i : natlt (n+1)) (k : nat) (d : idesc n)
  : Lemma (sizeof (insert_i i k d) == sizeof d * k)
          [SMTPat (sizeof (insert_i i k d)); SMTPat (sizeof d)]

val insert_at_lemma (#n:nat) (i : natlt (n+1)) (k : nat) (d : idesc n)
  : Lemma (insert_i i k d @! i == k)
          [SMTPat (insert_i i k d @! i)]
#pop-options

let rec abs_bring_forward_bij (#n:nat) (i : natlt n) (d : idesc n)
  : (abs d =~ natlt (d @! i) & abs (modulo_i i d))
  = match i with
    | 0 -> bij_self _
    | _ ->
      bij_prod (bij_self _) (abs_bring_forward_bij (i-1) (tail d))
      `bij_comp`
      bij_push_tuple3 #(natlt (d @! 0))

let rec conc_bring_forward_bij (#n:nat) (i : natlt n) (d : idesc n)
  : (conc d =~ szlt (d @! i) & conc (modulo_i i d))
  = match i with
    | 0 -> bij_self _
    | _ ->
      bij_prod (bij_self _) (conc_bring_forward_bij (i-1) (ICons?.tl d))
      `bij_comp`
      bij_push_tuple3 #(szlt (d @! 0))

(* A computationally relevant version of the above, for use in cimap. *)
[@@strict_on_arguments [2]]
inline_for_extraction noextract
let rec c_conc_bring_forward_bij (#n : Ghost.erased nat) (i : szlt n) (d : idesc n)
  : cb : (conc d ==~ szlt (d @! i) & conc (modulo_i i d)) { cb.bij == conc_bring_forward_bij i d }
  = if i = 0sz then
      cbij_self _
    else
      cbij_prod (cbij_self _) (c_conc_bring_forward_bij (i-^1sz) (ICons?.tl d))
      `cbij_comp`
      cbij_push_tuple3 #(szlt (d @! 0))

(* A computationally relevant version of the above, for use in cimap. *)
[@@strict_on_arguments [1]]
inline_for_extraction noextract
let rec c_bring_forward_ff (#n : Ghost.erased nat) (i : natlt n) (d : idesc n)
  (idx : conc d) : szlt (d @! i) & conc (modulo_i i d)
  = if i = 0 then
      idx
    else
      let h, t = idx <: szlt (d @! 0) & conc (tail d) in
      let x, t' = c_bring_forward_ff (i-1) (tail d) t in
      x, (h, t')

(* Idem, for gg *)
[@@strict_on_arguments [1]]
inline_for_extraction noextract
let rec c_bring_forward_gg (#n : Ghost.erased nat) (i : natlt n) (d : idesc n)
  (h:  szlt (d @! i))
  (t : conc (modulo_i i d))
  : Tot (conc d)
  = if i = 0 then
      (h, t)
    else
      let unfold h2, t2 = t <: szlt (d @! 0) & conc (tail (modulo_i i d)) in
      // [@@inline_let] let t' = c_bring_forward_gg #(n-1) (i-^1sz) (tail d) h t2 in
      // (h2, t')
      (h2, c_bring_forward_gg #(n-1) (i-1) (tail d) h t2)

val lemma_c_bring_forward_ff_ok
  (#n : Ghost.erased nat) (i : szlt n) (d : idesc n)
  (idx : conc d)
  : Lemma (c_bring_forward_ff #n i d idx == (c_conc_bring_forward_bij #n i d).cff idx)
          [SMTPat (c_bring_forward_ff #n i d idx)]

val lemma_c_bring_forward_gg_ok
  (#n : Ghost.erased nat) (i : szlt n) (d : idesc n)
  (h:  szlt (d @! i)) (t : conc (modulo_i i d))
  : Lemma (c_bring_forward_gg #n i d h t == (c_conc_bring_forward_bij #n i d).cgg (h, t))
          [SMTPat (c_bring_forward_gg #n i d h t)]

(*
   abs d --abs_bring_forward--> natlt (d @! i) & abs (modulo_i i d)
    |                                         |
    |                                         |
   down                                  down x down
    |                                         |
    v                                         v
   conc d --conc_bring_forward--> szlt (d @! i) & conc (modulo_i i d)
*)
let down2 (#n:nat) (i : natlt n) (d : idesc n{all_fit d})
  (tup : natlt (d @! i) & abs (modulo_i i d))
  : GTot (szlt (d @! i) & conc (modulo_i i d))
  = match tup with
    | (j, abs_mod) ->
      ((SZ.uint_to_t j <: szlt (d @! i)), down abs_mod)

val bring_forward_commute (#n:nat) (i : natlt n) (d : idesc n{all_fit d})
  (idx : abs d)
  : Lemma (down2 i d ((abs_bring_forward_bij i d).ff idx) ==
          (conc_bring_forward_bij i d).ff (down idx))

val bring_forward_commute2 (#n:nat) (i : natlt n) (d : idesc n)
  (j : szlt (d @! i)) (idx : conc (modulo_i i d))
  : Lemma (up ((conc_bring_forward_bij i d).gg (j, idx))
           == (abs_bring_forward_bij i d).gg (SZ.v j, up idx))

// This bijection only exists when all_fit holds.
let abs_conc_bij (#n:nat) (d : idesc n{all_fit d})
  : (abs d =~ conc d)
  = {
    ff = down;
    gg = up;
    ff_gg = (fun (v : conc d) -> down_up v);
    gg_ff = (fun (v : abs d) -> up_down v);
  }

(* Raw indices. These could be used to give Tensor an API
where read/write take a raw tuple instead of a conc
(a tuple of refined types) which could be nicer for inference
given the lack of subtyping on tuples. *)

(* Raw index type for a tensor, without any refinements. This
is the type we use for read/write operations to prevents
tuples of the wrong type. *)
inline_for_extraction noextract
let rec raw #n (i : idesc n) : Type0 =
  match i with
  | INil -> unit
  | ICons h ts -> sz & raw ts

let rec raw_fits #n (d : idesc n) (idx : raw d) : prop =
  match d with
  | INil -> True
  | ICons t ts ->
    let i, is = idx <: sz & raw ts in
    i < t /\ raw_fits ts is

let fold_outer (#n: nat {n > 1}) (i : idesc n) : idesc (n-1) = 
  let ICons h1 (ICons h2 ts) = i in
  (h1 * h2) @| ts