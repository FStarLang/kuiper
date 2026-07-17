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

let rec flatten_unflatten (#r : nat) (d : shape r) (x : natlt (sizeof d))
  : Lemma (ensures flatten d (unflatten d x) == x)
          (decreases d)
  = match d with
    | INil -> ()
    | ICons h t ->
      let minor : natlt (sizeof t) = x % sizeof t in
      flatten_unflatten t minor

let rec unflatten_flatten (#r : nat) (d : shape r) (x : abs d)
  : Lemma (ensures unflatten d (flatten d x) == x)
          (decreases d)
  = match d with
    | INil -> ()
    | ICons h t ->
      let (i1, i2) = x <: natlt h & abs t in
      unflatten_flatten t i2

inline_for_extraction noextract
let rec conc_set_at (#r : erased nat) (#d : shape r) (dim : szlt r) (idx : szlt (d @! dim)) (x : conc d)
  : Tot (c : conc d {up c == abs_set_at dim idx (up x)})
         (decreases (SZ.v dim))
  = assert r > 0;
    let (i1,i2) = x <: szlt (d @! 0) & conc (tail d) in
    if dim = 0sz then
      (idx, i2)
    else (
      let x1 = conc_set_at #_ #(tail d) (dim -^ 1sz) idx i2 in
      (i1, (x1 <: (conc (tail d))))
    )

let abs_le_cons (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (x: abs d1)
  : Lemma (requires ICons? d1)
          (ensures (let i1, i2 = x <: natlt (d1 @! 0) & abs (tail d1) in
                    abs_le d1 d2 x == ((i1 <: natlt (d2 @! 0)), abs_le (tail d1) (tail d2) i2)))
  = ()

let up_cons (#r : nat) (d : shape r { ICons? d }) (v : conc d)
  : Lemma (let i1, i2 = v <: szlt (d @! 0) & conc (tail d) in
           up v == ((SZ.v i1 <: natlt (d @! 0)), up i2))
  = ()

inline_for_extraction noextract
let conc_cons (#r : erased nat) (#d : shape r { ICons? d })
  (p : szlt (d @! 0) & conc (tail d)) : conc d
  = p

// Concrete counterpart of abs_le, commuting with up. The recursion is driven by
// the (non-erasable) cshapes: matching the erasable `shape` directly to build an
// informative `conc` would force a ghost computation.
inline_for_extraction noextract
[@@strict_on_arguments [3]]
let rec conc_le (#r : erased nat) (#d1 #d2 : shape r { shape_le d1 d2 })
  (cd1 : cshape d1) (cd2 : cshape d2) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_le d1 d2 (up x)})
  = match cd1 with
    | CNil -> ()
    | CCons #_ #_ _ #t1 ct1 ->
      (match cd2 with
       | CCons #_ #_ _ #t2 ct2 ->
         assert (r > 0);
         // Destructure *and* rebuild through the cshape-bound tail shapes `t1`/
         // `t2` (which reduce to concrete shapes at a JIT instantiation) rather
         // than `conc (tail d)`: `tail` does not reduce during extraction, so
         // `conc (tail d)` leaves the tuple tail as an unreduced recursive `conc`
         // type that F* casts to Top and karamel then rejects (Warning 26).
         let i1, i2 = x <: szlt (d1 @! 0) & conc t1 in
         let res : szlt (d2 @! 0) & conc t2 = ((i1 <: szlt (d2 @! 0)), conc_le ct1 ct2 i2) in
         abs_le_cons d1 d2 (up x);
         up_cons d1 x;
         up_cons d2 res;
         res)

let abs_set_at2_cons (#r : nat) (d1 d2 : shape r { shape_le d1 d2 }) (dim : natlt r)
  (idx : natlt (d2 @! dim)) (x : abs d1)
  : Lemma (ensures (let i1, i2 = x <: natlt (d1 @! 0) & abs (tail d1) in
                    abs_set_at2 d1 d2 dim idx x ==
                      (if dim = 0
                       then ((idx <: natlt (d2 @! 0)), abs_le (tail d1) (tail d2) i2)
                       else ((i1 <: natlt (d2 @! 0)),
                             abs_set_at2 (tail d1) (tail d2) (dim - 1) idx i2))))
  = ()

inline_for_extraction noextract
[@@strict_on_arguments [3]]
let rec conc_set_at2 (#r : erased nat) (#d1 #d2 : shape r { shape_le d1 d2 })
  (cd1 : cshape d1) (cd2 : cshape d2) (dim : szlt r) (idx : szlt (d2 @! dim)) (x : conc d1)
  : Tot (c : conc d2 {up c == abs_set_at2 d1 d2 dim idx (up x)})
  = assert r > 0;
    match cd1 with
    | CCons #_ #_ _ #t1 ct1 ->
      (match cd2 with
       | CCons #_ #_ _ #t2 ct2 ->
         let (i1,i2) = x <: szlt (d1 @! 0) & conc t1 in
         abs_set_at2_cons d1 d2 (SZ.v dim) idx (up x);
         up_cons d1 x;
         // Build through the cshape-bound tail shape `t2` (concrete at a JIT
         // instantiation), not `conc (tail d2)`, which extraction leaves stuck.
         let res : szlt (d2 @! 0) & conc t2 =
           if dim = 0sz then
             ((idx <: szlt (d2 @! 0)), conc_le ct1 ct2 i2)
           else
             ((i1 <: szlt (d2 @! 0)),
              conc_set_at2 ct1 ct2 (dim -^ 1sz) idx i2) in
         up_cons d2 res;
         res)

(* ----------------------------------------------------------------------- *)
(* Coordinate get / narrow: the pieces the cat kernel needs, expressed so no
   `conc (modulo_i dim d)` type ever appears in extracted code (that type stays
   a stuck application at extraction time and karamel casts it to `any`). The
   recursions are cshape-driven, exactly like `conc_le`/`conc_set_at2`. *)

(* modulo_i unfolding facts, used to line up the shapes narrowed across `dim`. *)
let modulo_zero (#n : pos) (d : shape n)
  : Lemma (modulo_i 0 d == tail d)
  = match d with | ICons _ _ -> ()

let modulo_succ (#n : nat) (dim : natlt n { dim > 0 }) (d : shape n)
  : Lemma (modulo_i dim d == ICons (d @! 0) (modulo_i (dim - 1) (tail d)))
  = match d with | ICons _ _ -> ()

inline_for_extraction noextract
[@@strict_on_arguments [2]]
let rec conc_get_at (#r : erased nat) (#d : shape r) (cd : cshape d) (dim : szlt r) (x : conc d)
  : Tot (v : szlt (d @! dim) { SZ.v v == abs_get_at (SZ.v dim) (up x) })
= match cd with
    | CCons #_ #_ _ #t ct ->
      let (h, tl) = x <: szlt (d @! 0) & conc t in
      up_cons d x;
      if dim = 0sz then (h <: szlt (d @! dim))
      else conc_get_at ct (dim -^ 1sz) tl

inline_for_extraction noextract
[@@strict_on_arguments [4]]
let rec conc_narrow (#r : erased nat) (dim : szlt r) (#d1 #d2 : shape r)
  (cd1 : cshape d1) (cd2 : cshape d2)
  (pf : squash (modulo_i (SZ.v dim) d1 == modulo_i (SZ.v dim) d2))
  (v : sz) (pfv : squash (SZ.v v < d1 @! dim)) (x : conc d2)
  : Tot (c : conc d1 { up c == abs_narrow (SZ.v dim) d1 d2 (SZ.v v) (up x) })
  = let vv : szlt (d1 @! dim) = v in
    match cd1 with
    | CCons #_ #_ _ #t1 ct1 ->
      (match cd2 with
       | CCons #_ #_ _ #t2 ct2 ->
         let (h, tl) = x <: szlt (d2 @! 0) & conc t2 in
         up_cons d2 x;
         let res : szlt (d1 @! 0) & conc t1 =
           if dim = 0sz then (
             modulo_zero d1; modulo_zero d2;
             ((vv <: szlt (d1 @! 0)), (tl <: conc t1))
           ) else (
             modulo_succ (SZ.v dim) d1; modulo_succ (SZ.v dim) d2;
             ((h <: szlt (d1 @! 0)),
              conc_narrow (dim -^ 1sz) ct1 ct2 () v () tl)
           ) in
         up_cons d1 res;
         res)