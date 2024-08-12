module Pulse.Lib.BigStar

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2

val bigstar
  (#[exact (`0)][@@@equate_strict] uid: int)
  ([@@@equate_strict] m : nat)
  ([@@@equate_strict] n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
: Tot slprop

val bigstar_split
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
  (i : nat { m <= i /\ i <= n })
: Lemma (ensures bigstar #u1 m n f == bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** bigstar #u1 i n (fun (j: nat { i <= j /\ j < n }) -> f j))

val bigstar_star
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f g h : (i:nat { m <= i /\ i < n }) -> slprop)
  (heq : (i:nat { m <= i /\ i < n }) -> squash (f i ** g i == h i))
: Lemma (bigstar #u1 m n f ** bigstar #u1 m n g == bigstar #u1 m n h)

val bigstar_congr
  (#u1 #u2: int)
  (m : nat)
  (n : nat { m <= n })
  (m' : nat)
  (n' : nat { m' <= n' /\ n' - m' == n - m })
  (f  : (i:nat { m <= i /\ i < n }) -> slprop)
  (f' : (i:nat { m' <= i /\ i < n' }) -> slprop)
  (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m'+i))))
: Lemma (bigstar #u1 m n f == bigstar #u2 m' n' f')


val bigstar_extensionality
  (#[exact (`0)] u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  (h: ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
  : stt_ghost unit emp_inames
      (requires bigstar #u1 m n f)
      (ensures fun _ -> bigstar #u1 m n g)

ghost
fn bigstar_eta
  ()
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n (fun i -> f i)

ghost
fn bigstar_uneta
  ()
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n (fun i -> f i)
  ensures  bigstar #u1 m n f

ghost
fn bigstar_rw_congr
  (#u1: int)
  (m : nat)
  (n : nat { m <= n })
  (f  : (i:nat { m <= i /\ i < n }) -> slprop)
  (f' : (i:nat { m <= i /\ i < n }) -> slprop)
  (h : (i:nat{m <= i /\ i < n}) -> squash (f i == f' i))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n f'

ghost
fn bigstar_extract
  (#u1 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (i : nat { m <= i /\ i < n })
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** f i ** bigstar #u1 (i+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j)

ghost
fn bigstar_compose
  (#u1 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (i : nat { m <= i /\ i < n })
  requires bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** f i ** bigstar #u1 (i+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j)
  ensures  bigstar #u1 m n f

ghost
fn bigstar_zs_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < m} -> slprop))
  requires bigstar #u1 m m f
  ensures  emp

val bigstar_zs_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < m} -> slprop))
  : stt_ghost unit emp_inames
      (requires emp)
      (ensures  fun _ -> bigstar #u1 m m f)

ghost
fn bigstar_single_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires bigstar #u1 m (m+1) f
  ensures  f m

val bigstar_single_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  : stt_ghost unit emp_inames
      (requires f m)
      (ensures  fun _ -> bigstar #u1 m (m+1) f)

ghost
fn bigstar_emp_elim
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  requires bigstar #u1 m n (fun _ -> emp)
  ensures  emp

val bigstar_emp_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (n : nat {m <= n})
  : stt_ghost unit emp_inames
      (requires emp)
      (ensures  fun _ -> bigstar #u1 m n (fun _ -> emp))

// No meta args in pulse syntax, so F* val for now
val bigstar_map
  (#u1 : int)
  (#[exact (`0)]u2 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (#g: (i: nat{m <= i /\ i < n} -> slprop))
  (stt: ((i: nat{m <= i /\ i < n}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
: stt_ghost unit emp_inames
            (bigstar #u1 m n f)
            (fun _ -> bigstar #u2 m n g)

ghost fn bigstar_commute
  (#u1 #u2 : int)
  (m0 : nat)
  (n0 : nat {m0 <= n0})
  (m1 : nat)
  (n1 : nat {m1 <= n1})
  (f: (i: nat{m0 <= i /\ i < n0} -> j: nat{m1 <= j /\ j < n1} -> slprop))
  requires bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> f i j))
  ensures  bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j))

// No meta args in pulse syntax, so F* val for now
[@@allow_ambiguous]
val bigstar_zip
  (#u1 #u2 : int)
  (#[exact (`0)]u3 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
: stt_ghost unit
            emp_inames
            (bigstar #u1 m n f ** bigstar #u2 m n g)
            (fun _ -> bigstar #u3 m n (fun i -> f i ** g i))

// No meta args in pulse syntax, so F* val for now
val bigstar_unzip
  (#[exact (`0)]u1 : int)
  (#[exact (`0)]u2 : int)
  (#u3 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
: stt_ghost unit
            emp_inames
            (bigstar #u3 m n (fun i -> f i ** g i))
            (fun _ -> bigstar #u1 m n f ** bigstar #u2 m n g)

ghost fn bigstar_if_elim
  (#u1 : int)
  (#m: nat)
  (#n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  requires bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (op_Equality #int i x) (p i) emp)
  ensures  p x

val bigstar_if_intro
  (#[exact (`0)]u1 : int)
  (m: nat)
  (n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  : stt_ghost unit emp_inames
      (requires p x)
      (ensures  fun _ -> bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (op_Equality #int i x) (p i) emp))

class permutation (a:Type) = {
   f          : a -> a;
   g          : a -> a;
   [@@@FStar.Tactics.Typeclasses.no_method]
   proof : (x: a) -> (y: a) -> squash (f x == y <==> g y == x);
}

ghost fn bigstar_permute
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (p: permutation (i: erased nat{m <= i /\ i < n}))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n (fun i -> f (p.f i))
