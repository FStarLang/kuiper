module Pulse.Lib.BigStar

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Mul
open Pulse.Lib.PartitionRange

val bigstar
  (#[exact (`0)][@@@mkey] uid: int)
  ([@@@mkey] m : nat)
  ([@@@mkey] n : nat {m <= n})
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

val bigstar_eq
  (#u1 #u2: int)
  (m : nat)
  (n : nat {m <= n})
  (f g : (i:nat { m <= i /\ i < n }) -> slprop)
  : Lemma (requires (forall i. m <= i /\ i < n ==> f i == g i))
          (ensures  bigstar #u1 m n f == bigstar #u2 m n g)
          [SMTPat (bigstar #u1 m n f); SMTPat (bigstar #u2 m n g)]

val bigstar_ext u1 u2 (m:nat) (n:nat{m<=n}) (f g: ((i:nat{m<=i /\ i<n}) -> slprop))
: Lemma
  (requires FStar.FunctionalExtensionality.feq f g)
  (ensures bigstar #u1 m n f == bigstar #u2 m n g)

val bigstar_extensionality_lem
  (u1 u2 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  (h: ((i: nat{m <= i /\ i < n}) -> slprop_equiv (f i) (g i)))
  : Lemma (slprop_equiv (bigstar #u1 m n f) (bigstar #u2 m n g))

ghost
fn bigstar_extensionality
  (#[exact (`0)] u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  (h: ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n g

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

ghost
fn bigstar_zs_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < m} -> slprop))
  requires emp
  ensures  bigstar #u1 m m f

ghost
fn bigstar_single_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires bigstar #u1 m (m+1) f
  ensures  f m

ghost
fn bigstar_single_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires f m
  ensures  bigstar #u1 m (m+1) f

ghost
fn bigstar_emp_elim
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  requires bigstar #u1 m n (fun _ -> emp)
  ensures  emp

ghost
fn bigstar_emp_elim'
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (f : (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f ** pure (forall x. f x == emp)
  ensures  emp

ghost
fn rec bigstar_emp_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (n : nat {m <= n})
  requires emp
  ensures  bigstar #u1 m n (fun _ -> emp)

ghost
fn bigstar_map
  (#u1 : int)
  (#[exact (`0)]u2 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (#g: (i: nat{m <= i /\ i < n} -> slprop))
  (stt: ((i: nat{m <= i /\ i < n}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
  requires bigstar #u1 m n f
  ensures  bigstar #u2 m n g

ghost
fn bigstar_commute
  (#u1 #u2 : int)
  (m0 : nat)
  (n0 : nat {m0 <= n0})
  (m1 : nat)
  (n1 : nat {m1 <= n1})
  (f: (i: nat{m0 <= i /\ i < n0} -> j: nat{m1 <= j /\ j < n1} -> slprop))
  requires bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> f i j))
  ensures  bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j))

[@@allow_ambiguous]
ghost
fn bigstar_zip
  (#u1 #u2 : int)
  (#[exact (`0)]u3 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f ** bigstar #u2 m n g
  ensures  bigstar #u3 m n (fun (i: nat { m <= i /\ i < n }) -> f i ** g i)

ghost
fn bigstar_unzip
  (#[exact (`0)]u1 : int)
  (#[exact (`0)]u2 : int)
  (#u3 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u3 m n (fun i -> f i ** g i)
  ensures  bigstar #u1 m n f ** bigstar #u2 m n g

ghost
fn bigstar_if_elim
  (#u1 : int)
  (#m: nat)
  (#n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  requires bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp)
  ensures  p x

ghost
fn bigstar_if_intro
  (#[exact (`0)]u1 : int)
  (m: nat)
  (n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  requires p x
  ensures  bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp)

class permutation (a:Type) = {
   f          : a -> a;
   g          : a -> a;
   [@@@FStar.Tactics.Typeclasses.no_method]
   proof : (x: a) -> (y: a) -> squash (f x == y <==> g y == x);
}

instance perm_inv (#a:Type) (p: permutation a) : permutation a = {
  f = p.g;
  g = p.f;
  proof = fun x y -> p.proof y x
}

ghost
fn bigstar_permute
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (p: permutation (i: nat{m <= i /\ i < n}))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n (fun i -> f (p.f i))

ghost
fn bigstar_exists
  (#a : Type0) // TODO: arbitrary type doesn't work here?
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: a -> (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n (fun i -> exists* (x: a). f x i)
  ensures  exists* (x: (i:nat { m <= i /\ i < n }) -> a). bigstar #u1 m n (fun i -> f (x i) i)

ghost
fn bigstar_flatten
  (#u1 #u2 : int)
  (#n1 : nat)
  (#n2 : nat)
  (#f: (i: nat{0 <= i /\ i < n1} -> j: nat{0 <= j /\ j < n2} -> slprop))
  requires bigstar #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i))
  ensures  bigstar #u1 0 (n1 * n2) (fun i -> f (i / n2) (i % n2))

ghost
fn bigstar_unflatten
  (#u1 #u2 : int)
  (#n1 : nat)
  (#n2 : nat)
  (#f: (i: nat{0 <= i /\ i < n1} -> j: nat{0 <= j /\ j < n2} -> slprop))
  requires bigstar #u1 0 (n1 * n2) (fun i -> f (i / n2) (i % n2))
  ensures  bigstar #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i))

ghost
fn bigstar_partition
  (n0:nat)
  (n1:nat)
  (f0: (idx 0 n0 -> slprop))
  (partition: disjoint_partitions 0 n0 n1)
requires
  bigstar 0 n0 f0
ensures
  bigstar 0 n1 (fun i -> star_over_partition f0 (select partition i))

ghost
fn bigstar_partition_inv
  (n0:nat)
  (n1:nat)
  (f0: (idx 0 n0 -> slprop))
  (partition: disjoint_partitions 0 n0 n1)
requires
 bigstar 0 n1 (fun i -> star_over_partition f0 (select partition i))
ensures
  bigstar 0 n0 f0
