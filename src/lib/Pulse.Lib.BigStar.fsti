module Pulse.Lib.BigStar

open Pulse.Lib.Pervasives
open FStar.Tactics.V2

let rec bigstar
  (#[exact (`0)] uid: int)
  (m : nat)
  (n : nat {m <= n})
  ([@@@equate_by_smt] f : (i:nat { m <= i /\ i < n } -> vprop))
: Tot vprop (decreases n - m) =
  if m = n then emp else f m ** bigstar #uid (m+1) n f

val star_aci () :
    squash (
      (forall (a b : vprop). {:pattern (a ** b)} a ** b == b ** a) /\
      (forall (a : vprop). {:pattern (a ** emp)} a ** emp == a) /\
      (forall (a b c : vprop). {:pattern (a ** b ** c)} a ** (b ** c) == (a ** b) ** c))

val bigstar_split (#uid: int) (m : nat) (n : nat {m <= n}) f (i : nat { m <= i /\ i <= n }) :
    Lemma (ensures bigstar #uid m n f == bigstar #uid m i f ** bigstar #uid i n f)

val bigstar_star (#uid: int) (#uid_l: int) (#uid_r: int) (m : nat) (n : nat {m <= n}) f g h
    (heq : (i:nat { m <= i /\ i < n }) -> squash (f i ** g i == h i)) :
    Pure
      (squash (bigstar #uid_l m n f ** bigstar #uid_r m n g == bigstar #uid m n h))
      (requires True) //forall i. f i ** g i == h i)
      (ensures fun _ -> True)

val bigstar_congr (#uid: int) (m : nat) (n : nat { m <= n }) (m' : nat) (n' : nat { m' <= n' /\ n' - m' == n - m })
    (f : (i:nat { m <= i /\ i < n }) -> vprop) (f' : (i:nat { m' <= i /\ i < n' }) -> vprop) 
    (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m'+i))))
    :
    Pure
      (squash (bigstar #uid m n f == bigstar #uid m' n' f'))
      (requires True)
      (ensures fun _ -> True)
      (decreases n-m)

val bigstar_rw_congr
   (#uid: int) (m : nat) (n : nat { m <= n })
   (f : (i:nat { m <= i /\ i < n }) -> vprop)
   (f' : (i:nat { m <= i /\ i < n }) -> vprop) 
   (h : ((i:nat{m <= i /\ i < n}) -> squash (f i == f' i)))
  : stt_ghost unit emp_inames
              (bigstar #uid m n f)
              (fun _ -> bigstar #uid m n f')

val bigstar_extract
    (#uid: int) (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (i : nat { m <= i /\ i < n })
  : stt_ghost unit emp_inames
              (bigstar #uid m n f)
              (fun _ -> bigstar #uid m i f ** f i ** bigstar #uid (i+1) n f)

val bigstar_compose
    (#uid0: int) (#uid1: int) (#[exact (`0)] uid2: int) (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (i : nat { m <= i /\ i < n })
  : stt_ghost unit emp_inames
              (bigstar #uid0 m i f ** f i ** bigstar #uid1 (i+1) n f)
              (fun _ -> bigstar #uid2 m n f)

[@@allow_ambiguous]
val bigstar_zip
    (#uid0: int) (#uid1: int) (#[exact (`0)] uid2: int) (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
  : stt_ghost unit emp_inames
              (bigstar #uid0 m n f ** bigstar #uid1 m n g)
              (fun _ -> bigstar #uid2 m n (fun i -> f i ** g i))

val bigstar_map
    (#uid: int) (#m : nat)
    (#n : nat {m <= n})
    (#f: (i: nat{m <= i /\ i < n} -> vprop))
    (#g: (i: nat{m <= i /\ i < n} -> vprop))
    (stt: ((i: nat{m <= i /\ i < n}) -> stt_ghost unit emp_inames
              (f i)
              (fun _ -> g i)))
  : stt_ghost unit emp_inames
              (bigstar #uid m n f)
              (fun _ -> bigstar #uid m n g)

val bigstar_unzip
    (#uid0: int) (#[exact (`0)] uid1: int) (#[exact (`0)] uid2: int) (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
  : stt_ghost unit emp_inames
              (bigstar #uid0 m n (fun i -> f i ** g i))
              (fun _ -> bigstar #uid1 m n f ** bigstar #uid2 m n g)
