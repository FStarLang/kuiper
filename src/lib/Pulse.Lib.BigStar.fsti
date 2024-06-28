module Pulse.Lib.BigStar

open Pulse.Lib.Pervasives
open FStar.Tactics.V2

val bigstar
  (#[exact (`0)][@@@equate_strict] uid: int)
  ([@@@equate_strict] m : nat)
  ([@@@equate_strict] n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> vprop))
: Tot vprop

val bigstar_split
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> vprop))
  (i : nat { m <= i /\ i <= n })
: Lemma (ensures bigstar m n f == bigstar m i f ** bigstar i n f)

val bigstar_star
  (m : nat)
  (n : nat {m <= n})
  (f g h : (i:nat { m <= i /\ i < n }) -> vprop)
  (heq : (i:nat { m <= i /\ i < n }) -> squash (f i ** g i == h i))
: Lemma (bigstar m n f ** bigstar m n g == bigstar m n h)

val bigstar_congr
  (m : nat)
  (n : nat { m <= n })
  (m' : nat)
  (n' : nat { m' <= n' /\ n' - m' == n - m })
  (f  : (i:nat { m <= i /\ i < n }) -> vprop)
  (f' : (i:nat { m' <= i /\ i < n' }) -> vprop)
  (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m'+i))))
: Lemma (bigstar m n f == bigstar m' n' f')

val bigstar_rw_congr
  (m : nat)
  (n : nat { m <= n })
  (f  : (i:nat { m <= i /\ i < n }) -> vprop)
  (f' : (i:nat { m <= i /\ i < n }) -> vprop)
  (h : ((i:nat{m <= i /\ i < n}) -> squash (f i == f' i)))
: stt_ghost unit
            emp_inames
            (bigstar m n f)
            (fun _ -> bigstar m n f')

val bigstar_extract
  (#u1 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> vprop))
  (i : nat { m <= i /\ i < n })
: stt_ghost unit emp_inames
            (bigstar #u1 m n f)
            (fun _ -> bigstar #u1 m i f ** f i ** bigstar #u1 (i+1) n f)

val bigstar_compose
  (#u1 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> vprop))
  (i : nat { m <= i /\ i < n })
: stt_ghost unit
            emp_inames
            (bigstar #u1 m i f ** f i ** bigstar #u1 (i+1) n f)
            (fun _ -> bigstar #u1 m n f)

val bigstar_map
  (#u1 : int)
  (#[exact (`0)]u2 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> vprop))
  (#g: (i: nat{m <= i /\ i < n} -> vprop))
  (stt: ((i: nat{m <= i /\ i < n}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
: stt_ghost unit emp_inames
            (bigstar #u1 m n f)
            (fun _ -> bigstar #u2 m n g)

[@@allow_ambiguous]
val bigstar_zip
  (#u1 #u2 : int)
  (#[exact (`0)]u3 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> vprop))
  (g: (i: nat{m <= i /\ i < n} -> vprop))
: stt_ghost unit
            emp_inames
            (bigstar #u1 m n f ** bigstar #u2 m n g)
            (fun _ -> bigstar #u3 m n (fun i -> f i ** g i))

val bigstar_unzip
  (#[exact (`0)]u1 : int)
  (#[exact (`0)]u2 : int)
  (#u3 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> vprop))
  (g: (i: nat{m <= i /\ i < n} -> vprop))
: stt_ghost unit
            emp_inames
            (bigstar #u3 m n (fun i -> f i ** g i))
            (fun _ -> bigstar #u1 m n f ** bigstar #u2 m n g)

val bigstar_extensionality
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> vprop))
  (g: (i: nat{m <= i /\ i < n} -> vprop))
  (h: ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
: stt_ghost unit
            emp_inames
            (bigstar m n f)
            (fun _ -> bigstar m n g)

val bigstar_eta
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> vprop))
: stt_ghost unit
            emp_inames
            (bigstar #u1 m n f)
            (fun _ -> bigstar #u1 m n (fun i -> f i))

val bigstar_uneta
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> vprop))
: stt_ghost unit
            emp_inames
            (bigstar #u1 m n (fun i -> f i))
            (fun _ -> bigstar #u1 m n f)
