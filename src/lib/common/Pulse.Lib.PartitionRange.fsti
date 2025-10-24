module Pulse.Lib.PartitionRange
open FStar.FiniteSet.Base
open Pulse.Lib.Pervasives
module Set = FStar.FiniteSet.Base

// The type of an index in the range [m, n)
let idx (m n:nat) = i:nat { m <= i /\ i < n }

// A set of indices in the range [m, n)
let idx_set (m n:nat) = s:set nat { forall x. x `Set.mem` s ==> m <= x /\ x < n }

// The range [m, n) mapped into k subsets, not necessarily disjoint
let partitions (m:nat) (n:nat) (k:nat) = i:nat{ i < k } -> idx_set m n

// Select the i-th partition
let select #m #n #k (p:partitions m n k) (i:nat{ i < k }) : idx_set m n = p i

// The union of the partitions in p from [from, to)
let rec union_partitions_aux #m #n #k
      (p:partitions m n k)
      (from:nat)
      (to:nat{ from <= to /\ to <= k })
: Tot (s:idx_set m n { forall (j:nat { from <= j /\ j < to }). select p j `Set.subset` s })
      (decreases to - from)
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  if from = to then Set.emptyset
  else Set.union (p from) (union_partitions_aux p (from + 1 <: nat) to)

val union_partitions_split
    (#m #n #k : nat)
    (p:partitions m n k)
    (from:nat)
    (mid:nat)
    (to:nat { from <= mid /\ mid <= to /\ to <= k})
: Lemma
    (ensures
      union_partitions_aux p from to `Set.equal`
     (union_partitions_aux p from mid `Set.union` union_partitions_aux p mid to))

// Union of all partitions in p
let union_partitions #m #n #k (p:partitions m n k) = union_partitions_aux p 0 k

// The set of all indices in the range [m, n)
let rec range (m:nat) (n:nat { m <= n })
: Tot (s:idx_set m n { forall x. Set.mem x s <==> m <= x /\ x < n })
      (decreases n - m)
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  if m = n then Set.emptyset
  else Set.union (Set.singleton m) (range (m + 1) n)

// All sets in parts are pairwise disjoint
let parts_disjoint #m #n #k (parts:partitions m n k) =
  forall (i j:nat). {:pattern (select parts i); (select parts j) }
      i < j /\ j < k ==> Set.disjoint (select parts i) (select parts j)

// parts covers the range [m, n) except for the indices in except
let parts_covers_range_except #m #n #k (parts:partitions m n k) (except:Set.set nat) =
  m <= n /\
  Set.difference (range m n) except `Set.subset` union_partitions parts

// parts covers the entire range [m, n)
let parts_covers_range #m #n #k (parts:partitions m n k) =
  parts_covers_range_except parts Set.emptyset

// A refinition of partitions that enforces the disjointness and coverage properties
let disjoint_partitions (m:nat) (n:nat) (k:nat) =
  parts:partitions m n k {
    parts_disjoint parts /\
    parts_covers_range parts
  }

// Iterated star over the elements of part
let rec star_over_partition (#m:nat) (#n:nat{m<=n}) (f:idx m n -> slprop) (part:idx_set m n)
: Tot slprop (decreases (cardinality part))
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  if cardinality part = 0
  then emp
  else (
    let i = Set.choose part in
    let part' = Set.remove i part in
    f i ** star_over_partition f part'
  )

val union_partitions_elements
    (#m #n #k : nat)
    (p:disjoint_partitions m n k)
    (from:nat)
    (to:nat { from <= to /\ to <= k})
: Lemma
  (ensures (forall i.{:pattern (i `Set.mem` union_partitions_aux p from to)}
              i `Set.mem` union_partitions_aux p from to
              <==> (exists (j:nat{from <= j /\ j < to}). {:pattern (select p j)}
                       i `Set.mem` select p j)))

val union_partitions_disjoint
    (#m #n #k : nat)
    (p:disjoint_partitions m n k)
    (from:nat)
    (mid:nat)
    (to:nat { from <= mid /\ mid <= to /\ to <= k})
: Lemma
  (ensures Set.disjoint (union_partitions_aux p from mid) (union_partitions_aux p mid to))

val union_cardinality_fact (#a:eqtype) (s1 s2:set a)
: Lemma (ensures Set.cardinality (Set.union s1 s2) == Set.cardinality s1 + Set.cardinality s2 - Set.cardinality (Set.intersection s1 s2))

val disjoint_cardinality_fact (#a:eqtype) (s1 s2:set a)
: Lemma
  (requires Set.disjoint s1 s2)
  (ensures Set.cardinality (Set.intersection s1 s2) == 0)

val star_over_partition_singleton
  (#m:nat) (#n : nat { m <= n })
  (f: (idx m n -> slprop) )
  (x: idx m n)
: Lemma
  (ensures FStar.FiniteSet.Base.all_finite_set_facts_lemma();
    star_over_partition f (Set.singleton x) == f x)

val star_over_partition_split
  (#m:nat) (#n : nat { m <= n })
  (f: (idx m n -> slprop) )
  (s0: idx_set m n)
  (s1: idx_set m n { Set.disjoint s0 s1 })
: Lemma
  (ensures
    FStar.FiniteSet.Base.all_finite_set_facts_lemma();
    star_over_partition f (Set.union s0 s1) ==
    star_over_partition f s0 ** star_over_partition f s1)

val star_over_partition_reindex
      (m:nat)
      (n:nat {m < n})
      (f: idx m n -> slprop)
      (s: idx_set m n { forall x. Set.mem x s ==> m < x /\ x < n })
: Lemma
  (ensures star_over_partition #m #n f s == star_over_partition #(m+1) #n f (s <: idx_set (m + 1) n))