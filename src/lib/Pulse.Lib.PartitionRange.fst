module Pulse.Lib.PartitionRange
open FStar.FiniteSet.Base
open FStar.FiniteSet.Ambient
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
= if from = to then Set.emptyset
  else Set.union (p from) (union_partitions_aux p (from + 1 <: nat) to)


let rec union_partitions_split
    (#m #n #k : nat)
    (p:partitions m n k)
    (from:nat)
    (mid:nat)
    (to:nat { from <= mid /\ mid <= to /\ to <= k})
: Lemma
    (ensures 
      union_partitions_aux p from to `Set.equal`
     (union_partitions_aux p from mid `Set.union` union_partitions_aux p mid to))
    (decreases (mid - from))
= if mid = from
  then (
    assert (forall (s:Set.set nat). Set.equal (Set.union Set.emptyset s) s)
  )
  else (
    union_partitions_split p (from + 1) mid to
  )

// Union of all partitions in p
let union_partitions #m #n #k (p:partitions m n k) = union_partitions_aux p 0 k

// The set of all indices in the range [m, n)
let rec range (m:nat) (n:nat { m <= n }) 
: Tot (s:idx_set m n { forall x. Set.mem x s <==> m <= x /\ x < n })
      (decreases n - m)
= if m = n then Set.emptyset
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
= if cardinality part = 0
  then emp
  else (
    let i = Set.choose part in
    let part' = Set.remove i part in
    f i ** star_over_partition f part'
  )

let rec union_partitions_elements
    (#m #n #k : nat)
    (p:disjoint_partitions m n k)
    (from:nat)
    (to:nat { from <= to /\ to <= k})
: Lemma
  (ensures (forall i.{:pattern (i `Set.mem` union_partitions_aux p from to)}
              i `Set.mem` union_partitions_aux p from to
              <==> (exists (j:nat{from <= j /\ j < to}). {:pattern (select p j)}
                       i `Set.mem` select p j)))
  (decreases to - from)
= if from = to then ()
  else (
    union_partitions_elements p (from + 1) to;
    assert (union_partitions_aux p from to `Set.equal`
            Set.union (select p from) (union_partitions_aux p (from + 1) to))
  )

let rec union_partitions_disjoint
    (#m #n #k : nat)
    (p:disjoint_partitions m n k)
    (from:nat)
    (mid:nat)
    (to:nat { from <= mid /\ mid <= to /\ to <= k})
: Lemma
  (ensures Set.disjoint (union_partitions_aux p from mid) (union_partitions_aux p mid to))
  (decreases mid - from)
= if mid = from then ()
  else (
    union_partitions_disjoint p (from + 1) mid to;
    assert (union_partitions_aux p from mid `Set.equal`
            Set.union (select p from) (union_partitions_aux p (from + 1) mid));
    union_partitions_elements p mid to
  )
  
let union_cardinality_fact (#a:eqtype) (s1 s2:set a)
: Lemma (ensures Set.cardinality (Set.union s1 s2) == Set.cardinality s1 + Set.cardinality s2 - Set.cardinality (Set.intersection s1 s2))
= ()

let disjoint_cardinality_fact (#a:eqtype) (s1 s2:set a)
: Lemma 
  (requires Set.disjoint s1 s2)
  (ensures Set.cardinality (Set.intersection s1 s2) == 0)
= ()

let star_over_partition_singleton
  (#m:nat) (#n : nat { m <= n })
  (f: (idx m n -> slprop) )
  (x: idx m n)
: Lemma 
  (ensures star_over_partition f (Set.singleton x) == f x)
= slprop_equivs()

#push-options "--fuel 2 --ifuel 0 --z3rlimit_factor 16 --split_queries no"
#restart-solver
let rec star_over_partition_split
  (#m:nat) (#n : nat { m <= n })
  (f: (idx m n -> slprop) )
  (s0: idx_set m n)
  (s1: idx_set m n { Set.disjoint s0 s1 })
: Lemma 
  (ensures star_over_partition f (Set.union s0 s1) ==
           star_over_partition f s0 ** star_over_partition f s1)
  (decreases (cardinality (Set.union s0 s1)))
= if Set.cardinality (Set.union s0 s1) = 0
  then slprop_equivs()
  else ( 
    let x = Set.choose (Set.union s0 s1) in
    let s0_ = s0 in
    let s1_ = s1 in
    let aux (s0:idx_set m n { Set.mem x s0 }) (s1:idx_set m n)
    : Lemma 
      (requires Set.union s0 s1 `Set.equal` Set.union s0_ s1_ /\ Set.disjoint s0 s1)
      (ensures star_over_partition f (Set.union s0 s1) ==
               star_over_partition f s0 ** star_over_partition f s1)
    = let s0' = Set.remove x s0 in
      assert (not (Set.mem x s1));
      union_cardinality_fact s0' s1; 
      disjoint_cardinality_fact s0' s1;
      calc (==) {
        star_over_partition f (Set.union s0 s1);
      (==) {}
        f x ** (star_over_partition f (Set.remove x (Set.union s0 s1)));
      (==) { assert ((Set.remove x s0 `Set.union` s1) `Set.equal` Set.remove x (s0 `Set.union` s1)) }
        f x ** (star_over_partition f (Set.remove x s0 `Set.union` s1));
      (==) {  star_over_partition_split f s0' s1 }
        f x ** (star_over_partition f (Set.remove x s0) ** (star_over_partition f s1));
      (==) { slprop_equivs() }
        (f x ** star_over_partition f (Set.remove x s0)) ** star_over_partition f s1;
      };
      if Set.cardinality s1 = 0
      then (
        assert (Set.equal (Set.union s0 s1) s0)
      )
      else (
        assert (Set.equal (Set.union (Set.singleton x) s0') s0);
        union_cardinality_fact (Set.singleton x) s0';
        disjoint_cardinality_fact (Set.singleton x) s0';
        calc (==) {
          (f x ** star_over_partition f (Set.remove x s0));
        (==) { star_over_partition_singleton f x }
          (star_over_partition f (Set.singleton x) ** star_over_partition f (Set.remove x s0));
        (==) { star_over_partition_split f (Set.singleton x) s0' }
          star_over_partition f (Set.union (Set.singleton x) s0');
        (==) { }
          star_over_partition f s0;
        }
      )
    in
    if Set.mem x s0
    then aux s0 s1
    else (
      aux s1 s0;
      assert (Set.equal (Set.union s0 s1) (Set.union s1 s0));
      slprop_equivs()
    )
  )
#pop-options
  
let rec star_over_partition_reindex 
      (m:nat)
      (n:nat {m < n})
      (f: idx m n -> slprop) 
      (s: idx_set m n { forall x. Set.mem x s ==> m < x /\ x < n })
: Lemma 
  (ensures star_over_partition #m #n f s == star_over_partition #(m+1) #n f (s <: idx_set (m + 1) n))
  (decreases cardinality s)
= if Set.cardinality s = 0
  then ()
  else (
    let x = Set.choose s in
    let s' = Set.remove x s in
    star_over_partition_reindex m n f s'
  )