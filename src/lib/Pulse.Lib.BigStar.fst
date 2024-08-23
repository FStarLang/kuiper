module Pulse.Lib.BigStar

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Mul
open FStar.FunctionalExtensionality
module SZ = FStar.SizeT

let rec bigstar
  (#uid : int)
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
: Tot slprop (decreases n - m) =
  if m = n then emp else f m ** bigstar #uid (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)

let bigstar_defn (#uid : int) (m : nat) (n : nat {m <= n}) (f : (i:nat { m <= i /\ i < n } -> slprop)) :
  Lemma (ensures bigstar #uid m n f == (if m = n then emp else f m ** bigstar #uid (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)))
  = ()

ghost fn bigstar_pop
  (#u1 : int)
  (#m : nat)
  (#n : nat {m < n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f
  ensures  f m ** bigstar #u1 (m + 1) n (fun (i: nat {(m + 1) <= i /\ i < n}) -> f i)
{
  unfold (bigstar #u1 m n f);
  rewrite (if m = n then emp else f m ** bigstar #u1 (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i))
      as  (                       f m ** bigstar #u1 (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i));
}

ghost fn bigstar_push
  (#u1 : int)
  (m : nat)
  (n : nat {m < n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  requires f m ** bigstar #u1 (m + 1) n (fun (i: nat {(m + 1) <= i /\ i < n}) -> f i)
  ensures  bigstar #u1 m n f
{
  rewrite (                       f m ** bigstar #u1 (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i))
      as  (if m = n then emp else f m ** bigstar #u1 (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i));
  fold (bigstar #u1 m n f);
}

let star_aci () :
    squash (
      (forall (a b : slprop). {:pattern (a ** b)} a ** b == b ** a) /\
      (forall (a : slprop). {:pattern (a ** emp)} a ** emp == a) /\
      (forall (a b c : slprop). {:pattern (a ** b ** c)} a ** (b ** c) == (a ** b) ** c)) =
  introduce forall (a b : slprop). a ** b == b ** a with elim_slprop_equiv (slprop_equiv_comm a b);
  introduce forall (a : slprop). a ** emp == a with elim_slprop_equiv (slprop_equiv_unit a);
  introduce forall (a b c : slprop). a ** (b ** c) == (a ** b) ** c with elim_slprop_equiv (slprop_equiv_assoc a b c)

let rec bigstar_feq (#u1 #u2: int) (m : nat) (n : nat {m <= n}) (f: (i: nat{m <= i /\ i < n} -> slprop)):
  Lemma (ensures bigstar #u1 m n f == bigstar #u2 m n (fun (i: nat { m <= i /\ i < n }) -> f i)) (decreases n - m) =
  star_aci ();
  if m = n then () else (
    bigstar_feq #u1 #u2 (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i);
    assert (f m ** (bigstar (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)) ==
            ((fun (i: nat { m <= i /\ i < n }) -> f i)) m ** (bigstar (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> (fun (i: nat { m <= i /\ i < n }) -> f i) i)));
    bigstar_defn #u1 m n f;
    bigstar_defn #u2 m n (fun (i: nat { m <= i /\ i < n }) -> f i);
    assert (bigstar #u1 m n f == bigstar #u2 m n (fun (i: nat { m <= i /\ i < n }) -> f i))
  )

let rec bigstar_split (#u1: int) (m : nat) (n : nat {m <= n}) (f: (i: nat{m <= i /\ i < n} -> slprop)) (i : nat { m <= i /\ i <= n }) :
    Lemma (ensures bigstar #u1 m n f == bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** bigstar #u1 i n (fun (j: nat { i <= j /\ j < n }) -> f j)) (decreases n - m) =
  star_aci ();
  if m = i then bigstar_feq #u1 #u1 i n f else (
    bigstar_split #u1 (m+1) n (fun (j: nat { (m + 1) <= j /\ j < n }) -> f j) i;
    bigstar_defn #u1 m n f;
    bigstar_defn #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j)
  )

let rec bigstar_star (#u1: int) (m : nat) (n : nat {m <= n}) (f g h : (i:nat { m <= i /\ i < n }) -> slprop)
    (heq : (i:nat { m <= i /\ i < n }) -> squash (f i ** g i == h i))
: Lemma (ensures bigstar #u1 m n f ** bigstar #u1 m n g == bigstar #u1 m n h)
        (decreases n - m)
= star_aci ();
  if m = n then () else (
    bigstar_star #u1 (m+1) n
      (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)
      (fun (i: nat { (m+1) <= i /\ i < n }) -> g i)
      (fun (i: nat { (m+1) <= i /\ i < n }) -> h i) heq;
    heq m;
    bigstar_defn #u1 m n f;
    bigstar_defn #u1 m n g;
    bigstar_defn #u1 m n h
  )

let rec bigstar_congr (#u1 #u2: int) (m : nat) (n : nat { m <= n }) (m' : nat) (n' : nat { m' <= n' /\ n' - m' == n - m })
    (f : (i:nat { m <= i /\ i < n }) -> slprop) (f' : (i:nat { m' <= i /\ i < n' }) -> slprop)
    (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m'+i))))
: Lemma (ensures bigstar #u1 m n f == bigstar #u2 m' n' f')
        (decreases n-m)
= if m = n then () else begin
    bigstar_congr #u1 #u2 (m+1) n (m'+1) n'
      (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)
      (fun (i: nat { (m'+1) <= i /\ i < n' }) -> f' i)
      (fun i -> h (i+1));
    h 0;
    bigstar_defn #u1 m n f;
    bigstar_defn #u2 m' n' f'
  end

ghost
fn __bigstar_extensionality
    (#u1: int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> slprop))
    (g: (i: nat{m <= i /\ i < n} -> slprop))
    (h: ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n g
{
  bigstar_congr #u1 #u1 m n m n f g (fun j -> h (m+j));
  ();
}
let bigstar_extensionality #u1 m n = __bigstar_extensionality #u1 m n

ghost
fn bigstar_eta
  ()
  (#u1: int)
  (#m : nat) (#n : nat{m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n (fun i -> f i)
{
  bigstar_extensionality #u1 m n f (fun i -> f i) (fun _ -> ());
}

ghost
fn bigstar_uneta
  ()
  (#u1: int)
  (#m : nat) (#n : nat{m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n (fun i -> f i)
  ensures  bigstar #u1 m n f
{
  bigstar_extensionality #u1 m n (fun i -> f i) f (fun _ -> ());
}

ghost
fn bigstar_rw_congr
   (#u1: int)
   (m : nat) (n : nat { m <= n })
   (f : (i:nat { m <= i /\ i < n }) -> slprop)
   (f' : (i:nat { m <= i /\ i < n }) -> slprop)
   (h : ((i:nat{m <= i /\ i < n}) -> squash (f i == f' i)))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n f'
{
  let h' :
    ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m+i)))
    = (fun (i:nat{i < n-m}) -> h (m+i));

  bigstar_congr #u1 #u1 m n m n f f' h';
  rewrite bigstar #u1 m n f as bigstar #u1 m n f';
  ();
}

ghost fn rec bigstar_extract
    (#u1 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> slprop))
    (i : nat { m <= i /\ i < n })
  requires bigstar #u1 m n f
  returns  _:unit
  ensures  bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** f i ** bigstar #u1 (i+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j)
  decreases (n-m)
{
  bigstar_pop #u1;
  if (m = i) {
    rewrite (emp ** f m ** bigstar #u1 (m+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j))
         as (bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** f i ** bigstar #u1 (i+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j));
  } else {
    bigstar_extract #u1 (m+1) n (fun (j: nat { (m+1) <= j /\ j < n }) -> f j) i;
    bigstar_push #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j);
  }
}

ghost fn rec bigstar_compose
    (#u1 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> slprop))
    (i : nat { m <= i /\ i < n })
  requires bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** f i ** bigstar #u1 (i+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j)
  returns  _:unit
  ensures  bigstar #u1 m n f
  decreases (n-m)
{
  if (m = i) {
    rewrite (bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j))
         as (emp);
    bigstar_push #u1 i n (fun (j: nat { i <= j /\ j < n }) -> f j);
    rewrite (bigstar #u1 i n (fun (j: nat { i <= j /\ j < n }) -> f j))
         as (bigstar #u1 m n (fun (j: nat { m <= j /\ j < n }) -> f j));
    bigstar_uneta () #u1 #m #n;
  } else {
    bigstar_pop #u1 #m #i;
    bigstar_compose #u1 (m+1) n (fun (j: nat { (m+1) <= j /\ j < n }) -> f j) i;
    bigstar_push #u1 m n (fun (j: nat { m <= j /\ j < n }) -> f j);
    bigstar_uneta () #u1 #m #n;
  }
}

ghost fn bigstar_zs_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < m} -> slprop))
  requires bigstar #u1 m m f
  ensures  emp
{
  rewrite bigstar #u1 m m f as emp;
}

ghost fn __bigstar_zs_intro
  (# u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < m} -> slprop))
  requires emp
  ensures  bigstar #u1 m m f
{
  rewrite emp as bigstar #u1 m m f;
}
let bigstar_zs_intro #u = __bigstar_zs_intro #u

ghost fn bigstar_single_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires bigstar #u1 m (m+1) f
  ensures  f m
{
  bigstar_pop #u1;
  bigstar_zs_elim #u1;
}

ghost fn __bigstar_single_intro
  (# u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires f m
  ensures  bigstar #u1 m (m+1) f
{
  bigstar_zs_intro #u1 (m+1) f;
  bigstar_push #u1 m (m+1) f;
}
let bigstar_single_intro #u = __bigstar_single_intro #u

ghost fn rec bigstar_emp_elim
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  requires bigstar #u1 m n (fun _ -> emp)
  ensures  emp
  decreases (n-m)
{
  if (m = n) {
    rewrite bigstar #u1 m n (fun _ -> emp) as emp;
  } else {
    bigstar_pop #u1;
    bigstar_emp_elim #u1 #(m+1) #n;
  }
}

ghost
fn rec __bigstar_emp_intro
  (#u1 : int)
  (m : nat)
  (n : nat {m <= n})
  requires emp
  ensures  bigstar #u1 m n (fun _ -> emp)
  decreases (n-m)
{
  if (m = n) {
    rewrite emp as bigstar #u1 m n (fun _ -> emp);
  } else {
    __bigstar_emp_intro #u1 (m+1) n;
    bigstar_push #u1 m n (fun _ -> emp);
  }
}
let bigstar_emp_intro #u1 m n = __bigstar_emp_intro #u1 m n



// As we work with bigstar, we need to make sure the domain of f,g remains
// the same, since it appears as an argument to bigstar. So, this function
// is further parametrized by lo,hi the bounds of the domain of f,g
ghost
fn rec bigstar_map'
  (#u1 #u2 : int)
  (#lo : nat)
  (#hi : nat{lo <= hi})
  (#m : nat{lo <= m})
  (#n : nat {m <= n /\ n <= hi})
  (#f: (i: nat{lo <= i /\ i < hi} -> slprop))
  (#g: (i: nat{lo <= i /\ i < hi} -> slprop))
  (stt: ((i: nat{lo <= i /\ i < hi}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
  requires  bigstar #u1 m n (fun (i: nat { m <= i /\ i < n }) -> f i)
  ensures   bigstar #u2 m n (fun (i: nat { m <= i /\ i < n }) -> g i)
  decreases (n-m)
{
  if (m = n) {
    rewrite bigstar #u1 m n f as emp;
    rewrite emp as bigstar #u2 m n g;
  } else {
    bigstar_extract m n (fun (i: nat { m <= i /\ i < n }) -> f i) m;
    stt m;
    bigstar_map' #u1 #u2 #lo #hi #(m+1) #n #f #g stt;
    rewrite bigstar #u1 m m f
         as bigstar #u2 m m g;
    bigstar_compose m n (fun (i: nat { m <= i /\ i < n }) -> g i) m;
  }
}

ghost
fn __bigstar_map
  (#u1 : int)
  (#u2 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (#g: (i: nat{m <= i /\ i < n} -> slprop))
  (stt: ((i: nat{m <= i /\ i < n}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
  requires bigstar #u1 m n f
  ensures  bigstar #u2 m n g
{
  bigstar_eta ();
  bigstar_map' #u1 #u2 #m #n #m #n #f #g stt;
  bigstar_uneta ();
}
let bigstar_map #u1 #u2 = __bigstar_map #u1 #u2

let lemma_eq
  (#u2 : int)
  (m0 : nat)
  (n0 : nat {m0 <= n0})
  (m1 : nat)
  (n1 : nat {m1 < n1})
  (f: (i: nat{m0 <= i /\ i < n0} -> j: nat{m1 <= j /\ j < n1} -> slprop))
  (i: nat{m0 <= i /\ i < n0})
  :
  Lemma (f i m1 ** bigstar #u2 (m1 + 1) n1 (fun (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j) == f i m1 ** bigstar #u2 (m1 + 1) n1 (fun (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j))
  = ()

ghost fn rec bigstar_commute
  (#u1 #u2 : int)
  (m0 : nat)
  (n0 : nat {m0 <= n0})
  (m1 : nat)
  (n1 : nat {m1 <= n1})
  (f: (i: nat{m0 <= i /\ i < n0} -> j: nat{m1 <= j /\ j < n1} -> slprop))
  requires bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> f i j))
  ensures  bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j))
  decreases (n1 - m1)
{
  if (m1 = n1) {
    bigstar_map #u1 #u1 #m0 #n0
      #(fun (i: nat{m0 <= i /\ i < n0}) -> bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> f i j)) #_
      (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar_zs_elim #_ #_ #_);
    bigstar_emp_elim #u1 #m0 #n0;
    bigstar_zs_intro #u2 m1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j));
    rewrite (bigstar #u2 m1 m1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j)))
        as  bigstar #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j));
  } else {
    bigstar_map #u1 #u1 #m0 #n0 (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar_pop #u2 #m1 #n1 #(fun (j: nat{m1 <= j /\ j < n1}) -> f i j));
    bigstar_star #u1 m0 n0
      (fun (i: nat{m0 <= i /\ i < n0}) -> f i m1)
      (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar #u2 (m1 + 1) n1 (fun (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j))
      (fun (i: nat{m0 <= i /\ i < n0}) -> f i m1 ** bigstar #u2 (m1 + 1) n1 (fun (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j))
      (fun (i: nat{m0 <= i /\ i < n0}) -> lemma_eq #u2 m0 n0 m1 n1 f i);
    rewrite (bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i m1 ** bigstar #u2 (m1 + 1) n1 (fun (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j)))
        as  (bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i m1) **
              bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> bigstar #u2 (m1 + 1) n1 (fun (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j)));
    bigstar_commute #u1 #u2 m0 n0 (m1 + 1) n1 (fun (i: nat{m0 <= i /\ i < n0}) (j: nat{(m1 + 1) <= j /\ j < n1}) -> f i j);
    bigstar_push #u2 m1 n1 (fun (j: nat{m1 <= j /\ j < n1}) -> bigstar #u1 m0 n0 (fun (i: nat{m0 <= i /\ i < n0}) -> f i j));
  }
}

let comb (f g : 'a -> slprop) : 'a -> slprop =
  fun x -> f x ** g x

ghost
fn rec bigstar_zip'
    (#u1 #u2 #u3 : int)
    (#lo #hi : nat)
    (m : nat {lo <= m})
    (n : nat {m <= n /\ n <= hi})
    (f: (i: nat{lo <= i /\ i < hi} -> slprop))
    (g: (i: nat{lo <= i /\ i < hi} -> slprop))
  requires  bigstar #u1 m n (fun (i: nat { m <= i /\ i < n }) -> f i) ** bigstar #u2 m n (fun (i: nat { m <= i /\ i < n }) -> g i)
  ensures   bigstar #u3 m n (fun (i: nat { m <= i /\ i < n }) -> comb f g i)
  decreases (n-m)
{
  if (n = m) {
    rewrite bigstar #u1 m n f as emp;
    rewrite bigstar #u2 m n g as emp;
    rewrite emp as bigstar #u3 m n (comb f g);
    ()
  } else {
    bigstar_pop #u1;
    bigstar_pop #u2;
    bigstar_zip' #_ #_ #u3 #lo #hi (m+1) n f g;
    fold (comb f g m);
    bigstar_push #u3 m n (fun (i: nat { m <= i /\ i < n }) -> comb f g i);
  }
}

ghost
fn __bigstar_zip
    (#u1 #u2 #u3 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> slprop))
    (g: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f ** bigstar #u2 m n g
  ensures  bigstar #u3 m n (fun (i: nat { m <= i /\ i < n }) -> f i ** g i)
{
  bigstar_eta () #u1;
  bigstar_eta () #u2;
  bigstar_zip' #u1 #u2 #u3 #m #n m n f g;
}
let bigstar_zip #u1 #u2 #u3 = __bigstar_zip #u1 #u2 #u3

ghost
fn rec bigstar_unzip'
    (#u1 #u2 #u3 : int)
    (#lo #hi : nat)
    (m : nat {lo <= m})
    (n : nat {m <= n /\ n <= hi})
    (f: (i: nat{lo <= i /\ i < hi} -> slprop))
    (g: (i: nat{lo <= i /\ i < hi} -> slprop))
  requires  bigstar #u3 m n (comb f g)
  ensures   bigstar #u1 m n f ** bigstar #u2 m n g
  decreases (n-m)
{
  if (n = m) {
    rewrite bigstar #u3 m n (comb f g) as emp;
    rewrite emp as bigstar #u1 m n f;
    rewrite emp as bigstar #u2 m n g;
    ()
  } else {
    bigstar_pop #u3;
    bigstar_uneta ();
    bigstar_unzip' #u1 #u2 #u3 #lo #hi (m+1) n f g;
    unfold (comb f g m);
    bigstar_eta () #u1;
    bigstar_push #u1 m n f;
    bigstar_eta () #u2;
    bigstar_push #u2 m n g;
  }
}

ghost
fn __bigstar_unzip
    (#u1 #u2 #u3 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> slprop))
    (g: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u3 m n (fun i -> f i ** g i)
  ensures  bigstar #u1 m n f ** bigstar #u2 m n g
{
  bigstar_unzip' #u1 #u2 #u3 #m #n m n f g;
}
let bigstar_unzip #u1 #u2 #u3 = __bigstar_unzip #u1 #u2 #u3

// Bigstar cond

ghost fn with_pure
  (#p #q: slprop)
  (b: prop)
  (op: unit -> stt_ghost unit emp_inames (p ** pure b) (fun _ -> q))
  (#_: squash b)
  requires p
  ensures  q
{
  op ();
}

ghost fn bigstar_if_elim
  (#u1 : int)
  (#m: nat)
  (#n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  requires bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp)
  ensures  p x
{
  bigstar_extract m n _ x;
  bigstar_map #u1 #u1 #m #x
    (fun (i: nat { m <= i /\ i < x }) -> with_pure (i = x == false) (fun _ -> elim_cond_false (i = x) (p (i <: nat)) emp));

  bigstar_emp_elim #_ #m #x;

  bigstar_map #u1 #u1 #(x + 1) #n
    (fun (i: nat { (x + 1) <= i /\ i < n }) -> with_pure (i = x == false) (fun _ -> elim_cond_false (i = x) (p (i <: nat)) emp));

  bigstar_emp_elim #_ #(x + 1) #n;
  elim_cond_true (x = x) (p x) emp;
}

ghost fn cond_rewrite_bool
  (b1 b2: bool)
  (#p #q: slprop)
  (#_: squash (b1 == b2))
  requires cond b1 p q
  ensures  cond b2 p q
{
  ()
}

ghost fn cond_rewrite_bool_2
  (b1 b2: bool)
  (#p #q: slprop)
  (#pf: unit -> squash (b1 <==> b2))
  requires cond b1 p q
  ensures  cond b2 p q
{
  pf ();
  ()
}

ghost
fn __bigstar_if_intro
  (#u1 : int)
  (m: nat)
  (n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  requires p x
  ensures  bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp)
{
  intro_cond_true (p x) emp;
  bigstar_emp_intro #u1 m x;
  bigstar_map #u1 #u1 #m #x #(fun _ -> emp) #(fun (i: nat { m <= i /\ i < x }) -> _ i) 
    (fun (i: nat { m <= i /\ i < x }) -> intro_cond_false (p (i <: nat)) emp);
  bigstar_map #u1 #u1 #m #x #(fun _ -> emp) #(fun (i: nat { m <= i /\ i < x }) -> _ i) 
    (fun (i: nat { m <= i /\ i < x }) -> cond_rewrite_bool false (i = x) #(p (i <: nat)) #emp);
  bigstar_emp_intro #u1 (x + 1) n;
  bigstar_map #u1 #u1 #(x + 1) #n #(fun _ -> emp) #(fun (i: nat { (x + 1) <= i /\ i < n }) -> _ i) 
    (fun (i: nat { (x + 1) <= i /\ i < n }) -> intro_cond_false (p (i <: nat)) emp);
  bigstar_map #u1 #u1 #(x + 1) #n #(fun _ -> emp) #(fun (i: nat { (x + 1) <= i /\ i < n }) -> _ i) 
    (fun (i: nat { (x + 1) <= i /\ i < n }) -> cond_rewrite_bool false (i = x) #(p (i <: nat)) #emp);
  bigstar_compose #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp) x;
}

let bigstar_if_intro
  (#[exact (`0)]u1 : int)
  (m: nat)
  (n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  : stt_ghost unit emp_inames
      (requires p x)
      (ensures  fun _ -> bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp))
  = __bigstar_if_intro #u1 m n x p

// #push-options "--print_implicits --print_bound_var_types"

ghost fn bigstar_permute
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (p: permutation (i: erased nat{m <= i /\ i < n}))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n (fun i -> f (p.f i))
{
  // bigstar 0 n (fun j -> f j)
  bigstar_map #u1 #u1 #m #n (fun j -> bigstar_if_intro #u1 m n (p.g j) (fun _ -> f j));
  // bigstar 0 n (fun j -> bigstar 0 n (fun i -> cond (i == p.g j) (f j) emp)
  bigstar_commute #u1 #u1 m n m n (fun j i -> cond (i = (p.g j)) (f j) emp);
  // bigstar 0 n (fun i -> bigstar 0 n (fun j -> cond (i == p.g j) (f j) emp)
  bigstar_map #u1 #u1 #m #n (fun (j:nat { m <= j /\ j < n }) ->
    bigstar_map #u1 #u1 #m #n (fun (i:nat { m <= i /\ i < n }) ->
      // let tmp: squash ((j = p.g i) == (i = p.f j)) = (equality_translate #m #n j (p.g i) i (p.f j) (p.proof j i)) in
      //  p.proof j i;
      cond_rewrite_bool (j = (p.g i)) (i = (p.f j)) #(f i) #emp
        // #tmp
        #(assume ((j = (p.g i)) <==> (i = (p.f j)))) // <- TODO
    ));
  // bigstar 0 n (fun i -> bigstar 0 n (fun j -> cond (j == p.f i) (f j) emp)
  bigstar_map #u1 #u1 #m #n (fun j -> bigstar_if_elim #u1 #m #n (p.f j) f);
  // bigstar 0 n (fun i -> f (p.f i))
}
