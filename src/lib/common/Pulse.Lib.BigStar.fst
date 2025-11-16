module Pulse.Lib.BigStar

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Mul
open FStar.FunctionalExtensionality
open Pulse.Lib.PartitionRange
module T = FStar.Tactics.V2

let narrow (m:nat) (n:nat{ m < n }) (f: (i:nat { m <= i /\ i < n }) -> slprop)
: (i:nat { (m + 1) <= i /\ i < n }) -> slprop
= fun i -> f i

let rec bigstar
  (#uid : int)
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
: Tot slprop (decreases n - m) =
  if m = n then emp else f m ** bigstar #uid (m+1) n (narrow m n f) //(fun (i: nat { (m+1) <= i /\ i < n }) -> f i)


let rec bigstar_sendable_
  (uid: int)
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
  (vis: loc_id -> 'a)
  (sa : (i:_ -> is_send_across vis (f i)))
: Tot (is_send_across vis (bigstar #uid m n f)) (decreases (n - m))
= if m = n
  then FStar.Tactics.Typeclasses.solve #(is_send_across vis emp)
  else let _ = bigstar_sendable_ uid (m + 1) n (narrow m n f) vis sa in
      FStar.Tactics.Typeclasses.solve
        #(is_send_across vis
           (f m ** bigstar #uid (m+1) n (narrow m n f)))

instance bigstar_sendable
  (uid: int)
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
  (vis: loc_id -> 'a)
  (sa : (i:_ -> is_send_across vis (f i)))
: is_send_across vis (bigstar #uid m n f)
= bigstar_sendable_ uid m n f vis sa

let bigstar_defn (#uid : int) (m : nat) (n : nat {m <= n}) (f : (i:nat { m <= i /\ i < n } -> slprop)) :
  Lemma (ensures bigstar #uid m n f == (if m = n then emp else f m ** bigstar #uid (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)))
  = assert (bigstar #uid m n f == (if m = n then emp else f m ** bigstar #uid (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i)))
        by (T.trefl())

ghost
fn bigstar_pop
  (#u1 : int)
  (#m : nat)
  (#n : nat {m < n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f
  ensures  f m ** bigstar #u1 (m + 1) n (fun (i: nat {(m + 1) <= i /\ i < n}) -> f i)
{
  unfold (bigstar #u1 m n f);
  rewrite (if m = n then emp else f m ** bigstar #u1 (m+1) n (narrow m n f)) //fun (i: nat { (m+1) <= i /\ i < n }) -> f i))
      as  (                       f m ** bigstar #u1 (m+1) n (fun (i: nat { (m+1) <= i /\ i < n }) -> f i));
}

ghost
fn bigstar_push
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
    (f : (i:nat { m <= i /\ i < n }) -> slprop)
    (f' : (i:nat { m' <= i /\ i < n' }) -> slprop)
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
fn bigstar_rewrite_ext
    (#u1 #u2: int)
    (m : nat) (n : nat { m <= n }) (k:nat { n <= k })
    (f : (i:nat { m <= i /\ i < k }) -> slprop)
    (f' : (i:nat { m <= i /\ i < n }) -> slprop)
    (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m+i))))
requires bigstar #u1 m n f
ensures bigstar #u2 m n f'
{
  bigstar_congr #u1 #u2 m n m n f f' h;
  rewrite bigstar #u1 m n f as bigstar #u2 m n f';
}

ghost
fn bigstar_rewrite_ext_l
    (#u1 #u2: int)
    (k:nat) (m : nat { k <= m }) (n : nat { m <= n })
    (f : (i:nat { k <= i /\ i < n }) -> slprop)
    (f' : (i:nat { m <= i /\ i < n }) -> slprop)
    (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m+i))))
requires bigstar #u1 m n f
ensures bigstar #u2 m n f'
{
  bigstar_congr #u1 #u2 m n m n f f' h;
  rewrite bigstar #u1 m n f as bigstar #u2 m n f';
}

let bigstar_eq (#u1 #u2: int) (m : nat) (n : nat {m <= n}) (f g : (i:nat { m <= i /\ i < n }) -> slprop)
  : Lemma (requires (forall i. m <= i /\ i < n ==> f i == g i))
          (ensures  bigstar #u1 m n f == bigstar #u2 m n g) = bigstar_congr #u1 #u2 m n m n f g (fun i -> ())

let rec bigstar_ext u1 u2 (m:nat) (n:nat{m<=n}) (f g: ((i:nat{m<=i /\ i<n}) -> slprop))
: Lemma
  (requires FStar.FunctionalExtensionality.feq f g)
  (ensures bigstar #u1 m n f == bigstar #u2 m n g)
  (decreases n - m)
= if m = n then ()
  else bigstar_ext u1 u2 (m + 1) n (narrow m n f) (narrow m n g)

let bigstar_extensionality_lem
  (u1 u2 : int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  (h: ((i: nat{m <= i /\ i < n}) -> slprop_equiv (f i) (g i)))
: Lemma (slprop_equiv (bigstar #u1 m n f) (bigstar #u2 m n g))
= introduce forall i. f i == g i
  with elim_slprop_equiv (h i);
  bigstar_ext u1 u2 m n f g;
  FStar.Squash.return_squash (slprop_equiv_refl (bigstar #u1 m n f))


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
{
  bigstar_congr #u1 #u1 m n m n f g (fun j -> h (m+j));
  ();
}

ghost
fn bigstar_ext'
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop))
  (g: (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f
  requires pure (forall (i: nat{m <= i /\ i < n}). f i == g i)
  ensures  bigstar #u1 m n g
{
  bigstar_extensionality #u1 m n f g (fun _ -> ())
}

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

ghost
fn rec bigstar_extract
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
    assert rewrites_to m i;
    rewrite (emp ** f m ** bigstar #u1 (m+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j))
         as (bigstar #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j) ** f i ** bigstar #u1 (i+1) n (fun (j: nat { (i+1) <= j /\ j < n }) -> f j));
  } else {
    bigstar_extract #u1 (m+1) n (fun (j: nat { (m+1) <= j /\ j < n }) -> f j) i;
    bigstar_push #u1 m i (fun (j: nat { m <= j /\ j < i }) -> f j);
  }
}

ghost
fn rec bigstar_compose
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

ghost
fn bigstar_zs_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < m} -> slprop))
  requires bigstar #u1 m m f
  ensures  emp
{
  rewrite bigstar #u1 m m f as emp;
}

ghost
fn bigstar_zs_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < m} -> slprop))
  requires emp
  ensures  bigstar #u1 m m f
{
  rewrite emp as bigstar #u1 m m f;
}

ghost
fn bigstar_single_elim
  (#u1 : int)
  (#m : nat)
  (#f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires bigstar #u1 m (m+1) f
  ensures  f m
{
  bigstar_pop #u1;
  bigstar_zs_elim #u1;
}

ghost
fn bigstar_single_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (f: (i: nat{m <= i /\ i < (m+1)} -> slprop))
  requires f m
  ensures  bigstar #u1 m (m+1) f
{
  bigstar_zs_intro #u1 (m+1) f;
  bigstar_push #u1 m (m+1) f;
}

ghost
fn rec bigstar_emp_elim'
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (f : (i: nat{m <= i /\ i < n} -> slprop))
  requires bigstar #u1 m n f ** pure (forall x. f x == emp)
  ensures  emp
  decreases (n-m)
{
  if (m = n) {
    rewrite bigstar #u1 m n (fun _ -> emp) as emp;
  } else {
    bigstar_pop #u1;
    bigstar_emp_elim' #u1 #(m+1) #n (fun x -> f x);
    rewrite f m as emp;
  }
}

ghost
fn rec bigstar_emp_elim
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  requires bigstar #u1 m n (fun _ -> emp)
  ensures  emp
  decreases (n-m)
{
  bigstar_emp_elim' #u1 #m #n (fun _ -> emp);
}

ghost
fn rec bigstar_emp_intro
  (#[exact (`0)] u1 : int)
  (m : nat)
  (n : nat {m <= n})
  requires emp
  ensures  bigstar #u1 m n (fun _ -> emp)
  decreases (n-m)
{
  if (m = n) {
    rewrite emp as bigstar #u1 m n (fun _ -> emp);
  } else {
    bigstar_emp_intro #u1 (m+1) n;
    bigstar_push #u1 m n (fun _ -> emp);
  }
}



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
    rewrite bigstar #u1 m n (fun i -> f i) as emp;
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
{
  bigstar_eta ();
  bigstar_map' #u1 #u2 #m #n #m #n #f #g stt;
  bigstar_uneta ();
}

ghost
fn bigstar_map2
  (#u1 #u2 : int)
  (#m : nat)  (#n : nat {m <= n})
  (#m' : nat) (#n' : nat {m' <= n'})
  (#f #g : (i:nat{m <= i /\ i < n}) -> (j:nat{m' <= j /\ j < n'}) -> slprop)
  (stt: ((i: nat{m <= i /\ i < n}) ->
            (j: nat{m' <= j /\ j < n'}) -> stt_ghost unit emp_inames
            (f i j)
            (fun _ -> g i j)))
  requires bigstar #u1 m n (fun x -> bigstar #u2 m' n' (fun y -> f x y))
  ensures  bigstar #u1 m n (fun x -> bigstar #u2 m' n' (fun y -> g x y))
{
  bigstar_map #u1 #u1 #m #n (fun x -> bigstar_map #u2 #u2 #m' #n' (stt x));
}

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

ghost
fn rec bigstar_commute
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
    rewrite bigstar #u1 m n (fun i -> f i) as emp;
    rewrite bigstar #u2 m n (fun i -> g i) as emp;
    rewrite emp as bigstar #u3 m n (comb f g);
    ()
  } else {
    bigstar_pop #u1;
    bigstar_pop #u2;
    bigstar_zip' #u1 #_ #u3 #lo #hi (m+1) n f g;
    fold (comb f g m);
    bigstar_push #u3 m n (fun (i: nat { m <= i /\ i < n }) -> comb f g i);
  }
}

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
{
  bigstar_eta () #u1;
  bigstar_eta () #u2;
  bigstar_zip' #u1 #u2 #u3 #m #n m n f g;
}

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
{
  bigstar_unzip' #u1 #u2 #u3 #m #n m n f g;
}

// Bigstar cond

ghost
fn with_pure
  (#p #q: slprop)
  (b: prop)
  (op: unit -> stt_ghost unit emp_inames (p ** pure b) (fun _ -> q))
  (#_: squash b)
  requires p
  ensures  q
{
  op ();
}

ghost
fn bigstar_if_elim
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

ghost
fn cond_rewrite_bool
  (b1 b2: bool)
  (#p #q: slprop)
  (#_: squash (b1 == b2))
  requires cond b1 p q
  ensures  cond b2 p q
{
  rewrite each b1 as b2;
}

ghost
fn cond_rewrite_bool_2
  (b1 b2: bool)
  (#p #q: slprop)
  (#pf: unit -> squash (b1 <==> b2))
  requires cond b1 p q
  ensures  cond b2 p q
{
  pf ();
  rewrite each b1 as b2;
}

ghost
fn bigstar_if_intro
  (#[exact (`0)]u1 : int)
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

ghost
fn bigstar_permute
  (#u1 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> slprop))
  (p: permutation (i: nat{m <= i /\ i < n}))
  requires bigstar #u1 m n f
  ensures  bigstar #u1 m n (fun i -> f (p.f i))
{
  bigstar_map #u1 #u1 #m #n (fun j -> bigstar_if_intro #u1 m n (p.g j) (fun _ -> f j));
  bigstar_commute #u1 #u1 m n m n (fun j i -> cond (i = (p.g j)) (f j) emp);
  ghost
  fn aux (i : nat { m <= i /\ i < n }) (j : nat { m <= j /\ j < n })
    requires cond (i = (p.g j)) (f j) emp
    ensures  cond (j = (p.f i)) (f j) emp
  {
    p.proof i j;
    assert (pure (i == p.g j <==> j == p.f i));
    assert (pure ((i = p.g j) = (j = p.f i)));
    cond_rewrite_bool (i = (p.g j)) (j = (p.f i)) #_ #_ #();
    ();
  };
  bigstar_map2 #u1 #u1 #m #n #m #n aux;
  bigstar_map #u1 #u1 #m #n (fun j -> bigstar_if_elim #u1 #m #n (p.f j) f);
}

ghost
fn bigstar_cut
  (#u1 : int)
  (#n1 : nat)
  (#n2 : nat{n1 <= n2})
  (n3 : nat{n1 <= n3 /\ n3 <= n2})
  (#f: (i: nat{n1 <= i /\ i < n2} -> slprop))
  requires bigstar #u1 n1 n2 f
  ensures  bigstar #u1 n1 n3 f ** bigstar #u1 n3 n2 f
{
  bigstar_split #u1 n1 n2 f n3;
  rewrite bigstar #u1 n1 n2 f
       as bigstar #u1 n1 n3 f ** bigstar #u1 n3 n2 f;
}

ghost
fn bigstar_paste
  (#u1 : int)
  (#n1 : nat)
  (#n2 : nat{n1 <= n2})
  (n3 : nat{n1 <= n3 /\ n3 <= n2})
  (#f: (i: nat{n1 <= i /\ i < n2} -> slprop))
  requires bigstar #u1 n1 n3 f ** bigstar #u1 n3 n2 f
  ensures  bigstar #u1 n1 n2 f
{
  bigstar_split #u1 n1 n2 f n3;
  rewrite bigstar #u1 n1 n3 f ** bigstar #u1 n3 n2 f
       as bigstar #u1 n1 n2 f;
}

ghost
fn bigstar_shift
  (#u1 : int)
  (#n1 : nat)
  (#n2 : nat{n1 <= n2})
  (s : int{s + n1 >= 0})
  (#f: (i: nat{n1 <= i /\ i < n2} -> slprop))
  requires bigstar #u1 n1 n2 f
  ensures  bigstar #u1 (n1 + s) (n2 + s) (fun x -> f (x - s))
{
  bigstar_congr #u1 #u1 n1 n2 (n1 + s) (n2 + s) f (fun x -> f (x - s))
    (fun (i:nat{i < n2-n1}) -> assert (f (n1+i) == f (((n1+s) + i) - s)));
  rewrite bigstar #u1 n1 n2 f
       as bigstar #u1 (n1 + s) (n2 + s) (fun x -> f (x - s));
}

ghost
fn rec bigstar_flatten
  (#u1 #u2 : int)
  (#n1 : nat)
  (#n2 : nat)
  (#f: (i: nat{0 <= i /\ i < n1} -> j: nat{0 <= j /\ j < n2} -> slprop))
  requires bigstar #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i))
  ensures  bigstar #u1 0 (n1 * n2) (fun i -> f (i / n2) (i % n2))
  decreases n1
{
  if (n1 = 0) {
    rewrite bigstar #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i)) as emp;
    rewrite emp as bigstar #u1 0 (n1 * n2) (fun i -> f (i / n2) (i % n2));
  } else {
    bigstar_extract 0 n1 (fun i -> bigstar #u2 0 n2 (f i)) (n1-1);
    rewrite
      bigstar #u1 ((n1-1)+1) n1 (fun i -> bigstar #u2 0 n2 (f i))
    as emp;
    bigstar_flatten #u1 #u2 #(n1-1) #n2 #(fun x -> f x);
    bigstar_shift #u2 #0 #n2 ((n1-1)*n2)
      #(fun i -> f (n1-1) i);
    rewrite each (0 + ((n1-1) `op_Multiply` n2)) as ((n1-1)*n2);
    rewrite each (n2 + ((n1-1) `op_Multiply` n2)) as (n1*n2);
    bigstar_ext' ((n1-1)*n2) (n1*n2)
      (fun i -> f (n1-1) (i - (n1-1)*n2))
      (fun i -> f (i/n2) (i%n2));
    // retag
    rewrite bigstar #u2 ((n1-1)*n2) (n1*n2) (fun i -> f (i/n2) (i%n2))
         as bigstar #u1 ((n1-1)*n2) (n1*n2) (fun i -> f (i/n2) (i%n2));
    let f' = (fun (ij : nat {0 <= ij /\ ij < n1 * n2}) -> f (ij / n2) (ij % n2));
    bigstar_ext' 0 ((n1 - 1) * n2) (fun i -> f (i / n2) (i % n2)) f';
    bigstar_paste #u1 #0 #(n1*n2) ((n1-1)*n2) #f';
  }
}

ghost
fn rec bigstar_unflatten
  (#u1 #u2 : int)
  (#n1 : nat)
  (#n2 : nat)
  (#f: (i: nat{0 <= i /\ i < n1} -> j: nat{0 <= j /\ j < n2} -> slprop))
  requires bigstar #u1 0 (n1 * n2) (fun i -> f (i / n2) (i % n2))
  ensures  bigstar #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i))
  decreases n1
{
  if (n1 = 0) {
    rewrite bigstar #u1 0 (n1 * n2) (fun i -> f (i / n2) (i % n2)) as emp;
    rewrite emp as bigstar #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i));
  }
  else {
    rewrite each (n1 * n2) as (n2 + ((n1 - 1) * n2));
    bigstar_cut #u1 #0 #(n2 + ((n1-1)*n2)) ((n1 - 1) * n2);
    rewrite each ((n2 + (n1 - 1) * n2)) as (n1 * n2);
    bigstar_rewrite_ext #u1 #u1 0 ((n1 - 1) * n2) (n1 * n2)
      (fun (i: nat{b2t (0 <= i) /\ b2t (i < n1 * n2)}) -> f (i / n2) (i % n2))
      (fun (i: nat{b2t (0 <= i) /\ b2t (i < (n1 - 1) * n2)}) -> f (i / n2) (i % n2))
      (fun _ -> ());
    bigstar_rewrite_ext_l #u1 #u1 0 ((n1 - 1) * n2) (n1 * n2)
      (fun (i: nat{b2t (0 <= i) /\ b2t (i < n1 * n2)}) -> f (i / n2) (i % n2))
      (fun (i: nat{b2t ((n1 - 1) * n2 <= i) /\ b2t (i < (n1 * n2))}) -> f (i / n2) (i % n2))
      (fun _ -> ());
    bigstar_unflatten #u1 #u2 #(n1 - 1) #n2 #(fun i -> f i);
    bigstar_extensionality #u1 ((n1-1)*n2) (n1*n2)
      (fun i -> f (i/n2) (i%n2))
      (fun i -> f (n1-1) (i - (n1-1)*n2))
      (fun _ -> ());
    bigstar_shift #u1 #((n1 - 1)*n2) #(n1 * n2) (-((n1 - 1)*n2));
    rewrite each (((n1 - 1) * n2 + - (n1 - 1) * n2)) as 0;
    rewrite each (n1 * n2 + - (n1 - 1) * n2) as n2;
    bigstar_extensionality #u1 0 n2
      (fun x -> f (n1 - 1) (x - - (n1 - 1) * n2 - (n1 - 1) * n2))
      (f (n1-1))
      (fun _ -> ());
    rewrite bigstar #u1 0 n2 (f (n1 - 1))
         as bigstar #u2 0 n2 (f (n1 - 1)); //retag
    rewrite emp as bigstar #u1 (n1 - 1 + 1) n1 (fun i -> bigstar #u2 0 n2 (f i));
    bigstar_compose #u1 0 n1 (fun i -> bigstar #u2 0 n2 (f i)) (n1 - 1);
  }
}

module Set = FStar.FiniteSet.Base
let rec bigstar_except
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop) )
  (s : Set.set nat)
: Tot slprop (decreases n - m)
= if m = n
  then emp
  else if Set.mem m s
  then bigstar_except #u1 (m+1) n f (Set.remove m s)
  else f m ** bigstar_except #u1 (m+1) n f s

let rec bigstar_except_equiv'
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f g: (i: nat{m <= i /\ i < n} -> slprop) )
  (_:squash (FStar.FunctionalExtensionality.feq f g))
: Lemma
  (ensures bigstar #u1 m n f == bigstar_except #u1 m n g Set.emptyset)
  (decreases n - m)
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  if m = n then ()
  else (
    assert (Set.remove m Set.emptyset `Set.equal` Set.emptyset);
    bigstar_except_equiv' #u1 (m+1) n (narrow m n f) g ()
  )

let bigstar_except_equiv
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop) )
: Lemma
  (ensures bigstar #u1 m n f == bigstar_except #u1 m n f Set.emptyset)
  (decreases n - m)
= bigstar_except_equiv' #u1 m n f f ()


let rec bigstar_except_equiv_emp
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop) )
  (s: Set.set nat { range m n `Set.subset` s })
: Lemma
  (ensures bigstar_except #u1 m n f s == emp)
  (decreases n - m)
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  if m = n then ()
  else bigstar_except_equiv_emp #u1 (m+1) n f (Set.remove m s)

#push-options "--ifuel 0 --z3rlimit_factor 16 --fuel 2"
#restart-solver
let rec bigstar_except_equiv_split
  (#u1: int)
  (m : nat)
  (n : nat {m <= n})
  (f: (i: nat{m <= i /\ i < n} -> slprop) )
  (s0: idx_set m n)
  (s1: Set.set nat)
: Lemma
  (requires Set.disjoint s0 s1)
  (ensures
    bigstar_except #u1 m n f s1 ==
    star_over_partition f s0 **
    bigstar_except #u1 m n f (Set.union s0 s1))
  (decreases (n - m))
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  let _ : squash (Set.disjoint s0 s1) = () in
  if m = n
  then (
    assert (Set.cardinality s0 = 0);
    assert (Set.equal (Set.union s0 s1) s1);
    slprop_equivs()
  )
  else (
    if Set.mem m s1
    then (
      calc (==) {
        bigstar_except #u1 m n f s1;
      (==) { slprop_equivs () }
        bigstar_except #u1 (m + 1) n f (Set.remove m s1);
      (==) {  bigstar_except_equiv_split #u1 (m + 1) n f s0 (Set.remove m s1) }
        star_over_partition f (s0 <: idx_set (m + 1) n) **
        bigstar_except #u1 (m + 1) n f (Set.union s0 (Set.remove m s1));
      (==) { star_over_partition_reindex m n f s0;
             assert (Set.remove m (Set.union s0 s1) `Set.equal` (Set.union s0 (Set.remove m s1)))
            }
        star_over_partition f s0 **
        bigstar_except #u1 m n f (Set.union s0 s1);
      }
    )
    else if Set.mem m s0
    then (
      let s0' : idx_set (m + 1) n = Set.remove m s0 in
      calc (==) {
        bigstar_except #u1 m n f s1;
      (==) { assert (Set.equal (Set.remove m s1) s1) }
        f m **
        bigstar_except #u1 (m + 1) n f s1;
      (==) {  bigstar_except_equiv_split #u1 (m + 1) n f s0' s1 }
        f m **
        (star_over_partition f s0' **
         bigstar_except #u1 (m + 1) n f (Set.union s0' s1));
      (==) {slprop_equivs ()}
        (f m ** star_over_partition #(m + 1) #n f s0') **
        bigstar_except #u1 (m + 1) n f (Set.union s0' s1);
      (==) { star_over_partition_reindex m n f s0' }
        (f m ** star_over_partition #m #n f s0') **
        bigstar_except #u1 (m + 1) n f (Set.union s0' s1);
      (==) { star_over_partition_split #m #n f (Set.singleton m) s0'; slprop_equivs () }
        star_over_partition #m #n f (Set.union (Set.singleton m) s0') **
        bigstar_except #u1 (m + 1) n f (Set.union s0' s1);
      (==) { assert (Set.equal s0 (Set.union (Set.singleton m) s0')) }
        star_over_partition #m #n f s0 **
        bigstar_except #u1 (m + 1) n f (Set.union s0' s1);
      (==) { assert (Set.equal (Set.remove m (Set.union s0 s1)) (Set.union s0' s1)); slprop_equivs () }
        star_over_partition #m #n f s0 **
        bigstar_except #u1 m n f (Set.union s0 s1);
      }
    )
    else (
      calc (==) {
        bigstar_except #u1 m n f s1;
      (==) { }
        f m **
        bigstar_except #u1 (m + 1) n f s1;
      (==) {  bigstar_except_equiv_split #u1 (m + 1) n f s0 s1 }
        f m **
        (star_over_partition #(m+1) #n f s0 **
         bigstar_except #u1 (m + 1) n f (Set.union s0 s1));
      (==) {slprop_equivs ()}
        star_over_partition #(m+1) #n f s0 **
        (f m ** bigstar_except #u1 (m + 1) n f (Set.union s0 s1));
      (==) {}
        star_over_partition #(m+1) #n f s0 **
        bigstar_except #u1 m n f (Set.union s0 s1);
      (==) { star_over_partition_reindex m n f s0}
        star_over_partition #m #n f s0 **
        bigstar_except #u1 m n f (Set.union s0 s1);
      }
    )
  )
#pop-options

let union_partitions_aux_split
    (#m #n #k : nat)
    (p:disjoint_partitions m n k)
    (from:nat)
    (mid:nat)
    (to:nat { from <= mid /\ mid <= to /\ to <= k})
: Lemma
    (union_partitions_aux p from to `Set.equal`
     (union_partitions_aux p from mid `Set.union` union_partitions_aux p mid to) /\
     Set.disjoint (union_partitions_aux p from mid) (union_partitions_aux p mid to))
= union_partitions_split p from mid to;
  union_partitions_disjoint p from mid to


let union_partitions_aux_step
    (#m #n #k : nat)
    (p:disjoint_partitions m n k)
    (from:nat)
    (to:nat { from <= to /\ to < k})
: Lemma
  (ensures
    union_partitions_aux p from (to + 1) `Set.equal`
   (select p to `Set.union`    union_partitions_aux p from to) /\
    select p to `Set.disjoint` union_partitions_aux p from to)
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  union_partitions_aux_split p from to (to + 1)

let star_of_part_i #n #k (parts:disjoint_partitions 0 n k) (f:idx 0 n -> slprop) (i:idx 0 k)
: slprop
= star_over_partition f (select parts i)

let rec bigstar_partition_equiv_except
  (#u1: int)
  (n : nat)
  (j : nat)
  (k : nat { j <= k })
  (f: (i:idx 0 n -> slprop))
  (parts: disjoint_partitions 0 n k)
: Lemma
  (ensures
    bigstar_except #u1 0 n f (union_partitions_aux parts 0 j) ==
    bigstar_except #u1 j k (star_of_part_i parts f) Set.emptyset)
  (decreases k - j)
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  if j = k
  then (
    bigstar_except_equiv_emp #u1 0 n f (union_partitions_aux parts 0 j)
  )
  else (
    union_partitions_aux_step parts 0 j;
    let _ : squash (Set.disjoint (select parts j) (union_partitions_aux parts 0 j)) = () in
    calc (==) {
      bigstar_except #u1 j k (star_of_part_i parts f) Set.emptyset;
    (==) { }
      star_over_partition f (select parts j) **
      bigstar_except #u1 (j+1) k (star_of_part_i parts f) Set.emptyset;
    (==) { bigstar_partition_equiv_except #u1 n (j+1) k f parts }
      star_over_partition f (select parts j) **
      bigstar_except #u1 0 n f (union_partitions_aux parts 0 (j+1));
    (==) { }
      star_over_partition f (select parts j) **
      bigstar_except #u1 0 n f (select parts j `Set.union` union_partitions_aux parts 0 j);
    (==) { bigstar_except_equiv_split #u1 0 n f (select parts j) (union_partitions_aux parts 0 j) }
      bigstar_except #u1 0 n f (union_partitions_aux parts 0 j);
    }
  )


let bigstar_partition_equiv
  (#u1: int)
  (n : nat)
  (k : nat)
  (f: (i:idx 0 n -> slprop))
  (parts: disjoint_partitions 0 n k)
: Lemma
  (ensures
    bigstar #u1 0 n f ==
    bigstar #u1 0 k (star_of_part_i parts f))
= calc (==) {
    bigstar #u1 0 n f;
  (==) { bigstar_except_equiv #u1 0 n f }
    bigstar_except #u1 0 n f Set.emptyset;
  (==) { bigstar_partition_equiv_except #u1 n 0 k f parts }
    bigstar_except #u1 0 k (star_of_part_i parts f) Set.emptyset;
  (==) { bigstar_except_equiv #u1 0 k (star_of_part_i parts f) }
    bigstar #u1 0 k (star_of_part_i parts f);
  }

let bigstar_partition_equiv_eta
  (#u1: int)
  (n : nat)
  (k : nat)
  (f: (i:idx 0 n -> slprop))
  (parts: disjoint_partitions 0 n k)
: Lemma
  (ensures
    bigstar #u1 0 n f ==
    bigstar #u1 0 k (fun i -> star_over_partition f (select parts i)))
= calc(==) {
    bigstar #u1 0 n f;
  (==) {  bigstar_partition_equiv #u1 n k f parts }
    bigstar #u1 0 k (star_of_part_i parts f);
  (==) {   bigstar_ext u1 u1 0 k (star_of_part_i parts f) (fun i -> star_over_partition f (select parts i)) }
    bigstar #u1 0 k (fun i -> star_over_partition f (select parts i));
}



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
{
  bigstar_partition_equiv_eta #0 n0 n1 f0 partition;
  rewrite (bigstar #0 0 n0 f0) as
          (bigstar 0 n1 (fun i -> star_over_partition f0 (select partition i)))
}

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
{
  bigstar_partition_equiv_eta #0 n0 n1 f0 partition;
  rewrite (bigstar 0 n1 (fun i -> star_over_partition f0 (select partition i))) as
          (bigstar #0 0 n0 f0);
}
