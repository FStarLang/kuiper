module Pulse.Lib.BigStar

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Mul
open FStar.FunctionalExtensionality
module SZ = FStar.SizeT

let star_aci () :
    squash (
      (forall (a b : vprop). {:pattern (a ** b)} a ** b == b ** a) /\
      (forall (a : vprop). {:pattern (a ** emp)} a ** emp == a) /\
      (forall (a b c : vprop). {:pattern (a ** b ** c)} a ** (b ** c) == (a ** b) ** c)) =
  introduce forall (a b : vprop). a ** b == b ** a with elim_vprop_equiv (vprop_equiv_comm a b);
  introduce forall (a : vprop). a ** emp == a with elim_vprop_equiv (vprop_equiv_unit a);
  introduce forall (a b c : vprop). a ** (b ** c) == (a ** b) ** c with elim_vprop_equiv (vprop_equiv_assoc a b c)

let rec bigstar_split (m : nat) (n : nat {m <= n}) f (i : nat { m <= i /\ i <= n }) :
    Lemma (ensures bigstar m n f == bigstar m i f ** bigstar i n f) (decreases n - m) =
  star_aci ();
  if m = i then () else bigstar_split (m+1) n f i

let rec bigstar_star (m : nat) (n : nat {m <= n}) f g h
    (heq : (i:nat { m <= i /\ i < n }) -> squash (f i ** g i == h i)) :
    Pure
      (squash (bigstar m n f ** bigstar m n g == bigstar m n h))
      (requires True) //forall i. f i ** g i == h i)
      (ensures fun _ -> True)
      (decreases n - m) =
  star_aci ();
  if m = n then () else (bigstar_star (m+1) n f g h heq; heq m)

let rec bigstar_congr (m : nat) (n : nat { m <= n }) (m' : nat) (n' : nat { m' <= n' /\ n' - m' == n - m })
    (f : (i:nat { m <= i /\ i < n }) -> vprop) (f' : (i:nat { m' <= i /\ i < n' }) -> vprop) 
    (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m'+i))))
    :
    Pure
      (squash (bigstar m n f == bigstar m' n' f'))
      (requires True)
      (ensures fun _ -> True)
      (decreases n-m)
       =
  if m = n then () else begin
    bigstar_congr (m+1) n (m'+1) n' f f' (fun i -> h (i+1));
    h 0
  end

```pulse
ghost
fn bigstar_rw_congr
   (m : nat) (n : nat { m <= n })
   (f : (i:nat { m <= i /\ i < n }) -> vprop)
   (f' : (i:nat { m <= i /\ i < n }) -> vprop) 
   (h : ((i:nat{m <= i /\ i < n}) -> squash (f i == f' i)))
  requires bigstar m n f
  ensures  bigstar m n f'
{
  let h' :
    ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m+i)))
    = (fun (i:nat{i < n-m}) -> h (m+i));

  bigstar_congr m n m n f f' h';
  rewrite bigstar m n f as bigstar m n f';
  ();
}
```


```pulse
ghost fn rw (a b : vprop) requires a ** pure (a == b) ensures b {
  rewrite a as b
}
```

```pulse
ghost fn bigstar_extract
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (i : nat { m <= i /\ i < n })
  requires bigstar m n f
  returns _:unit
  ensures bigstar m i f ** f i ** bigstar (i+1) n f
{
  bigstar_split m n f i;
  rw (bigstar m n f) (bigstar m i f ** bigstar i n f);
  rw (bigstar i n f) (f i ** bigstar (i+1) n f);
}
```

```pulse
ghost fn bigstar_compose
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (i : nat { m <= i /\ i < n })
  requires bigstar m i f ** f i ** bigstar (i+1) n f
  returns _:unit
  ensures bigstar m n f
{
  bigstar_split m n f i;
  rw (f i ** bigstar (i+1) n f) (bigstar i n f);
  rw (bigstar m i f ** bigstar i n f) (bigstar m n f);
}
```

```pulse
ghost fn bigstar_zip
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
  requires bigstar m n f ** bigstar m n g
  ensures  bigstar m n (fun i -> f i ** g i)
{
  admit();
}
```

```pulse
ghost fn bigstar_unzip
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
  requires bigstar m n (fun i -> f i ** g i)
  ensures  bigstar m n f ** bigstar m n g
{
  admit();
}
```
