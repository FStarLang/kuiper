module Pulse.Lib.BigStar

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Mul
open FStar.FunctionalExtensionality
module SZ = FStar.SizeT

let rec bigstar
  (#uid : int)
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> vprop))
: Tot vprop (decreases n - m) =
  if m = n then emp else f m ** bigstar (m+1) n f

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
    (heq : (i:nat { m <= i /\ i < n }) -> squash (f i ** g i == h i))
: Lemma (ensures bigstar m n f ** bigstar m n g == bigstar m n h)
        (decreases n - m)
= star_aci ();
  if m = n then () else (bigstar_star (m+1) n f g h heq; heq m)

let rec bigstar_congr (m : nat) (n : nat { m <= n }) (m' : nat) (n' : nat { m' <= n' /\ n' - m' == n - m })
    (f : (i:nat { m <= i /\ i < n }) -> vprop) (f' : (i:nat { m' <= i /\ i < n' }) -> vprop)
    (h : ((i:nat{i < n-m}) -> squash (f (m+i) == f' (m'+i))))
: Lemma (ensures bigstar m n f == bigstar m' n' f')
        (decreases n-m)
= if m = n then () else begin
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
ghost fn bigstar_extract
    (#u1 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (i : nat { m <= i /\ i < n })
  requires bigstar #u1 m n f
  returns _:unit
  ensures bigstar #u1 m i f ** f i ** bigstar #u1 (i+1) n f
{
  bigstar_split m n f i;
  rewrite bigstar #u1 m n f
       as bigstar #u1 m i f ** bigstar #u1 i n f;
  rewrite bigstar #u1 i n f
       as f i ** bigstar #u1 (i+1) n f;
}
```

#set-options "--print_implicits"

```pulse
ghost fn bigstar_compose
    (#u1 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (i : nat { m <= i /\ i < n })
  requires bigstar #u1 m i f ** f i ** bigstar #u1 (i+1) n f
  returns _:unit
  ensures bigstar #u1 m n f
{
  bigstar_split m n f i;
  rewrite f i ** bigstar #u1 (i+1) n f
       as bigstar #u1 i n f;
  rewrite bigstar #u1 m i f ** bigstar #u1 i n f
       as bigstar #u1 m n f;
}
```

// As we work with bigstar, we need to make sure the domain of f,g remains
// the same, since it appears as an argument to bigstar. So, this function
// is further parametrized by lo,hi the bounds of the domain of f,g
```pulse
ghost
fn rec bigstar_map'
  (#u1 #u2 : int)
  (#lo : nat)
  (#hi : nat{lo <= hi})
  (#m : nat{lo <= m})
  (#n : nat {m <= n /\ n <= hi})
  (#f: (i: nat{lo <= i /\ i < hi} -> vprop))
  (#g: (i: nat{lo <= i /\ i < hi} -> vprop))
  (stt: ((i: nat{lo <= i /\ i < hi}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
  requires  bigstar #u1 m n f
  ensures   bigstar #u2 m n g
  decreases (n-m)
{
  if (m = n) {
    rewrite bigstar #u1 m n f as emp;
    rewrite emp as bigstar #u2 m n g;
  } else {
    bigstar_extract m n f m;
    stt m;
    bigstar_map' #u1 #u2 #lo #hi #(m+1) #n #f #g stt;
    rewrite bigstar #u1 m m f
         as bigstar #u2 m m g;
    bigstar_compose m n g m;
  }
}
```

```pulse
ghost
fn __bigstar_map
  (#u1 : int)
  (#u2 : int)
  (#m : nat)
  (#n : nat {m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> vprop))
  (#g: (i: nat{m <= i /\ i < n} -> vprop))
  (stt: ((i: nat{m <= i /\ i < n}) -> stt_ghost unit emp_inames
            (f i)
            (fun _ -> g i)))
  requires bigstar #u1 m n f
  ensures  bigstar #u2 m n g
{
  bigstar_map' #u1 #u2 #m #n #m #n #f #g stt;
}
```
let bigstar_map #u1 #u2 = __bigstar_map #u1 #u2

let comb (f g : 'a -> vprop) : 'a -> vprop =
  fun x -> f x ** g x

```pulse
ghost
fn rec bigstar_zip'
    (#u1 #u2 #u3 : int)
    (#lo #hi : nat)
    (m : nat {lo <= m})
    (n : nat {m <= n /\ n <= hi})
    (f: (i: nat{lo <= i /\ i < hi} -> vprop))
    (g: (i: nat{lo <= i /\ i < hi} -> vprop))
  requires  bigstar #u1 m n f ** bigstar #u2 m n g
  ensures   bigstar #u3 m n (comb f g)
  decreases (n-m)
{
  if (n = m) {
    rewrite bigstar #u1 m n f as emp;
    rewrite bigstar #u2 m n g as emp;
    rewrite emp as bigstar #u3 m n (comb f g);
    ()
  } else {
    rewrite bigstar #u1 m n f as f m ** bigstar #u1 (m+1) n f;
    rewrite bigstar #u2 m n g as g m ** bigstar #u2 (m+1) n g;
    bigstar_zip' #_ #_ #u3 #lo #hi (m+1) n f g;
    rewrite (f m ** g m) ** bigstar #u3 (m+1) n (comb f g)
         as bigstar #u3 m n (comb f g);
  }
}
```

```pulse
ghost
fn __bigstar_zip
    (#u1 #u2 #u3 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
  requires bigstar #u1 m n f ** bigstar #u2 m n g
  ensures  bigstar #u3 m n (fun i -> f i ** g i)
{
  bigstar_zip' #u1 #u2 #u3 #m #n m n f g;
}
```
let bigstar_zip #u1 #u2 #u3 = __bigstar_zip #u1 #u2 #u3

```pulse
ghost
fn rec bigstar_unzip'
    (#u1 #u2 #u3 : int)
    (#lo #hi : nat)
    (m : nat {lo <= m})
    (n : nat {m <= n /\ n <= hi})
    (f: (i: nat{lo <= i /\ i < hi} -> vprop))
    (g: (i: nat{lo <= i /\ i < hi} -> vprop))
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
    rewrite bigstar #u3 m n (comb f g)
         as (f m ** g m) ** bigstar #u3 (m+1) n (comb f g);
    bigstar_unzip' #u1 #u2 #u3 #lo #hi (m+1) n f g;
    rewrite f m ** bigstar #u1 (m+1) n f as bigstar #u1 m n f;
    rewrite g m ** bigstar #u2 (m+1) n g as bigstar #u2 m n g;
  }
}
```

```pulse
ghost
fn __bigstar_unzip
    (#u1 #u2 #u3 : int)
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
  requires bigstar #u3 m n (fun i -> f i ** g i)
  ensures  bigstar #u1 m n f ** bigstar #u2 m n g
{
  bigstar_unzip' #u1 #u2 #u3 #m #n m n f g;
}
```
let bigstar_unzip #u1 #u2 #u3 = __bigstar_unzip #u1 #u2 #u3

```pulse
ghost
fn bigstar_extensionality
    (m : nat)
    (n : nat {m <= n})
    (f: (i: nat{m <= i /\ i < n} -> vprop))
    (g: (i: nat{m <= i /\ i < n} -> vprop))
    (h: ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
  requires bigstar m n f
  ensures  bigstar m n g
{
  bigstar_congr m n m n f g (fun j -> h (m+j));
  ();
}
```

```pulse
ghost
fn bigstar_eta
  ()
  (#m : nat) (#n : nat{m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> vprop))
  requires bigstar m n f
  ensures  bigstar m n (fun i -> f i)
{
  bigstar_extensionality m n f (fun i -> f i) (fun _ -> ());
}
```

```pulse
ghost
fn bigstar_uneta
  ()
  (#m : nat) (#n : nat{m <= n})
  (#f: (i: nat{m <= i /\ i < n} -> vprop))
  requires bigstar m n (fun i -> f i)
  ensures  bigstar m n f
{
  bigstar_extensionality m n (fun i -> f i) f (fun _ -> ());
}
```
