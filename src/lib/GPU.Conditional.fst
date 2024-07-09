module GPU.Conditional

open Pulse.Lib.Pervasives
open FStar.Tactics.V2

let if_ (b: bool) (p: slprop): slprop = (cond b p emp)

// INTRODUCTION and ELIMINATION RULES

```pulse
ghost fn if_intro_true (b: bool) (p: slprop)
  requires p ** (pure b)
  ensures  if_ b p
{
  rewrite p as if_ b p;
}
```

```pulse
ghost fn if_intro_false (b: bool) (p: slprop)
  requires emp ** (pure (not b))
  ensures  if_ b p
{
  rewrite emp as if_ b p;
}
```

```pulse
ghost fn if_elim_true (b: bool) (p: slprop)
  requires if_ b p ** (pure b)
  ensures  p
{
  unfold if_ b p;
  rewrite (if b then p else emp) as p;
}
```

```pulse
ghost fn if_elim_false (b: bool) (p: slprop)
  requires if_ b p ** (pure (not b))
  ensures  emp
{
  unfold if_ b p;
  rewrite (if b then p else emp) as emp;
}
```

// SPLIT and JOIN RULES

```pulse
ghost fn case_split (b: bool) (p: slprop)
  requires p
  ensures  if_ b p ** if_ (not b) p
{
  if b {
    if_intro_true b p;
    if_intro_false (not b) p;
  } else {
    if_intro_true (not b) p;
    if_intro_false b p;
  }
}
```

```pulse
ghost fn case_join (b: bool) (p: slprop)
  requires if_ b p ** if_ (not b) p
  ensures  p
{
  if b {
    if_elim_true b p;
    if_elim_false (not b) p;
  } else {
    if_elim_true (not b) p;
    if_elim_false b p;
  
  }
}
```

// COMBINE and SPLIT RULES

```pulse
ghost fn combine (b: bool) (p1 p2: slprop)
  requires if_ b p1 ** if_ b p2
  ensures  if_ b (p1 ** p2)
{
  if b {
    if_elim_true b p1;
    if_elim_true b p2;
    if_intro_true b (p1 ** p2);
  } else {
    if_elim_false b p1;
    if_elim_false b p2;
    if_intro_false b (p1 ** p2);
  }
}
```

```pulse
ghost fn split (b: bool) (p1 p2: slprop)
  requires if_ b (p1 ** p2)
  ensures  if_ b p1 ** if_ b p2
{
  if b {
    if_elim_true b (p1 ** p2);
    if_intro_true b p1;
    if_intro_true b p2;
  } else {
    if_elim_false b (p1 ** p2);
    if_intro_false b p1;
    if_intro_false b p2;
  }
}
```

// MAP

```pulse
ghost fn if_map (#b: bool) (#p #q: slprop) (f: unit -> (stt_ghost unit emp_inames (p) (fun _ -> q)))
  requires if_ b p
  ensures  if_ b q
{
  if b {
    if_elim_true b p;
    f ();
    if_intro_true b q;
  } else {
    if_elim_false b p;
    if_intro_false b q;
  }
}
```

```pulse
ghost fn if_flatten (#b1 #b2: bool) (#p: slprop)
  requires if_ b1 (if_ b2 p)
  ensures  if_ (b1 && b2) p
{
  if b1 {
    if_elim_true b1 (if_ b2 p);
    if b2 {
      if_elim_true b2 p;
      if_intro_true (b1 && b2) p;
    } else {
      if_elim_false b2 p;
      if_intro_false (b1 && b2) p;
    }
  } else {
    if_elim_false b1 (if_ b2 p);
    if_intro_false (b1 && b2) p;
  }
}
```

// REWRITE

```pulse
ghost fn if_rewrite (#b: bool) (#p1 p2: slprop) (#e: (squash b -> squash (p1 == p2)))
  requires if_ b p1
  ensures  if_ b p2
{
  if b {
    e ();
    rewrite each p1 as p2;
  } else {
    ()
  }
}
```
