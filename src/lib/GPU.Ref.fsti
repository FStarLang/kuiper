module GPU.Ref

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open GPU.Base
open GPU.Types

val gpu_ref (a:Type u#0) : Type u#0

val gpu_pts_to
  (#a:Type u#0)
  (x:gpu_ref a)
  (#[exact (`1.0R)] f : perm)
  (v : a)
: slprop

```pulse
val fn gpu_alloc0
  (#a:Type u#0)
  {| sized a |}
  ()
  requires cpu
  returns  x : gpu_ref a
  ensures  cpu ** (exists* (v:a). gpu_pts_to x #1.0R v)
```

// ```pulse
// val fn gpu_alloc
//   (#a:Type u#0)
//   {| sized a |}
//   (v:a)
//   requires cpu
//   returns  x : gpu_ref a
//   ensures  cpu ** gpu_pts_to x #1.0R v
// ```

```pulse
val fn gpu_free
  (#a:Type u#0)
  (r : gpu_ref a)
  (#v : erased a)
  requires cpu ** gpu_pts_to r #1.0R v
  ensures  cpu
```

```pulse
val fn gpu_read
  (#a:Type u#0)
  (r : gpu_ref a)
  (#f : perm)
  (#v0 : erased a)
  requires gpu ** gpu_pts_to r #f v0
  returns  v : a
  ensures  gpu ** gpu_pts_to r #f v ** pure (v == reveal v0)
```

```pulse
val fn gpu_write
  (#a:Type u#0)
  (r : gpu_ref a)
  (v : a)
  requires gpu ** (exists* v0. gpu_pts_to r #1.0R v0)
  ensures  gpu ** gpu_pts_to r #1.0R v
```

```pulse
val fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (r  : ref a)
  (gr : gpu_ref a)
  (#f : perm)
  (#v : erased a)
  (#gv : erased a)
  requires cpu ** pts_to r #1.0R v ** gpu_pts_to gr #f gv
  ensures  cpu ** pts_to r #1.0R v ** gpu_pts_to gr #f v
```

```pulse
val fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (r  : ref a)
  (gr : gpu_ref a)
  (#f : perm)
  (#v : erased a)
  (#gv : erased a)
  requires cpu ** pts_to r #1.0R v ** gpu_pts_to gr #f gv
  ensures  cpu ** pts_to r #1.0R gv ** gpu_pts_to gr #f gv
```
