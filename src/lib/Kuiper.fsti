module Kuiper

include FStar.Mul

include FStar.FunctionalExtensionality { (^->>), (^->) }

include FStar.SizeT { (/^), (%^), (+^), (-^), ( *^ )  }

include Pulse
include Pulse.Lib.BigStar
include Pulse.Lib.GhostReference { ref as gref, pts_to as gref_pts_to }
include Pulse.Lib.Vec { vec, op_Array_Assignment, op_Array_Access }

include FStar.Seq { seq, lseq, cons, empty }

include Kuiper.ForEvery
include Kuiper.Common
include Kuiper.Epoch
include Kuiper.Assert
include Kuiper.Base
include Kuiper.Ref
include Kuiper.Sized
include Kuiper.Scalars
include Kuiper.Array
include Kuiper.Kernel
include Kuiper.SHMem
include Kuiper.SizeT
include Kuiper.Conditional
include Kuiper.IntAliases
include Kuiper.AtomicOps
include Kuiper.Functions
include Kuiper.Seq.Common { op_At_Bang }
include Kuiper.Len { len }
include Kuiper.Divides
include Kuiper.PtsTo
include Kuiper.Enumerable { enumerable }

[@@coercion; pulse_unfold]
unfold let kpr_box_to_ref  (#a:Type0) (b:Pulse.Lib.Box.box a) : Pulse.Lib.Reference.ref a = Pulse.Lib.Box.box_to_ref b

[@@coercion; pulse_unfold]
unfold let i2r (i:int) : real = Real.of_int i

[@@coercion; pulse_unfold]
unfold let ei2r (i:erased int) : real = Real.of_int i

[@@coercion; pulse_unfold]
unfold let en2r (i:erased nat) : real = Real.of_int i

[@@coercion; pulse_unfold]
unfold let sz2r (i:sz) : real = Real.of_int i
