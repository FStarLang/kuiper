module Kuiper

include FStar.Mul

include Pulse
include Pulse.Lib.BigStar

include FStar.Seq { seq, lseq, cons, empty }

include Kuiper.Epoch
include Kuiper.Assert
include Kuiper.Base
include Kuiper.Ref
include Kuiper.Sized
include Kuiper.Scalars
include Kuiper.Array
include Kuiper.Kernel
include Kuiper.SizeT
include Kuiper.Conditional
include Kuiper.IntAliases
include Kuiper.AtomicOps
include Kuiper.Functions

include Pulse.Lib.GhostReference { ref as gref, pts_to as gref_pts_to }

include Kuiper.Seq.Common { op_At_Bang }

include Kuiper.Len { len }
