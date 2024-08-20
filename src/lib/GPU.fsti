module GPU

include FStar.Mul

include Pulse
include Pulse.Lib.BigStar

include FStar.Seq { seq, cons, empty }

include GPU.Base
include GPU.Ref
include GPU.Sized
include GPU.Scalars
include GPU.Array
include GPU.Kernel
include GPU.SizeT
include GPU.Conditional

include GPU.Float32 { float }
include GPU.Float64 { double }

unfold type f32 = float
unfold type f64 = double
unfold type u8  = FStar.UInt8.t
unfold type u16 = FStar.UInt16.t
unfold type u32 = FStar.UInt32.t
unfold type u64 = FStar.UInt64.t
unfold type i8  = FStar.Int8.t
unfold type i16 = FStar.Int16.t
unfold type i32 = FStar.Int32.t
unfold type i64 = FStar.Int64.t
unfold type sz  = FStar.SizeT.t
