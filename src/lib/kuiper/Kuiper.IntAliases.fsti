module Kuiper.IntAliases


unfold type f32 = Kuiper.Float32.t
unfold type f64 = Kuiper.Float64.t
unfold type float = f32
unfold type double = f64

unfold type u8  = FStar.UInt8.t
unfold type u16 = FStar.UInt16.t
unfold type u32 = FStar.UInt32.t
unfold type u64 = FStar.UInt64.t
unfold type i8  = FStar.Int8.t
unfold type i16 = FStar.Int16.t
unfold type i32 = FStar.Int32.t
unfold type i64 = FStar.Int64.t
unfold type sz  = FStar.SizeT.t
unfold type szp = x:sz{FStar.SizeT.v x > 0}
