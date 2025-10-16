module Kuiper.Sized

module SZ = Kuiper.SizeT

module U8  = FStar.UInt8
module U16 = FStar.UInt16
module U32 = FStar.UInt32
module U64 = FStar.UInt64

module I8  = FStar.Int8
module I16 = FStar.Int16
module I32 = FStar.Int32
module I64 = FStar.Int64

module F16 = Kuiper.Float16
module F32 = Kuiper.Float32
module F64 = Kuiper.Float64

inline_for_extraction
class sized (t:Type) = {
  size : SZ.t;
}

inline_for_extraction instance _ : sized U8.t  = { size = 1sz; }
inline_for_extraction instance _ : sized U16.t = { size = 2sz; }
inline_for_extraction instance _ : sized U32.t = { size = 4sz; }
inline_for_extraction instance _ : sized U64.t = { size = 8sz; }

inline_for_extraction instance _ : sized I8.t  = { size = 1sz; }
inline_for_extraction instance _ : sized I16.t = { size = 2sz; }
inline_for_extraction instance _ : sized I32.t = { size = 4sz; }
inline_for_extraction instance _ : sized I64.t = { size = 8sz; }

inline_for_extraction instance _ : sized F16.t = { size = 2sz; }
inline_for_extraction instance _ : sized F32.t = { size = 4sz; }
inline_for_extraction instance _ : sized F64.t = { size = 8sz; }
