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
  default: t;
}

inline_for_extraction instance _ : sized U8.t  = { size = 1sz; default = 0uy }
inline_for_extraction instance _ : sized U16.t = { size = 2sz; default = 0us }
inline_for_extraction instance _ : sized U32.t = { size = 4sz; default = 0ul }
inline_for_extraction instance _ : sized U64.t = { size = 8sz; default = 0uL }

inline_for_extraction instance _ : sized I8.t  = { size = 1sz; default = 0y }
inline_for_extraction instance _ : sized I16.t = { size = 2sz; default = 0s }
inline_for_extraction instance _ : sized I32.t = { size = 4sz; default = 0l }
inline_for_extraction instance _ : sized I64.t = { size = 8sz; default = 0L }

inline_for_extraction instance _ : sized F16.t = { size = 2sz; default = F16.zero }
inline_for_extraction instance _ : sized F32.t = { size = 4sz; default = F32.zero }
inline_for_extraction instance _ : sized F64.t = { size = 8sz; default = F64.zero }
