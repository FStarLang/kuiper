module GPU.Scalars

open GPU.Sized

module U8  = FStar.UInt8
module U16 = FStar.UInt16
module U32 = FStar.UInt32
module U64 = FStar.UInt64
module F32 = FStar.Float
module SZ  = FStar.SizeT

inline_for_extraction instance _ : sized U8.t  = { size = 1sz; }
inline_for_extraction instance _ : sized U16.t = { size = 2sz; }
inline_for_extraction instance _ : sized U32.t = { size = 4sz; }
inline_for_extraction instance _ : sized U64.t = { size = 8sz; }
inline_for_extraction instance _ : sized F32.t = { size = 4sz; }
inline_for_extraction instance _ : sized SZ.t  = { size = 8sz; }
