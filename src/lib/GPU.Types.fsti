module GPU.Types

module U32 = FStar.UInt32
module U64 = FStar.UInt64
module SZ  = FStar.SizeT

inline_for_extraction
class sized (t:Type) = {
  size : SZ.t;
}

inline_for_extraction
instance _ : sized U32.t = { size = 4sz; }
inline_for_extraction
instance _ : sized U64.t = { size = 8sz; }
inline_for_extraction
instance _ : sized SZ.t  = { size = 8sz; }
