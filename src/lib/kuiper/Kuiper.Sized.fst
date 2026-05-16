module Kuiper.Sized

module SZ = Kuiper.SizeT

inline_for_extraction noextract
class sized (t:Type) = {
  size : SZ.t;
  default: t;
}

inline_for_extraction noextract instance _ : sized UInt8.t  = { size = 1sz; default = 0uy }
inline_for_extraction noextract instance _ : sized UInt16.t = { size = 2sz; default = 0us }
inline_for_extraction noextract instance _ : sized UInt32.t = { size = 4sz; default = 0ul }
inline_for_extraction noextract instance _ : sized UInt64.t = { size = 8sz; default = 0uL }

(* Note, we extract F*'s SizeT into uint32_t. *)
inline_for_extraction noextract instance _ : sized SizeT.t  = { size = 4sz; default = 0sz }
