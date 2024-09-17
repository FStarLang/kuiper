module GPU.Sized

module SZ = FStar.SizeT

inline_for_extraction
class sized (t:Type) = {
  size : SZ.t;
}
