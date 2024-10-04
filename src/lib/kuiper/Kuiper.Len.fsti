module Kuiper.Len

class has_len (a:Type) = {
  len : a -> GTot nat;
}

(* These have to be exposed if we want to use them in specs. 
Ideally we could mark this module SMT-transparent. *)

instance has_len_list (a:Type) : has_len (list a) = {
  len = List.length;
}

instance has_len_seq (a:Type) : has_len (Seq.seq a) = {
  len = Seq.length; (* Why don't I have to eta these two...? I guess it's fine since it's a primitive effect? *)
}

instance has_len_erased (a:Type) (_ : has_len a) : has_len (Ghost.erased a) = {
  len = (fun x -> len (Ghost.reveal x));
}
