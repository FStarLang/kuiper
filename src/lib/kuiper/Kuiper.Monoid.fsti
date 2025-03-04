module Kuiper.Monoid

class monoid0 (a:Type) = {
  mzero : a;
  mplus : a -> a -> a;
}

unfold let (++) #a {| monoid0 a |} x y = mplus #a x y

instance monoid0_list (a:Type) : monoid0 (list a) = {
  mzero = [];
  mplus = FStar.List.Tot.Base.append;
}

instance monoid0_seq (a:Type) : monoid0 (Seq.seq a) = {
  mzero = Seq.empty;
  mplus = Seq.append;
}
