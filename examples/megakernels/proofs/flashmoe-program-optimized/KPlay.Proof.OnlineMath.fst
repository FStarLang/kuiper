module KPlay.Proof.OnlineMath

open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor

module S = FStar.Seq
module SM = Kuiper.Spec.Softmax

inline_for_extraction noextract
let exp_sum (#e:pos) (r:lseq real e) (i:nat{i <= e}) : real =
  rsum (seq_map exp (S.slice r 0 i))

inline_for_extraction noextract
let denominator (#e:pos) (r:lseq real e) (i:nat{i <= e}) (m:f32) : real =
  exp_sum r i /. exp (to_real m)

inline_for_extraction noextract
let exp_sum_step (#e:pos) (r:lseq real e) (i:nat{i < e})
  : Lemma (exp_sum r (i + 1) == exp_sum r i +. exp (S.index r i))
  = assert (S.equal (S.slice r 0 (i + 1))
      (S.append (S.slice r 0 i) (S.create 1 (S.index r i))));
    assert (S.equal (seq_map exp (S.slice r 0 (i + 1)))
      (S.append (seq_map exp (S.slice r 0 i))
        (S.create 1 (exp (S.index r i)))));
    rsum_append (seq_map exp (S.slice r 0 i))
      (S.create 1 (exp (S.index r i)))

inline_for_extraction noextract
let field_step (s x old_m new_m:real)
  : Lemma (
      (s /. exp old_m) *. exp (old_m -. new_m) +. exp (x -. new_m)
      == (s +. exp x) /. exp new_m)
  = exp_sub old_m new_m;
    exp_sub x new_m

inline_for_extraction noextract
let denominator_step (#e:pos) (r:lseq real e) (i:nat{i < e}) (m m':f32)
  : Lemma (
      denominator r i m *. exp (to_real m -. to_real m') +.
        exp (S.index r i -. to_real m')
      == denominator r (i + 1) m')
  = exp_sum_step r i;
    field_step (exp_sum r i) (S.index r i) (to_real m) (to_real m')

inline_for_extraction noextract
let denominator_positive (#e:pos) (r:lseq real e) (m:f32)
  : Lemma (denominator r e m >. 0.0R)
  = assert (S.equal (S.slice r 0 e) r);
    let _ : real = rsum (seq_map exp r) in
    ()

inline_for_extraction noextract
let normalized_value (#e:pos) (r:lseq real e) (m:f32) (i:nat{i < e})
  : Lemma (
      exp (S.index r i -. to_real m) /. denominator r e m
      == S.index (SM.softmax_real_seq r) i)
  = denominator_positive r m;
    assert (S.equal (S.slice r 0 e) r);
    exp_sub (S.index r i) (to_real m)

inline_for_extraction noextract
let denominator_step_all (#e:pos) (i:nat{i < e}) (m m':f32)
  : Lemma (forall (r:lseq real e).
      denominator r i m *. exp (to_real m -. to_real m') +.
        exp (S.index r i -. to_real m')
      == denominator r (i + 1) m')
  = let proof (r:lseq real e)
      : Lemma (
          denominator r i m *. exp (to_real m -. to_real m') +.
            exp (S.index r i -. to_real m')
          == denominator r (i + 1) m')
      = denominator_step r i m m'
    in
    FStar.Classical.forall_intro proof

inline_for_extraction noextract
let normalized_value_all (#e:pos) (m:f32) (i:nat{i < e})
  : Lemma (forall (r:lseq real e).
      denominator r e m >. 0.0R /\
      exp (S.index r i -. to_real m) /. denominator r e m
        == S.index (SM.softmax_real_seq r) i)
  = let proof (r:lseq real e)
      : Lemma (
          denominator r e m >. 0.0R /\
          exp (S.index r i -. to_real m) /. denominator r e m
            == S.index (SM.softmax_real_seq r) i)
      = denominator_positive r m;
        normalized_value r m i
    in
    FStar.Classical.forall_intro proof

