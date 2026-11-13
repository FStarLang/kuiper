module KPlay.Proof.SupportBoundary

module F32 = Kuiper.Float32
module F = Kuiper.Floating
module A = Kuiper.Approximates.Base

(* A fact about the existing abstraction, not about any router execution. *)
inline_for_extraction noextract
let approximation_is_not_a_finiteness_certificate ()
  : Lemma (
      A.v_approximates (F.infinity #F32.t) (A.to_real (F.infinity #F32.t)) /\
      ~(F.is_finite (F.infinity #F32.t)))
  = A.to_real_ok (F.infinity #F32.t)
