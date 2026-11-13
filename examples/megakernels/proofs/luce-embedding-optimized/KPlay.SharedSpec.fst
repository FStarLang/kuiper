module KPlay.SharedSpec

module Seq = FStar.Seq.Base
module R = FStar.Real

(* Qwen3_5TextModel.embed_tokens on decode_reference's input_ids = [[t]].
   The supplied text_config fixes these logical dimensions, not a layout. *)
inline_for_extraction noextract
let vocab_size : pos = 248320

inline_for_extraction noextract
let hidden_size : pos = 1024

inline_for_extraction noextract
let token_id = t:nat{t < vocab_size}

inline_for_extraction noextract
let hidden_index = d:nat{d < hidden_size}

inline_for_extraction noextract
let hidden_vector (a:Type0) = h:Seq.seq a{Seq.length h == hidden_size}

inline_for_extraction noextract
let embedding_weights (a:Type0) =
  e:Seq.seq (hidden_vector a){Seq.length e == vocab_size}

(* Pure selection has no position/cache state and performs no scalar operation.
   Keeping cells parametric does not exclude exceptional BF16 representations. *)
inline_for_extraction noextract
let embedding
  (#a:Type0)
  (e:embedding_weights a)
  (t:token_id)
  : Tot (hidden_vector a)
  = Seq.index e t

inline_for_extraction noextract
let embedding_post
  (#a:Type0)
  (e:embedding_weights a)
  (t:token_id)
  (h:hidden_vector a)
  : prop
  = forall (d:hidden_index).
      Seq.index h d == Seq.index (Seq.index e t) d

(* The algebraic view assumes a role-supplied cellwise modeling relation to e.
   Real equality is not IEEE equality or an assertion of bitwise equivalence. *)
inline_for_extraction noextract
let algebraic_embedding
  (e:embedding_weights R.real)
  (t:token_id)
  : Tot (hidden_vector R.real)
  = embedding e t

inline_for_extraction noextract
let algebraic_post
  (e:embedding_weights R.real)
  (t:token_id)
  (h:hidden_vector R.real)
  : prop
  = embedding_post e t h

inline_for_extraction noextract
let embedding_correct
  (#a:Type0)
  (e:embedding_weights a)
  (t:token_id)
  : Lemma (embedding_post e t (embedding e t))
  = ()

inline_for_extraction noextract
let embedding_post_unique
  (#a:Type0)
  (e:embedding_weights a)
  (t:token_id)
  (h:hidden_vector a)
  : Lemma
      (requires embedding_post e t h)
      (ensures h == embedding e t)
  = Seq.lemma_eq_intro h (embedding e t);
    Seq.lemma_eq_elim h (embedding e t)

(* Instantiate rel with the library's v_approximates for the algebraic view.
   This transport uses only selection; it assumes no laws about rel. *)
inline_for_extraction noextract
let embedding_transports_relation
  (#a:Type0)
  (#b:Type0)
  (rel:a -> b -> prop)
  (e:embedding_weights a)
  (r:embedding_weights b)
  (t:token_id)
  : Lemma
      (requires
        forall (i:token_id) (d:hidden_index).
          rel (Seq.index (Seq.index e i) d)
              (Seq.index (Seq.index r i) d))
      (ensures
        forall (d:hidden_index).
          rel (Seq.index (embedding e t) d)
              (Seq.index (embedding r t) d))
  = ()
