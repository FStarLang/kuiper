module Kuiper.StateMachine

open Kuiper.Common

type st (st0:Type0) = 
  | S of st0
  | Done

noeq
type stm_t = {
  state0 : Type0;
  init  : state0;
  next  : state0 -> st state0;
}

// N steps
let fin_n (n:pos) : stm_t = {
  state0 = natlt n;
  init = 0;
  next = (fun s -> if s = n-1 then Done else S (s + 1));
}

let seq (stm1 stm2 : stm_t) : stm_t = {
  state0 = either stm1.state0 stm2.state0;
  init = Inl (stm1.init);
  next = (fun s ->
    match s with
    | Inl s1 -> (
      match stm1.next s1 with
      | S s1' -> S (Inl s1')
      | Done  -> S (Inr stm2.init)
    )
    | Inr s2 -> (
      match stm2.next s2 with
      | S s2' -> S (Inr s2')
      | Done  -> Done)
  );
}

let repeat (n:pos) (stm0 : stm_t) : stm_t = {
  state0 = (natlt n & stm0.state0);
  init = (0, stm0.init);
  next = (fun s ->
    let (i, s0) = s in
    match stm0.next s0 with
    | S s0' -> S (i, s0')
    | Done  -> 
      if i = n - 1 then Done
      else S (i + 1, stm0.init)
  );
}

type even_odd_t = | Even | Odd

// Infinite
let even_odd_inf : stm_t = {
  state0 = even_odd_t;
  init = Even;
  next = (fun s ->
    match s with
    | Even -> S Odd
    | Odd  -> S Even);
}

// Just two
let even_odd : stm_t = {
  state0 = even_odd_t;
  init = Even;
  next = (fun s ->
    match s with
    | Even -> S Odd
    | Odd  -> Done);
}

// Usual stm_t for barrier in GEMMs
let even_odd_n (n:pos) : stm_t =
  repeat n even_odd
