module Kuiper.Example.FloatEquality

#lang-pulse

open Kuiper

// Exercise the public, typeclass-dispatched API through extraction.
let ieee_f16 (x y:f16) : bool = ieee_eq x y
let bits_f16 (x y:f16) : bool = bit_eq x y
let eq_f16 (x y:f16) : bool = eq x y

let ieee_bf16 (x y:bf16) : bool = ieee_eq x y
let bits_bf16 (x y:bf16) : bool = bit_eq x y
let eq_bf16 (x y:bf16) : bool = eq x y

let ieee_f32 (x y:f32) : bool = ieee_eq x y
let bits_f32 (x y:f32) : bool = bit_eq x y
let eq_f32 (x y:f32) : bool = eq x y
let mul_zero_f32 (x:f32) : f32 = mul x zero
let add_zero_f32 (x:f32) : f32 = add x zero
let reciprocal_f32 (x:f32) : f32 = div one x

let ieee_f64 (x y:f64) : bool = ieee_eq x y
let bits_f64 (x y:f64) : bool = bit_eq x y
let eq_f64 (x y:f64) : bool = eq x y
let mul_zero_f64 (x:f64) : f64 = mul x zero
let add_zero_f64 (x:f64) : f64 = add x zero
let reciprocal_f64 (x:f64) : f64 = div one x
