module TestFor

#lang-pulse

open Kuiper
open Kuiper.For

inline_for_extraction noextract
fn g (x : sz)
{
}

fn test ()
{
  forevery_emp_intro
    (between 0sz 10sz);
  for_loop 0sz 10sz
    (fun x -> emp)
    (fun x -> emp)
    g;
  forevery_emp_elim
    (between 0sz 10sz);
  ()
}
