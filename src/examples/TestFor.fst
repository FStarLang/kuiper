module TestFor

#lang-pulse

open Kuiper
open Kuiper.For

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

fn test_nested ()
{
  forevery_emp_intro (between 0sz 10sz);
  for_loop 0sz 10sz
    (fun x -> emp)
    (fun x -> emp)
    fn x {
      test ();
    };
  forevery_emp_elim (between 0sz 10sz);
  ()
}

fn test_nested_lit ()
{
  forevery_emp_intro (between 0sz 10sz);
  for_loop 0sz 10sz
    (fun x -> emp)
    (fun x -> emp)
    fn x {
      forevery_emp_intro (between 0sz 20sz);
      for_loop 0sz 20sz
        (fun y -> emp)
        (fun y -> emp)
        g;
      forevery_emp_elim (between 0sz 20sz);
    };
  forevery_emp_elim (between 0sz 10sz);
  ()
}

fn test_nested_lit_shadowed ()
{
  forevery_emp_intro (between 0sz 10sz);
  for_loop 0sz 10sz
    (fun x -> emp)
    (fun x -> emp)
    fn x {
      forevery_emp_intro (between 0sz 20sz);
      for_loop 0sz 20sz
        (fun y -> emp)
        (fun y -> emp)
        (fun _ -> g x); // should use the outer counter, and not shadow the names
      forevery_emp_elim (between 0sz 20sz);
    };
  forevery_emp_elim (between 0sz 10sz);
  ()
}
