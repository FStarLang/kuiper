module Kuiper.Kernel.Stream
#lang-pulse

open Pulse

[@@FStar.Attributes.custard_extern "cudaStream_t";
   FStar.Attributes.custard_c_header "kuiper.h"]
val stream_t: Type0

val stream_live (s: stream_t) : slprop

[@@FStar.Attributes.custard_extern "KPR_FRESH_STREAM";
   FStar.Attributes.custard_c_header "kuiper.h"]
noextract
fn fresh_stream ()
  returns s:stream_t
  ensures stream_live s

[@@FStar.Attributes.custard_extern "KPR_MUST_stream_destroy";
   FStar.Attributes.custard_c_header "kuiper.h"]
noextract
fn destroy_stream
  (s: stream_t)
  requires stream_live s