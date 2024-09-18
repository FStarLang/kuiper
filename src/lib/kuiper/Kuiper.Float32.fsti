module Kuiper.Float32

new
val t : Type0

let float = t

[@@noextract_to "krml"] val zero : t
[@@noextract_to "krml"] val one : t

[@@noextract_to "krml"] val add : t -> t -> t
[@@noextract_to "krml"] val sub : t -> t -> t
[@@noextract_to "krml"] val neg : t -> t
[@@noextract_to "krml"] val mul : t -> t -> t
[@@noextract_to "krml"] val div : t -> t -> t
[@@noextract_to "krml"] val rem : t -> t -> t
