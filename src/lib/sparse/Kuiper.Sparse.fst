module Kuiper.Sparse

#lang-pulse
include Kuiper.Sparse.Common
include Kuiper.Sparse.Array
// include Kuiper.Sparse.Array.Iterator
include Kuiper.Sparse.Matrix
// TODO tal vez no esta bueno incluir estos dos porque puede haber coincidencias de nombres
// se pueden incluir qualified?
include Kuiper.Sparse.Array.PtsTo
include Kuiper.Sparse.Matrix.PtsTo